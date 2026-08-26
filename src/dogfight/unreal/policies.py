from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from dogfight.ai.action_provider import ActionProvider
from dogfight.ai.native_bt import AIPilot
from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext
from dogfight.envs.observation import build_observation
from dogfight.sim.state_schema import StateIndex
from dogfight.unreal.client import RemoteClientContext
from dogfight.unreal.protocol import CMD



@dataclass
class SafetyOverrideConfig:
    """Same thresholds/semantics as single_agent_env.py's
    _apply_safety_override (2026-08-11), ported to the live Unreal UDP
    path. min_altitude_m is the actual crash floor (competition rule:
    1000ft/304.8m) used for the predictive time-to-impact estimate;
    hard_floor_m is a slightly higher unconditional backstop.
    """

    enabled: bool = True
    altitude_m: float = 500.0
    pitch_deg: float = -5.0
    roll_deg: float = 45.0
    time_horizon_s: float = 15.0
    hard_floor_m: float = 400.0
    min_altitude_m: float = 304.8


class SafetyOverrideCommandPolicy:
    """Wraps any CommandPolicy with the hard, non-learned recovery override
    verified in training/frozen-eval (single_agent_env.py._apply_safety_
    override, 2026-08-11: crash 10-25% -> 0%, wez_episode_rate improved,
    on stage6e_delta_down_400iter -- this project's best-verified safety
    fix, but only ever wired into our own JSBSim env, never the live
    Unreal inference path used for actual competition connection). The
    wrapped policy never sees this -- it's a pure post-hoc clamp on the
    outgoing CMD, applied every compute_command call regardless of the
    inner policy's own action_repeat caching, so a dangerous descent is
    always caught even if the inner policy is mid-repeat-hold.

    Sign conventions (roll_cmd/pitch_cmd) match single_agent_env.py's
    verified values exactly, since DogFightViewer's server embeds the same
    JSBSimAIP physics core (see BattleServer package's
    Plugins/JSBSimAIP) as this project's own training/eval simulator --
    action[1]=+1 pitches nose DOWN (pull-up is pitch_cmd=-1.0),
    action[0]=+1 rolls right (wings-level correction is -sign(roll)).
    """

    def __init__(self, inner: "CommandPolicy", config: SafetyOverrideConfig | None = None):
        self.inner = inner
        self.config = config or SafetyOverrideConfig()
        self._prev_altitude: float | None = None
        self._prev_time: float | None = None
        self.override_active_count = 0

    def reset(self, context: RemoteClientContext) -> None:
        self.inner.reset(context)
        self._prev_altitude = None
        self._prev_time = None

    def compute_command(self, context: RemoteClientContext) -> CMD:
        cmd = self.inner.compute_command(context)
        cfg = self.config
        if not cfg.enabled or context.own_plane.plane_info is None:
            return cmd

        own_plane = context.own_plane.plane_info
        # position.z IS altitude already (up-positive, meters) -- empirically
        # confirmed 2026-08-19 (see plane_info_to_state()). The previous
        # `-position.z` here reported e.g. -4572m instead of +4572m at a
        # 15000ft cruise start, so hard_floor_breach (altitude < 400m) was
        # true on almost every frame, permanently forcing pitch_cmd=-1.0
        # (climb) regardless of the inner policy's action -- this is exactly
        # the "safety 때문에 계속 위로만 상승" failure mode reported.
        altitude = float(own_plane.position.z)
        pitch = float(own_plane.rotation.pitch)
        roll = float(own_plane.rotation.roll)

        now = time.time()
        vertical_rate = None
        if self._prev_altitude is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.0:
                vertical_rate = (altitude - self._prev_altitude) / dt
        self._prev_altitude = altitude
        self._prev_time = now

        hard_floor_breach = altitude < cfg.hard_floor_m
        fixed_trigger = altitude < cfg.altitude_m and pitch <= cfg.pitch_deg
        predictive_trigger = False
        if vertical_rate is not None and vertical_rate < 0.0:
            time_to_impact = (altitude - cfg.min_altitude_m) / max(1.0, -vertical_rate)
            predictive_trigger = time_to_impact < cfg.time_horizon_s

        if not (hard_floor_breach or fixed_trigger or predictive_trigger):
            return cmd

        self.override_active_count += 1
        roll_cmd = cmd.roll_cmd
        if abs(roll) > cfg.roll_deg:
            roll_cmd = -1.0 if roll > 0 else 1.0
        return CMD(
            plane_id=cmd.plane_id,
            index=cmd.index,
            roll_cmd=roll_cmd,
            pitch_cmd=-1.0,
            yaw_cmd=cmd.yaw_cmd,
            throttle_cmd=1.0,
        )


@dataclass
class ConstantCommandPolicy:
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    yaw_cmd: float = 0.0
    throttle_cmd: float = 1.0

    def reset(self, context: RemoteClientContext) -> None:
        return None

    def compute_command(self, context: RemoteClientContext) -> CMD:
        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=self.roll_cmd,
            pitch_cmd=self.pitch_cmd,
            yaw_cmd=self.yaw_cmd,
            throttle_cmd=self.throttle_cmd,
        )


class RLLightweightCommandPolicy:
    def __init__(
        self,
        action_provider: RLActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.geometry = GeometryInfo()

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        ownship_state = plane_info_to_state(context.own_plane.plane_info)
        target_state = plane_info_to_state(context.enemy_plane.plane_info)
        observation = self._build_observation(ownship_state, target_state)

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={"frame_index": context.frame_index},
            )
        )
        action = np.asarray(action_result.action, dtype=np.float32)

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _build_observation(self, ownship_state, target_state) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
        )


class ProviderCommandPolicy:
    def __init__(
        self,
        action_provider: ActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
        ownship_force_side: int = 1,
        target_force_side: int = 2,
        action_repeat: int = 1,
        debug_action_repeat: bool = False,
        debug_raw_state_frames: int = 0,
        log_csv_path: str | None = None,
        action_rate_limit: float | None = None,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.ownship_force_side = ownship_force_side
        self.target_force_side = target_force_side
        self.action_repeat = max(1, int(action_repeat))
        self.debug_action_repeat = debug_action_repeat
        self.debug_raw_state_frames = int(debug_raw_state_frames)
        self.geometry = GeometryInfo()
        self._state_pair_count = 0
        self._cached_action: np.ndarray | None = None
        self._last_policy_count: int | None = None
        self._last_policy_frame_index: int | None = None
        self._raw_debug_printed = 0
        # Must match single_agent_env.py's _apply_action_rate_limit exactly
        # (same per-step |delta| cap on roll/pitch/yaw only, throttle
        # unrestricted) -- if a checkpoint was trained with this env-level
        # clamp on, physics only ever saw smoothed commands during training,
        # so live inference must apply the identical clamp or the real
        # aircraft receives control input the policy never actually learned
        # under.
        self.action_rate_limit = action_rate_limit
        self._prev_applied_action: np.ndarray | None = None
        # Per-frame CSV telemetry log (2026-08-20), added so a live
        # DogFightViewer run's exact trajectory/geometry/action numbers can
        # be inspected after the fact -- much more precise than eyeballing
        # a screen recording, and this env has no video-reading capability
        # anyway. One row per real policy decision (not per repeated-action
        # frame). Written incrementally (not buffered) so a crash/disconnect
        # doesn't lose the log.
        self._log_csv_path = log_csv_path
        self._log_csv_file = None
        self._log_csv_writer = None
        if log_csv_path:
            import csv as _csv
            self._log_csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
            self._log_csv_writer = _csv.writer(self._log_csv_file)
            self._log_csv_writer.writerow([
                "frame_index", "sim_time_s", "own_n", "own_e", "own_alt_m", "own_roll_deg", "own_pitch_deg", "own_yaw_deg",
                "own_speed_mps", "own_vn_mps", "own_ve_mps", "own_vz_mps",
                "enemy_n", "enemy_e", "enemy_alt_m", "enemy_speed_mps",
                "enemy_vn_mps", "enemy_ve_mps", "enemy_vz_mps",
                "enemy_roll_deg", "enemy_pitch_deg", "enemy_yaw_deg",
                "distance_m", "ata_deg", "aa_deg",
                "roll_cmd", "pitch_cmd", "yaw_cmd", "throttle_cmd",
                # Populated by supervisory/pursuit ActionProviders through
                # ActionResult.info; blank for a raw learned policy.
                "tactical_state", "closure_rate_mps",
                "los_az_deg", "los_el_deg", "target_bank_deg", "bank_error_deg",
                # Added 2026-08-21 (pulse-test follow-up, user request):
                # measured directly instead of post-hoc finite-differencing
                # a plain angle log, since angles wrap and naive diffs are
                # wrong across a +-180/0-360 boundary. flight_path_angle_deg
                # is asin(vertical_speed/own_speed) -- the velocity-vector
                # angle, not Euler pitch, per the user's diagnosis that
                # Euler pitch alone doesn't reflect actual climb/dive.
                "roll_rate_degps", "pitch_rate_degps", "yaw_rate_degps",
                "vertical_speed_mps", "flight_path_angle_deg",
                # W2 coordinated-turn feed-forward diagnostics.  Blank for
                # providers that do not publish these ActionResult fields.
                "turn_pitch_feedforward", "desired_gamma_deg", "gamma_error_deg",
                "pitch_soft_limited", "physical_pull_limited",
                "pull_energy_scale", "pull_geometry_scale",
                "opening_pull_boost_active",
                # Integrated BFM manager/guidance diagnostics (W14+).
                "threat_ata_deg", "aim_az_deg", "aim_el_deg",
                "advantage_score", "advantage_ttc_s", "advantage_manager_reason",
                "los_rate_degps", "ata_rate_degps",
                "own_yaw_rate_degps", "target_yaw_rate_degps",
                "intercept_horizon_s", "desired_turn_rate_degps",
                "target_throttle", "speed_error_mps",
                "target_yaw_accel_degps2", "target_turn_stable_s",
                "prediction_stable", "lag_pursuit_active", "lag_offset_m",
                "lag_pursuit_scale",
                "lag_energy_preserve_active",
                "world_elevation_deg", "world_elevation_rate_degps",
                "target_climb_rate_mps",
                "target_vertical_accel_mps2", "target_vertical_stable_s",
                "vertical_prediction_horizon_s", "vertical_prediction_stable",
                "vertical_bank_relief_active", "defensive_escape_active",
                "defensive_escape_sign", "bank_reversal_active",
                "rear_rate_match_active", "engagement_mode",
                "overshoot_control_active", "vertical_follow_active",
                "energy_deficit", "integrated_rate_bank_active",
                "energy_gamma_guard_active", "horizontal_energy_hold_active",
                "mutual_commit_active",
                "predictive_guidance_active", "predictive_horizon_s",
                "predictive_score", "predictive_future_error_deg",
                "predictive_future_threat_error_deg",
                "predictive_worst_target_rate_degps",
                "predictive_sign_guard_active",
                "rule_desired_turn_rate_degps",
                "predictive_rule_score", "predictive_score_gain",
                "predictive_override_active", "spiral_recovery_active",
                "rate_deficit_overbank_active",
                "sequential_maneuver_phase",
                "high_bank_speed_control_active",
                "high_bank_desired_speed_mps",
                "turn_match_active",
                "turn_match_candidate_s",
                "vertical_alignment_active",
                "energy_recovery_active",
                "post_defense_conversion_active",
                "post_defense_conversion_armed",
                "conversion_rear_offset_m",
                "conversion_lateral_offset_m",
                "terminal_track_active",
                "terminal_vertical_unload_active",
                "attack_conversion_active",
                "headon_deconflict_active",
                "lift_vector_active", "lift_vector_target_bank_deg",
                "lift_vector_target_gamma_deg", "lift_vector_los_rate_degps",
                "lift_vector_accel_mps2", "lift_vector_authority",
                "lift_vector_conflict", "first_merge_passed",
                "formula_vpp_active", "formula_vpp_blend",
                "formula_vpp_mode", "formula_vpp_turn_rate_degps",
                "formula_vpp_t_cpa_s", "formula_vpp_d_cpa_m",
                "formula_vpp_turn_circle_active",
                "formula_vpp_pass_detected",
                "formula_vpp_recommit_active",
                "formula_vpp_recommit_remaining_s",
                "formula_projectile_tof_s", "formula_apg_weight",
                "formula_energy_gamma_correction",
                "formula_zem_defense_active", "formula_zem_escape_sign",
                "formula_zem_burst_remaining_s",
                "formula_zem_cooldown_remaining_s",
                "formula_rate_bank_active",
                "planner3d_active", "planner3d_replanned",
                "planner3d_manoeuvre", "planner3d_score",
                "planner3d_horizon_s",
                "planner3d_threat_pressure",
                "planner3d_turn_rate_degps", "planner3d_gamma_deg",
                "planner3d_target_speed_mps", "planner3d_future_ata_deg",
                "planner3d_future_threat_ata_deg", "planner3d_future_range_m",
                "planner3d_worst_target_turn_degps",
                "planner3d_worst_target_gamma_deg",
                # W56 residual-RL diagnostics. Blank for rule-only modes.
                "residual_gate_active", "residual_gate_block_reason",
                "residual_raw_roll", "residual_raw_pitch", "residual_raw_throttle",
                "residual_scaled_roll", "residual_scaled_pitch", "residual_scaled_throttle",
                "rule_roll_cmd", "rule_pitch_cmd", "rule_yaw_cmd", "rule_throttle_cmd",
                "provider_roll_cmd", "provider_pitch_cmd", "provider_yaw_cmd", "provider_throttle_cmd",
                "residual_bundle_id", "residual_safety_active",
                # Explicit-phase controller diagnostics (EP1+).
                "engagement_phase", "bank_owner", "pitch_owner", "throttle_owner",
                "bank_authority", "pitch_authority", "throttle_authority",
                "bank_proposal_count", "pitch_proposal_count", "throttle_proposal_count",
                "proposal_conflict", "proposal_saturated",
                "active_bank_proposals", "active_pitch_proposals", "active_throttle_proposals",
                "spawn_yaw_class",
                "explicit_bank_override_active", "explicit_commit_sign",
                "explicit_commit_age_s", "explicit_reversal_candidate",
                "explicit_reversal_candidate_age_s",
                "explicit_initial_candidate", "explicit_initial_candidate_age_s",
                "explicit_opening_pull_active", "explicit_opening_pull_cmd",
                # 2026-08-25: terminal_stabilizer.py's roll-commit filter and
                # energy-throttle floor -- blank for providers that don't
                # publish these ActionResult.info fields.
                "roll_commit_active", "roll_commit_sign", "energy_floor_active",
                "energy_unload_active", "energy_pitch_scale",
                # W114 measured-envelope diagnostics.
                "measured_envelope_active", "measured_target_speed_mps",
                "measured_pursuit_speed_margin_mps",
                "measured_speed_error_mps", "measured_opening_fill_active",
                "measured_opening_fill_cmd", "measured_energy_unload_active",
                "measured_pitch_floor", "measured_safe_terminal",
                "measured_raw_throttle",
                "measured_opening_roll_active", "measured_bank_hold_active",
                "measured_roll_schedule_cmd",
                "measured_enemy_speed_mps", "measured_speed_recovery_active",
                "measured_immediate_threat", "measured_energy_bank_active",
                "measured_energy_bank_target_deg",
                "measured_rear_trap", "measured_rear_trap_age_s",
                "measured_vertical_phase", "measured_vertical_phase_age_s",
                "measured_vertical_pitch_cmd", "measured_vertical_escape_active",
                "measured_counter_turn_sign", "measured_counter_min_distance_m",
                "measured_counter_overshoot_detected", "measured_counter_used",
                "measured_sustained_turn_cut",
                "measured_vertical_speed_ema_mps", "measured_altitude_loss_m",
                "measured_vertical_recovery_active",
                "measured_vertical_recovery_bank_deg",
                "measured_los_aim_delta_deg", "measured_los_disagree_age_s",
                "measured_los_reacquire_active", "measured_los_reacquire_sign",
                "measured_los_reacquire_used",
                "measured_terminal_lift_active", "measured_terminal_lift_age_s",
                "measured_terminal_lift_bank_deg",
                "measured_rear_capture_active", "measured_rear_capture_az_deg",
                "measured_rear_capture_offset_m", "measured_speed_advantage_mps",
                "distance_handoff_active", "distance_handoff_near_latched",
                "distance_handoff_rl_weight", "distance_handoff_distance_m",
                "distance_handoff_rule_roll", "distance_handoff_rule_pitch",
                "distance_handoff_rule_yaw", "distance_handoff_rule_throttle",
                "distance_handoff_rl_roll", "distance_handoff_rl_pitch",
                "distance_handoff_rl_yaw", "distance_handoff_rl_throttle",
                "max_rate_turn_active", "max_rate_turn_commit_sign",
                "max_rate_turn_matched_target",
                "max_rate_turn_target_yaw_rate_degps",
                "max_rate_turn_target_bank_deg", "max_rate_turn_target_speed_mps",
                "postmerge_pull_boost_active", "postmerge_pull_boost_age_s",
                "postmerge_pull_boost_cmd", "postmerge_pull_boost_duration_s",
            ])
        self._prev_log_time: float | None = None
        self._prev_log_roll: float | None = None
        self._prev_log_pitch: float | None = None
        self._prev_log_yaw: float | None = None
        self._prev_log_alt: float | None = None
        self._spawn_yaw_class: str | None = None

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)
        self._state_pair_count = 0
        self._cached_action = None
        self._last_policy_count = None
        self._last_policy_frame_index = None
        self._prev_applied_action = None
        self._prev_log_time = None
        self._prev_log_roll = None
        self._prev_log_pitch = None
        self._prev_log_yaw = None
        self._prev_log_alt = None
        self._spawn_yaw_class = None

    def _apply_action_rate_limit(self, action: np.ndarray) -> np.ndarray:
        """Mirrors single_agent_env.py's _apply_action_rate_limit exactly --
        see that method's docstring."""
        if self.action_rate_limit is None:
            self._prev_applied_action = action.copy()
            return action
        limit = float(self.action_rate_limit)
        if self._prev_applied_action is None:
            limited = action.copy()
        else:
            limited = action.copy()
            delta = np.clip(
                action[:3] - self._prev_applied_action[:3], -limit, limit
            )
            limited[:3] = np.clip(self._prev_applied_action[:3] + delta, -1.0, 1.0)
        self._prev_applied_action = limited.copy()
        return limited

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        own_plane = context.own_plane.plane_info
        enemy_plane = context.enemy_plane.plane_info
        pair_count = self._state_pair_count
        self._state_pair_count += 1

        if self._raw_debug_printed < self.debug_raw_state_frames:
            self._raw_debug_printed += 1
            print(
                "[RAW_STATE] "
                f"own.pos=({own_plane.position.x:.1f},{own_plane.position.y:.1f},{own_plane.position.z:.1f}) "
                f"own.rot(r,p,y)=({own_plane.rotation.roll:.1f},{own_plane.rotation.pitch:.1f},{own_plane.rotation.yaw:.1f}) "
                f"own.vel=({own_plane.velocity.x:.1f},{own_plane.velocity.y:.1f},{own_plane.velocity.z:.1f}) | "
                f"enemy.pos=({enemy_plane.position.x:.1f},{enemy_plane.position.y:.1f},{enemy_plane.position.z:.1f}) "
                f"enemy.rot(r,p,y)=({enemy_plane.rotation.roll:.1f},{enemy_plane.rotation.pitch:.1f},{enemy_plane.rotation.yaw:.1f}) "
                f"enemy.vel=({enemy_plane.velocity.x:.1f},{enemy_plane.velocity.y:.1f},{enemy_plane.velocity.z:.1f})"
            )

        policy_updated = (
            self._cached_action is None
            or pair_count % self.action_repeat == 0
        )
        if policy_updated:
            action = self._compute_provider_action(context, own_plane, enemy_plane)
            self._cached_action = action
            self._last_policy_count = pair_count
            self._last_policy_frame_index = context.frame_index
        else:
            action = np.asarray(self._cached_action, dtype=np.float32)

        if self.debug_action_repeat:
            self._print_action_repeat_debug(
                context=context,
                pair_count=pair_count,
                policy_updated=policy_updated,
                action=action,
            )

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _compute_provider_action(self, context, own_plane, enemy_plane) -> np.ndarray:
        ownship_state = plane_info_to_state(own_plane)
        target_state = plane_info_to_state(enemy_plane)
        observation = self._build_observation(ownship_state, target_state)

        own_speed = float(
            np.linalg.norm([own_plane.velocity.x, own_plane.velocity.y, own_plane.velocity.z])
        )
        target_speed = float(
            np.linalg.norm([enemy_plane.velocity.x, enemy_plane.velocity.y, enemy_plane.velocity.z])
        )
        my_plane_data = AIPilot.BuildPlaneData(
            [own_plane.position.x, own_plane.position.y, own_plane.position.z],
            [own_plane.rotation.roll, own_plane.rotation.pitch, own_plane.rotation.yaw],
            own_speed,
            self.ownship_force_side,
        )
        target_plane_data = AIPilot.BuildPlaneData(
            [enemy_plane.position.x, enemy_plane.position.y, enemy_plane.position.z],
            [enemy_plane.rotation.roll, enemy_plane.rotation.pitch, enemy_plane.rotation.yaw],
            target_speed,
            self.target_force_side,
        )

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={
                    "frame_index": context.frame_index,
                    "my_plane_id": context.plane_id,
                    "target_plane_id": enemy_plane.plane_id,
                    "my_force_side": self.ownship_force_side,
                    "target_force_side": self.target_force_side,
                    "my_plane_data": my_plane_data,
                    "target_plane_data": target_plane_data,
                },
            )
        )
        action = np.asarray(action_result.action, dtype=np.float32)
        action = self._apply_action_rate_limit(action)

        if self._log_csv_writer is not None:
            distance = self.geometry._get_distance(ownship_state, target_state)
            ata = self.geometry._get_antenna_train_angle(ownship_state, target_state, False)
            aa = self.geometry._get_aspect_angle(ownship_state, target_state, False)

            # Wrap-safe angular rates (roll/yaw can cross +-180; a naive
            # diff is wrong across that boundary) plus vertical
            # speed/flight-path angle, computed here (not post-hoc) so a
            # short pulse test doesn't need re-deriving these from raw
            # angle columns every time -- see the header comment above.
            now_t = float(context.frame_index) / 60.0
            roll_rate = pitch_rate = yaw_rate = vertical_speed = 0.0
            if self._prev_log_time is not None:
                dt = now_t - self._prev_log_time
                if dt > 1e-3:
                    def _wrap180(a: float) -> float:
                        return (a + 180.0) % 360.0 - 180.0
                    roll_rate = _wrap180(own_plane.rotation.roll - self._prev_log_roll) / dt
                    pitch_rate = _wrap180(own_plane.rotation.pitch - self._prev_log_pitch) / dt
                    yaw_rate = _wrap180(own_plane.rotation.yaw - self._prev_log_yaw) / dt
                    vertical_speed = (own_plane.position.z - self._prev_log_alt) / dt
            flight_path_angle = float(np.degrees(np.arcsin(
                np.clip(vertical_speed / own_speed, -1.0, 1.0) if own_speed > 1e-3 else 0.0
            )))
            self._prev_log_time = now_t
            self._prev_log_roll = own_plane.rotation.roll
            self._prev_log_pitch = own_plane.rotation.pitch
            self._prev_log_yaw = own_plane.rotation.yaw
            self._prev_log_alt = own_plane.position.z
            if self._spawn_yaw_class is None:
                spawn_yaw = (float(own_plane.rotation.yaw) + 180.0) % 360.0 - 180.0
                self._spawn_yaw_class = "yawp090" if spawn_yaw >= 0.0 else "yawm090"

            raw_residual = action_result.info.get("raw_residual", ["", "", ""])
            scaled_residual = action_result.info.get("scaled_residual", ["", "", ""])
            rule_action = action_result.info.get("rule_action", ["", "", "", ""])
            provider_action = action_result.info.get("final_action", ["", "", "", ""])

            self._log_csv_writer.writerow([
                context.frame_index,
                now_t,
                own_plane.position.x, own_plane.position.y, own_plane.position.z,
                own_plane.rotation.roll, own_plane.rotation.pitch, own_plane.rotation.yaw,
                own_speed,
                own_plane.velocity.x, own_plane.velocity.y, own_plane.velocity.z,
                enemy_plane.position.x, enemy_plane.position.y, enemy_plane.position.z,
                target_speed,
                enemy_plane.velocity.x, enemy_plane.velocity.y, enemy_plane.velocity.z,
                enemy_plane.rotation.roll, enemy_plane.rotation.pitch, enemy_plane.rotation.yaw,
                float(distance), float(ata), float(aa),
                float(action[0]), float(action[1]), float(action[2]), float(action[3]),
                action_result.info.get("state", ""),
                action_result.info.get("closure_rate", ""),
                action_result.info.get("los_az", ""),
                action_result.info.get("los_el", ""),
                action_result.info.get("target_bank", ""),
                action_result.info.get("bank_error", ""),
                roll_rate, pitch_rate, yaw_rate, vertical_speed, flight_path_angle,
                action_result.info.get("turn_pitch_feedforward", ""),
                action_result.info.get("desired_gamma", ""),
                action_result.info.get("gamma_error", ""),
                action_result.info.get("pitch_soft_limited", ""),
                action_result.info.get("physical_pull_limited", ""),
                action_result.info.get("pull_energy_scale", ""),
                action_result.info.get("pull_geometry_scale", ""),
                action_result.info.get("opening_pull_boost_active", ""),
                action_result.info.get("threat_ata", ""),
                action_result.info.get("aim_az", ""),
                action_result.info.get("aim_el", ""),
                action_result.info.get("advantage_score", ""),
                action_result.info.get("advantage_ttc", ""),
                action_result.info.get("advantage_manager_reason", ""),
                action_result.info.get("los_rate", ""),
                action_result.info.get("ata_rate", ""),
                action_result.info.get("own_yaw_rate", ""),
                action_result.info.get("target_yaw_rate", ""),
                action_result.info.get("intercept_horizon", ""),
                action_result.info.get("desired_turn_rate", ""),
                action_result.info.get("target_throttle", ""),
                action_result.info.get("speed_error", ""),
                action_result.info.get("target_yaw_accel", ""),
                action_result.info.get("target_turn_stable_s", ""),
                action_result.info.get("prediction_stable", ""),
                action_result.info.get("lag_pursuit_active", ""),
                action_result.info.get("lag_offset_m", ""),
                action_result.info.get("lag_pursuit_scale", ""),
                action_result.info.get("lag_energy_preserve_active", ""),
                action_result.info.get("world_elevation", ""),
                action_result.info.get("world_elevation_rate", ""),
                action_result.info.get("target_climb_rate", ""),
                action_result.info.get("target_vertical_accel", ""),
                action_result.info.get("target_vertical_stable_s", ""),
                action_result.info.get("vertical_prediction_horizon", ""),
                action_result.info.get("vertical_prediction_stable", ""),
                action_result.info.get("vertical_bank_relief_active", ""),
                action_result.info.get("defensive_escape_active", ""),
                action_result.info.get("defensive_escape_sign", ""),
                action_result.info.get("bank_reversal_active", ""),
                action_result.info.get("rear_rate_match_active", ""),
                action_result.info.get("engagement_mode", ""),
                action_result.info.get("overshoot_control_active", ""),
                action_result.info.get("vertical_follow_active", ""),
                action_result.info.get("energy_deficit", ""),
                action_result.info.get("integrated_rate_bank_active", ""),
                action_result.info.get("energy_gamma_guard_active", ""),
                action_result.info.get("horizontal_energy_hold_active", ""),
                action_result.info.get("mutual_commit_active", ""),
                action_result.info.get("predictive_guidance_active", ""),
                action_result.info.get("predictive_horizon", ""),
                action_result.info.get("predictive_score", ""),
                action_result.info.get("predictive_future_error", ""),
                action_result.info.get("predictive_future_threat_error", ""),
                action_result.info.get("predictive_worst_target_rate", ""),
                action_result.info.get("predictive_sign_guard_active", ""),
                action_result.info.get("rule_desired_turn_rate", ""),
                action_result.info.get("predictive_rule_score", ""),
                action_result.info.get("predictive_score_gain", ""),
                action_result.info.get("predictive_override_active", ""),
                action_result.info.get("spiral_recovery_active", ""),
                action_result.info.get("rate_deficit_overbank_active", ""),
                action_result.info.get("sequential_maneuver_phase", ""),
                action_result.info.get("high_bank_speed_control_active", ""),
                action_result.info.get("high_bank_desired_speed", ""),
                action_result.info.get("turn_match_active", ""),
                action_result.info.get("turn_match_candidate_s", ""),
                action_result.info.get("vertical_alignment_active", ""),
                action_result.info.get("energy_recovery_active", ""),
                action_result.info.get("post_defense_conversion_active", ""),
                action_result.info.get("post_defense_conversion_armed", ""),
                action_result.info.get("conversion_rear_offset_m", ""),
                action_result.info.get("conversion_lateral_offset_m", ""),
                action_result.info.get("terminal_track_active", ""),
                action_result.info.get("terminal_vertical_unload_active", ""),
                action_result.info.get("attack_conversion_active", ""),
                action_result.info.get("headon_deconflict_active", ""),
                action_result.info.get("lift_vector_active", ""),
                action_result.info.get("lift_vector_target_bank", ""),
                action_result.info.get("lift_vector_target_gamma", ""),
                action_result.info.get("lift_vector_los_rate", ""),
                action_result.info.get("lift_vector_accel", ""),
                action_result.info.get("lift_vector_authority", ""),
                action_result.info.get("lift_vector_conflict", ""),
                action_result.info.get("first_merge_passed", ""),
                action_result.info.get("formula_vpp_active", ""),
                action_result.info.get("formula_vpp_blend", ""),
                action_result.info.get("formula_vpp_mode", ""),
                action_result.info.get("formula_vpp_turn_rate_degps", ""),
                action_result.info.get("formula_vpp_t_cpa_s", ""),
                action_result.info.get("formula_vpp_d_cpa_m", ""),
                action_result.info.get("formula_vpp_turn_circle_active", ""),
                action_result.info.get("formula_vpp_pass_detected", ""),
                action_result.info.get("formula_vpp_recommit_active", ""),
                action_result.info.get("formula_vpp_recommit_remaining_s", ""),
                action_result.info.get("formula_projectile_tof_s", ""),
                action_result.info.get("formula_apg_weight", ""),
                action_result.info.get("formula_energy_gamma_correction", ""),
                action_result.info.get("formula_zem_defense_active", ""),
                action_result.info.get("formula_zem_escape_sign", ""),
                action_result.info.get("formula_zem_burst_remaining_s", ""),
                action_result.info.get("formula_zem_cooldown_remaining_s", ""),
                action_result.info.get("formula_rate_bank_active", ""),
                action_result.info.get("planner3d_active", ""),
                action_result.info.get("planner3d_replanned", ""),
                action_result.info.get("planner3d_manoeuvre", ""),
                action_result.info.get("planner3d_score", ""),
                action_result.info.get("planner3d_horizon", ""),
                action_result.info.get("planner3d_threat_pressure", ""),
                action_result.info.get("planner3d_turn_rate", ""),
                action_result.info.get("planner3d_gamma", ""),
                action_result.info.get("planner3d_target_speed", ""),
                action_result.info.get("planner3d_future_ata", ""),
                action_result.info.get("planner3d_future_threat_ata", ""),
                action_result.info.get("planner3d_future_range", ""),
                action_result.info.get("planner3d_worst_target_turn", ""),
                action_result.info.get("planner3d_worst_target_gamma", ""),
                action_result.info.get("residual_gate_active", ""),
                action_result.info.get("residual_gate_block_reason", ""),
                *raw_residual,
                *scaled_residual,
                *rule_action,
                *provider_action,
                action_result.info.get("bundle_id", ""),
                action_result.info.get("safety_override_active", ""),
                action_result.info.get("engagement_phase", ""),
                action_result.info.get("bank_owner", ""),
                action_result.info.get("pitch_owner", ""),
                action_result.info.get("throttle_owner", ""),
                action_result.info.get("bank_authority", ""),
                action_result.info.get("pitch_authority", ""),
                action_result.info.get("throttle_authority", ""),
                action_result.info.get("bank_proposal_count", ""),
                action_result.info.get("pitch_proposal_count", ""),
                action_result.info.get("throttle_proposal_count", ""),
                action_result.info.get("proposal_conflict", ""),
                action_result.info.get("proposal_saturated", ""),
                action_result.info.get("active_bank_proposals", ""),
                action_result.info.get("active_pitch_proposals", ""),
                action_result.info.get("active_throttle_proposals", ""),
                self._spawn_yaw_class,
                action_result.info.get("explicit_bank_override_active", ""),
                action_result.info.get("explicit_commit_sign", ""),
                action_result.info.get("explicit_commit_age_s", ""),
                action_result.info.get("explicit_reversal_candidate", ""),
                action_result.info.get("explicit_reversal_candidate_age_s", ""),
                action_result.info.get("explicit_initial_candidate", ""),
                action_result.info.get("explicit_initial_candidate_age_s", ""),
                action_result.info.get("explicit_opening_pull_active", ""),
                action_result.info.get("explicit_opening_pull_cmd", ""),
                action_result.info.get("roll_commit_active", ""),
                action_result.info.get("roll_commit_sign", ""),
                action_result.info.get("energy_floor_active", ""),
                action_result.info.get("energy_unload_active", ""),
                action_result.info.get("energy_pitch_scale", ""),
                action_result.info.get("measured_envelope_active", ""),
                action_result.info.get("measured_target_speed_mps", ""),
                action_result.info.get("measured_pursuit_speed_margin_mps", ""),
                action_result.info.get("measured_speed_error_mps", ""),
                action_result.info.get("measured_opening_fill_active", ""),
                action_result.info.get("measured_opening_fill_cmd", ""),
                action_result.info.get("measured_energy_unload_active", ""),
                action_result.info.get("measured_pitch_floor", ""),
                action_result.info.get("measured_safe_terminal", ""),
                action_result.info.get("measured_raw_throttle", ""),
                action_result.info.get("measured_opening_roll_active", ""),
                action_result.info.get("measured_bank_hold_active", ""),
                action_result.info.get("measured_roll_schedule_cmd", ""),
                action_result.info.get("measured_enemy_speed_mps", ""),
                action_result.info.get("measured_speed_recovery_active", ""),
                action_result.info.get("measured_immediate_threat", ""),
                action_result.info.get("measured_energy_bank_active", ""),
                action_result.info.get("measured_energy_bank_target_deg", ""),
                action_result.info.get("measured_rear_trap", ""),
                action_result.info.get("measured_rear_trap_age_s", ""),
                action_result.info.get("measured_vertical_phase", ""),
                action_result.info.get("measured_vertical_phase_age_s", ""),
                action_result.info.get("measured_vertical_pitch_cmd", ""),
                action_result.info.get("measured_vertical_escape_active", ""),
                action_result.info.get("measured_counter_turn_sign", ""),
                action_result.info.get("measured_counter_min_distance_m", ""),
                action_result.info.get("measured_counter_overshoot_detected", ""),
                action_result.info.get("measured_counter_used", ""),
                action_result.info.get("measured_sustained_turn_cut", ""),
                action_result.info.get("measured_vertical_speed_ema_mps", ""),
                action_result.info.get("measured_altitude_loss_m", ""),
                action_result.info.get("measured_vertical_recovery_active", ""),
                action_result.info.get("measured_vertical_recovery_bank_deg", ""),
                action_result.info.get("measured_los_aim_delta_deg", ""),
                action_result.info.get("measured_los_disagree_age_s", ""),
                action_result.info.get("measured_los_reacquire_active", ""),
                action_result.info.get("measured_los_reacquire_sign", ""),
                action_result.info.get("measured_los_reacquire_used", ""),
                action_result.info.get("measured_terminal_lift_active", ""),
                action_result.info.get("measured_terminal_lift_age_s", ""),
                action_result.info.get("measured_terminal_lift_bank_deg", ""),
                action_result.info.get("measured_rear_capture_active", ""),
                action_result.info.get("measured_rear_capture_az_deg", ""),
                action_result.info.get("measured_rear_capture_offset_m", ""),
                action_result.info.get("measured_speed_advantage_mps", ""),
                action_result.info.get("distance_handoff_active", ""),
                action_result.info.get("distance_handoff_near_latched", ""),
                action_result.info.get("distance_handoff_rl_weight", ""),
                action_result.info.get("distance_handoff_distance_m", ""),
                action_result.info.get("distance_handoff_rule_roll", ""),
                action_result.info.get("distance_handoff_rule_pitch", ""),
                action_result.info.get("distance_handoff_rule_yaw", ""),
                action_result.info.get("distance_handoff_rule_throttle", ""),
                action_result.info.get("distance_handoff_rl_roll", ""),
                action_result.info.get("distance_handoff_rl_pitch", ""),
                action_result.info.get("distance_handoff_rl_yaw", ""),
                action_result.info.get("distance_handoff_rl_throttle", ""),
                action_result.info.get("max_rate_turn_active", ""),
                action_result.info.get("max_rate_turn_commit_sign", ""),
                action_result.info.get("max_rate_turn_matched_target", ""),
                action_result.info.get("max_rate_turn_target_yaw_rate_degps", ""),
                action_result.info.get("max_rate_turn_target_bank_deg", ""),
                action_result.info.get("max_rate_turn_target_speed_mps", ""),
                action_result.info.get("postmerge_pull_boost_active", ""),
                action_result.info.get("postmerge_pull_boost_age_s", ""),
                action_result.info.get("postmerge_pull_boost_cmd", ""),
                action_result.info.get("postmerge_pull_boost_duration_s", ""),
            ])
            self._log_csv_file.flush()

        return action

    def _print_action_repeat_debug(
        self,
        context: RemoteClientContext,
        pair_count: int,
        policy_updated: bool,
        action: np.ndarray,
    ) -> None:
        repeat_offset = pair_count % self.action_repeat
        print(
            "[DogFightEnv][Unreal][ACTION_REPEAT] "
            f"pair_count={pair_count} repeat={self.action_repeat} "
            f"repeat_offset={repeat_offset} policy_updated={policy_updated} "
            f"cmd_frame={context.frame_index} "
            f"own_frame={context.own_plane.frame_index} "
            f"enemy_frame={context.enemy_plane.frame_index} "
            f"policy_frame={self._last_policy_frame_index} "
            f"policy_count={self._last_policy_count} "
            f"action={np.asarray(action, dtype=np.float32).tolist()}"
        )

    def _build_observation(self, ownship_state, target_state) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
        )


def plane_info_to_state(plane_info) -> np.ndarray:
    state = np.zeros(51, dtype=np.float32)
    state[0] = plane_info.position.x
    state[1] = plane_info.position.y
    # D is NED down-positive; empirically confirmed 2026-08-19 via
    # --debug-raw-state-frames that Unreal's PlaneInfo.position.z is
    # altitude in meters, UP-positive (observed 4572.0 == exactly 15000ft
    # at a level cruise start -- a raw D value would have to be negative
    # there, not +4572). Previously this was assigned unnegated (state[2]
    # = position.z directly), which fed every relative-geometry calc in
    # GeoMathUtil (_get_distance/_get_aspect_angle/_get_antenna_train_angle/
    # _get_los_angle, all consuming state[0:3] raw) a sign-inverted vertical
    # axis, and separately fed StateIndex.ALT the same value re-negated
    # (see below) -- i.e. own altitude observed as around -4572m instead of
    # +4572m. A policy trained to treat negative/low altitude as imminent
    # ground impact would read that as "already crashing" on every live
    # connection and pull full climb immediately, matching the "climbs
    # immediately, never turns toward target" failure seen across every
    # model tested against the real Unreal server this session.
    state[2] = -plane_info.position.z
    state[3] = plane_info.rotation.roll
    # PITCH must NOT be wrapped like YAW below: observation.py normalizes
    # pitch with normalize(pitch, -90.0, 90.0) (symmetric range, matching
    # Unreal's native Rotator.pitch convention already), not a 0-360 compass
    # range. `% 360.0` here would send any negative pitch (nose-down --
    # true for almost this entire session's live flights, e.g. -38.56deg in
    # live_run1.csv) to a large positive value (-38.56 % 360 == 321.44),
    # which normalize()'s clip then floors to exactly +1.0 -- i.e. the
    # policy would read a hard dive as "pointed straight up", the opposite
    # of reality. Found 2026-08-20 as a regression introduced alongside the
    # (correct) YAW wrap fix just below.
    state[4] = plane_info.rotation.pitch
    # YAW: observation.py's normalize(yaw, 0.0, 360.0) (tactical16 obs[2])
    # uses np.clip, not wraparound -- it assumes JSBSim's own [0,360) compass
    # heading convention (empirically confirmed 2026-08-20 via a local JSBSim
    # rollout: StateIndex.YAW never went negative there). Unreal's
    # PlaneInfo.rotation.yaw is a native Rotator component in [-180,180], so
    # every negative live yaw (16.8% of frames in artifacts/logs/live_run1.csv,
    # including frame 1 at -90.0) was clipping to exactly -1.0 regardless of
    # magnitude -- i.e. the policy's own-heading feature collapsed to a
    # single constant, wrong value (equivalent to reading heading=0) across
    # an entire 180deg span of real headings, an observation shape training
    # never produced. Wrap into [0,360) to match training's convention.
    state[5] = plane_info.rotation.yaw % 360.0
    state[6] = plane_info.velocity.x
    state[7] = plane_info.velocity.y
    state[8] = plane_info.velocity.z
    # PlaneInfo (MT_PlaneInfo) carries position/rotation/velocity only -- no
    # airspeed or health field. StateIndex.KCAS/ALT/HEALTH are read directly
    # by observation.py's tactical16 builder (obs[3]/obs[4]/obs[5]/obs[13]),
    # so leaving them at the np.zeros() default silently fed the policy a
    # constant "speed=0, altitude=0(ground), health=0(dead)" observation over
    # live Unreal inference, diverging from training's populated state.
    # ALT: position.z already IS altitude (up-positive, meters) per the
    # empirical check above -- no negation.
    state[StateIndex.ALT] = plane_info.position.z
    # KCAS: PlaneInfo has no calibrated-airspeed field; ground-speed
    # magnitude from velocity is an approximation (ignores air density/wind)
    # but is far closer to training-time KCAS than a hardcoded 0.
    state[StateIndex.KCAS] = float(
        np.linalg.norm(
            [plane_info.velocity.x, plane_info.velocity.y, plane_info.velocity.z]
        )
    )
    # HEALTH: not present anywhere in the PlaneInfo struct. The protocol
    # defines MessageType.MT_Damage separately, but no struct/unpack exists
    # for it anywhere in this client yet (todo: implement once the wire
    # format is known). Default to 1.0 (full health) rather than 0.0 (dead)
    # so the policy at least doesn't see a constant "already destroyed"
    # signal; this is a placeholder, not a real fix.
    state[StateIndex.HEALTH] = 1.0
    # SIM_TIME: never populated before (2026-08-20 fix). observation.py's
    # _build_tactical19 finite-differences relative position over
    # (sim_time - prev_sim_time) to estimate the target's relative velocity
    # (this env has no live velocity readout wired into JSBSim training, so
    # tactical19 uses the same finite-difference method on both paths for
    # train/inference consistency -- see _build_tactical19's docstring).
    # Leaving SIM_TIME at the np.zeros() default meant it was *always
    # exactly 0.0* on every call, so `sim_time > _prev_sim_time` was never
    # true and the relative-velocity features were silently pinned to zero
    # on every live connection -- i.e. tactical19 gave the policy no more
    # lead-pursuit information than tactical16 ever did over live inference,
    # reproducing this session's core "can't aim, flees, oscillates/inverts"
    # failure mode even on a checkpoint that tested well in JSBSim eval.
    # plane_info.index is a monotonically increasing frame counter at the
    # rules deck's documented 60Hz telemetry rate -- divide by 60 to get a
    # seconds-scale, monotonically increasing proxy for elapsed sim time.
    state[StateIndex.SIM_TIME] = float(plane_info.index) / 60.0
    return state
