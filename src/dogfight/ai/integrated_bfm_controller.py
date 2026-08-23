"""Integrated 1v1 BFM acquisition controller for live Unreal telemetry.

Unlike the W1 family, this controller does not collapse every large LOS angle
to a fixed bank.  It estimates engagement rates, classifies tactical state,
computes a bounded horizontal intercept point, commands a desired course rate,
and converts that rate into speed-aware bank/pull/throttle commands.

This is an acquisition/track controller, not a learned weapons policy.  A
learned fine-track provider can be handed control inside weapons_track later.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex


G = 9.80665


def wrap180(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def signed_unit(value: float, fallback: int = 1) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return fallback


@dataclass
class IntegratedBFMConfig:
    controller_name: str = "ibfm1"

    # Tactical gates.
    merge_initial_s: float = 2.0
    break_hold_s: float = 12.0
    defensive_range_m: float = 2200.0
    defensive_threat_ata_deg: float = 35.0
    defensive_closure_mps: float = 20.0
    track_ata_deg: float = 35.0
    weapons_ata_deg: float = 10.0
    weapons_range_m: float = 1400.0

    # Estimator smoothing.
    rate_alpha: float = 0.25
    target_velocity_alpha: float = 0.20
    target_velocity_clip_mps: float = 450.0

    # Horizontal intercept horizons by manager state.
    horizon_merge_s: float = 0.3
    horizon_break_s: float = 0.8
    horizon_defensive_s: float = 0.0
    horizon_reacquire_max_s: float = 6.0
    horizon_track_max_s: float = 1.0
    horizon_weapons_s: float = 0.15

    # Guidance: desired course rate = az gain + LOS-rate feed-forward.
    az_to_turn_rate_gain: float = 0.060
    los_rate_gain: float = 0.35
    turn_rate_merge_degps: float = 4.0
    turn_rate_break_degps: float = 6.0
    turn_rate_reacquire_degps: float = 5.5
    turn_rate_track_degps: float = 3.5
    turn_rate_weapons_degps: float = 2.0
    turn_rate_defensive_degps: float = 6.0
    acquisition_min_turn_rate_degps: float = 0.0
    acquisition_min_turn_rate_ata_deg: float = 0.0
    max_bank_deg: float = 70.0
    break_bank_deg: float = 67.0
    sign_min_hold_s: float = 3.0
    # Values below 180 enable a rear-hemisphere commitment zone.  W15 uses
    # 120deg so +180/-180 representation wraps cannot reverse guidance; the
    # predicted aimpoint must first return to a physically unambiguous side.
    guidance_rear_commit_deg: float = 180.0
    use_horizontal_course_guidance: bool = False
    use_constant_turn_prediction: bool = False
    adaptive_turn_prediction: bool = False
    turn_prediction_max_arc_deg: float = 180.0
    turn_prediction_unstable_horizon_s: float = 6.0
    turn_prediction_min_stable_s: float = 0.0
    turn_prediction_yaw_accel_limit_degps2: float = 1.0e9
    lag_pursuit_ata_deg: float = 0.0
    lag_pursuit_range_m: float = 0.0
    lag_pursuit_closure_mps: float = 1.0e9
    lag_pursuit_offset_min_m: float = 0.0
    lag_pursuit_offset_max_m: float = 0.0
    lag_pursuit_offset_gain_s: float = 0.0
    vertical_alignment_elevation_deg: float = 0.0
    vertical_alignment_range_m: float = 0.0
    vertical_alignment_closure_mps: float = 1.0e9
    vertical_alignment_lag_offset_m: float = 0.0
    vertical_alignment_lag_gain_m_per_deg: float = 0.0
    vertical_alignment_lag_max_m: float = 0.0
    vertical_alignment_target_closure_mps: float = 0.0

    # Stable gun-track mode: after acquisition, match the bandit's measured
    # turn rate instead of repeatedly cutting across its circle.  Hysteresis
    # prevents rapid switching around the entry ATA.
    turn_match_enter_ata_deg: float = 0.0
    turn_match_exit_ata_deg: float = 0.0
    turn_match_enter_range_m: float = 0.0
    turn_match_exit_range_m: float = 0.0
    turn_match_min_threat_ata_deg: float = 0.0
    turn_match_enter_course_error_deg: float = 180.0
    turn_match_exit_course_error_deg: float = 180.0
    turn_match_enter_max_closure_mps: float = 1.0e9
    turn_match_entry_hold_s: float = 0.0
    turn_match_require_stable_prediction: bool = False
    turn_match_yaw_rate_gain: float = 1.0
    turn_match_course_gain: float = 0.0
    turn_match_los_rate_gain: float = 0.0
    turn_match_rate_limit_degps: float = 14.0
    turn_match_sign_guard_error_deg: float = 0.0
    turn_match_sign_guard_rate_degps: float = 3.0
    terminal_track_enter_ata_deg: float = 0.0
    terminal_track_exit_ata_deg: float = 0.0
    terminal_track_enter_range_m: float = 0.0
    terminal_track_exit_range_m: float = 0.0
    terminal_track_prelock_ata_deg: float = 0.0
    terminal_track_prelock_range_m: float = 0.0
    terminal_track_min_threat_ata_deg: float = 0.0
    terminal_track_yaw_rate_gain: float = 0.7
    terminal_track_course_gain: float = 1.2
    terminal_track_los_rate_gain: float = 0.4
    terminal_track_rate_limit_degps: float = 14.0
    # In the gun cone, point the nose at the target's current world-frame
    # elevation.  Long-horizon vertical extrapolation is useful for pursuit,
    # but a diving target can otherwise command nose-down while it is still
    # visibly above the nose and spoil the final few degrees of ATA.
    terminal_pitch_attitude_kp: float = 0.0
    terminal_pitch_los_rate_gain: float = 0.0
    terminal_vertical_unload_el_deg: float = 0.0
    terminal_vertical_unload_ratio: float = 0.7
    terminal_vertical_unload_bank_scale: float = 0.15
    direct_course_bank_full_error_deg: float = 0.0
    direct_course_bank_exponent: float = 1.0
    direct_course_rear_ambiguity_deg: float = 180.0
    direct_course_roll_bias: float = 0.0
    direct_course_min_bank_deg: float = 0.0
    direct_course_min_bank_ata_deg: float = 0.0
    bank_reversal_error_deg: float = 0.0
    bank_reversal_full_cmd: float = 0.0
    bank_reversal_release_bank_deg: float = 0.0

    # Roll rate cascade, live pulse-derived.
    bank_kp: float = 1.0
    roll_rate_limit_degps: float = 35.0
    roll_rate_gain_degps_per_unit: float = 140.0
    roll_cmd_limit: float = 0.30

    # Optional initial acquisition commit.  Disabled by default so W14-W16
    # remain bit-for-bit on their original control path.  Live W16 data
    # showed that the normal bank-error cascade reduced roll_cmd too early:
    # it needed about 7s to reach 65deg bank while ATA escaped behind us.
    initial_commit_duration_s: float = 0.0
    initial_commit_ata_deg: float = 60.0
    initial_commit_bank_deg: float = 75.0
    initial_commit_release_bank_deg: float = 65.0
    initial_commit_min_roll_cmd: float = 0.0

    # Optional coarse max-rate turn.  Unlike the short initial commit this
    # holds a measured high-bank state until the target reaches the forward
    # tracking cone.  Disabled by default for W14-W18 compatibility.
    coarse_turn_ata_deg: float = 0.0
    coarse_turn_reentry_ata_deg: float = 0.0
    coarse_turn_bank_deg: float = 0.0
    coarse_turn_roll_bias: float = 0.0

    # Optional short max-authority identification schedule.  These values are
    # zero by default; W20 enables them to find the live plant's upper bound
    # before tuning downward.
    max_test_duration_s: float = 0.0
    max_test_requires_coarse_turn: bool = True
    max_test_roll_full_until_deg: float = 0.0
    max_test_roll_full_cmd: float = 0.0
    max_test_roll_taper_until_deg: float = 0.0
    max_test_roll_taper_cmd: float = 0.0
    max_test_roll_brake_above_deg: float = 0.0
    max_test_roll_brake_cmd: float = 0.0
    max_test_pitch_bank_gate_deg: float = 0.0
    max_test_pitch_ata_gate_deg: float = 0.0
    max_test_pitch_cmd: float = 0.0
    max_test_pitch_relaxed_cmd: float = 0.0
    max_test_pitch_vertical_gain: float = 0.0
    max_test_target_vertical_speed_mps: float = 0.0
    max_test_pitch_bank_relax_gain: float = 0.0
    max_test_min_speed_mps: float = 0.0
    max_test_min_altitude_m: float = 0.0

    # Vertical flight-path and banked-turn pull.
    gamma_limit_deg: float = 15.0
    gamma_kp: float = 1.0
    pitch_rate_limit_degps: float = 10.0
    pitch_rate_gain_degps_per_unit: float = 25.0
    pitch_trim: float = -0.05
    turn_pull_at_max_bank: float = -0.22
    pitch_cmd_limit: float = 0.40
    fine_vertical_los_blend: float = 0.25
    vertical_prediction_horizon_s: float = 0.0
    use_altitude_rate_vertical_prediction: bool = False
    vertical_prediction_stable_horizon_s: float = 1.0
    vertical_prediction_unstable_horizon_s: float = 0.4
    vertical_prediction_min_stable_s: float = 0.7
    vertical_prediction_accel_limit_mps2: float = 35.0
    vertical_maneuver_gamma_limit_deg: float = 0.0
    vertical_maneuver_ata_gate_deg: float = 0.0
    vertical_bank_relief_elevation_deg: float = 0.0
    vertical_bank_relief_min_scale: float = 1.0
    vertical_speed_damping_deadband_mps: float = 0.0
    vertical_speed_damping_gain: float = 0.0
    vertical_speed_damping_track_desired_gamma: bool = False

    # Reactive defensive vertical separation.  The original defensive state
    # only changed gains/throttle and therefore flew an easy, persistent arc.
    defensive_vertical_escape: bool = False
    defensive_escape_threat_ata_deg: float = 20.0
    defensive_escape_own_ata_deg: float = 60.0
    defensive_escape_range_m: float = 1800.0
    defensive_escape_gamma_deg: float = 30.0
    defensive_escape_deadband_m: float = 120.0
    defensive_escape_switch_s: float = 2.3

    # After a nose-on threat has been defeated, avoid falling straight back
    # into pure pursuit.  Briefly target a point behind the bandit's velocity
    # vector and outside its measured turn circle, creating a side/rear entry
    # opportunity before terminal tracking takes over.
    post_defense_conversion_duration_s: float = 0.0
    post_defense_conversion_arm_window_s: float = 3.0
    post_defense_conversion_min_threat_ata_deg: float = 40.0
    post_defense_conversion_max_range_m: float = 3000.0
    post_defense_conversion_rear_offset_m: float = 1500.0
    post_defense_conversion_lateral_offset_m: float = 800.0
    post_defense_conversion_min_target_speed_mps: float = 40.0

    # Optional bang-bang sustained pull for coarse acquisition.  Negative
    # pitch command is nose-up on the live plant.  Defaults disable it so
    # earlier controller modes are unchanged.
    sustained_pull_rear_cmd: float = 0.0
    sustained_pull_rear_ata_deg: float = 90.0
    sustained_pull_mid_cmd: float = 0.0
    sustained_pull_mid_ata_deg: float = 60.0
    sustained_pull_min_bank_deg: float = 55.0
    sustained_pull_min_speed_mps: float = 0.0
    sustained_pull_min_altitude_m: float = 0.0

    # State-specific energy management.
    corner_speed_mps: float = 285.0
    corner_throttle_base: float = 0.48
    corner_speed_gain: float = 0.004
    throttle_min: float = 0.20
    throttle_max: float = 0.95
    throttle_slew_per_s: float = 0.45
    throttle_merge: float = 0.90
    throttle_defensive: float = 0.95
    throttle_weapons: float = 0.48

    # Optional closure-aware approach energy control.  Scalar speed matching
    # cannot detect a 300m/s head-on closure when both aircraft have the same
    # airspeed, so use radial closure directly before entering the WEZ.
    closure_throttle_ata_deg: float = 0.0
    closure_throttle_range_m: float = 0.0
    closure_throttle_far_range_m: float = 3000.0
    closure_throttle_near_range_m: float = 1500.0
    closure_target_far_mps: float = 120.0
    closure_target_mid_mps: float = 70.0
    closure_target_near_mps: float = 20.0
    closure_throttle_base: float = 0.50
    closure_throttle_gain: float = 0.0
    minimum_energy_speed_mps: float = 0.0
    minimum_energy_throttle: float = 0.0


class IntegratedBFMController(ActionProvider):
    def __init__(self, config: IntegratedBFMConfig | None = None):
        self.cfg = config or IntegratedBFMConfig()
        self.geometry = GeometryInfo()
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._start_time: float | None = None
        self._prev_time: float | None = None
        self._prev_distance: float | None = None
        self._prev_ata: float | None = None
        self._prev_los_az: float | None = None
        self._prev_horizontal_los_error: float | None = None
        self._prev_world_elevation: float | None = None
        self._prev_own_yaw: float | None = None
        self._prev_target_yaw: float | None = None
        self._prev_bank: float | None = None
        self._prev_pitch: float | None = None
        self._prev_alt: float | None = None
        self._prev_target_alt: float | None = None
        self._prev_target_position: np.ndarray | None = None
        self._target_velocity = np.zeros(3, dtype=np.float64)
        self._los_rate = 0.0
        self._horizontal_los_rate = 0.0
        self._world_elevation_rate = 0.0
        self._ata_rate = 0.0
        self._own_yaw_rate = 0.0
        self._target_yaw_rate = 0.0
        self._target_yaw_accel = 0.0
        self._target_turn_stable_s = 0.0
        self._roll_rate = 0.0
        self._pitch_rate = 0.0
        self._target_climb_rate = 0.0
        self._target_vertical_accel = 0.0
        self._target_vertical_stable_s = 0.0
        self._state = "merge"
        self._state_since = 0.0
        self._break_until = 0.0
        self._turn_sign = 1
        self._last_sign_change = -1e9
        self._coarse_turn_initialized = False
        self._coarse_turn_latched = False
        self._turn_match_latched = False
        self._turn_match_candidate_s = 0.0
        self._terminal_track_latched = False
        self._post_defense_conversion_until = 0.0
        self._post_defense_conversion_armed_until = 0.0
        self._throttle = self.cfg.throttle_merge
        self.state_log: list[str] = []
        self.info_log: list[dict] = []

    def _ema(self, previous: float, current: float) -> float:
        alpha = self.cfg.rate_alpha
        return alpha * current + (1.0 - alpha) * previous

    def _set_state(self, state: str, now: float) -> None:
        if state != self._state:
            self._state = state
            self._state_since = now

    def _intercept_horizon(
        self,
        state: str,
        relative_position: np.ndarray,
        target_velocity: np.ndarray,
        own_speed: float,
    ) -> float:
        cfg = self.cfg
        if state == "merge":
            return cfg.horizon_merge_s
        if state == "break_turn":
            return cfg.horizon_break_s
        if state == "defensive":
            return cfg.horizon_defensive_s
        if state == "weapons_track":
            return cfg.horizon_weapons_s

        max_horizon = (
            cfg.horizon_track_max_s if state == "offensive_track"
            else cfg.horizon_reacquire_max_s
        )

        # Constant-velocity intercept: |r + vt| = own_speed * t.
        r2 = float(np.dot(relative_position, relative_position))
        a = float(np.dot(target_velocity, target_velocity) - own_speed * own_speed)
        b = 2.0 * float(np.dot(relative_position, target_velocity))
        roots: list[float] = []
        if abs(a) < 1e-6:
            if abs(b) > 1e-6:
                roots.append(-r2 / b)
        else:
            discriminant = b * b - 4.0 * a * r2
            if discriminant >= 0.0:
                root = math.sqrt(discriminant)
                roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        positive = [value for value in roots if value > 0.0 and math.isfinite(value)]
        if positive:
            return float(np.clip(min(positive), 0.1, max_horizon))
        time_to_range = math.sqrt(r2) / max(own_speed, 1.0)
        return float(np.clip(time_to_range, 0.1, max_horizon))

    def _manager(
        self,
        now: float,
        elapsed: float,
        ata: float,
        threat_ata: float,
        distance: float,
        closure: float,
    ) -> str:
        cfg = self.cfg
        immediate_threat = (
            threat_ata <= cfg.defensive_threat_ata_deg
            and distance <= cfg.defensive_range_m
            and closure >= cfg.defensive_closure_mps
        )
        if immediate_threat:
            return "defensive"
        if ata <= cfg.weapons_ata_deg and distance <= cfg.weapons_range_m:
            return "weapons_track"
        if ata <= cfg.track_ata_deg:
            return "offensive_track"
        if elapsed <= cfg.merge_initial_s:
            self._break_until = max(self._break_until, now + cfg.break_hold_s)
            return "merge"
        if now <= self._break_until:
            return "break_turn"
        return "reacquire"

    def _turn_rate_limit(self, state: str) -> float:
        cfg = self.cfg
        return {
            "merge": cfg.turn_rate_merge_degps,
            "break_turn": cfg.turn_rate_break_degps,
            "reacquire": cfg.turn_rate_reacquire_degps,
            "offensive_track": cfg.turn_rate_track_degps,
            "weapons_track": cfg.turn_rate_weapons_degps,
            "defensive": cfg.turn_rate_defensive_degps,
        }[state]

    def compute_action(self, context: ActionContext) -> ActionResult:
        cfg = self.cfg
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            action = np.array([0.0, cfg.pitch_trim, 0.0, cfg.throttle_merge], dtype=np.float32)
            return ActionResult(action, f"{cfg.controller_name}[no_target]", info={"state": "no_target"})

        now = float(own[StateIndex.SIM_TIME])
        if self._start_time is None:
            self._start_time = now
            self._state_since = now
        elapsed = now - self._start_time
        dt = None
        if self._prev_time is not None and now > self._prev_time:
            dt = now - self._prev_time

        own_position = np.asarray(own[StateIndex.N : StateIndex.D + 1], dtype=np.float64)
        target_position = np.asarray(target[StateIndex.N : StateIndex.D + 1], dtype=np.float64)
        relative_position = target_position - own_position
        own_speed = max(1.0, float(own[StateIndex.KCAS]))
        target_speed = max(0.0, float(target[StateIndex.KCAS]))
        altitude = float(own[StateIndex.ALT])
        target_altitude = float(target[StateIndex.ALT])
        bank = wrap180(float(own[StateIndex.ROLL]))
        pitch = float(own[StateIndex.PITCH])
        own_yaw = float(own[StateIndex.YAW])
        target_yaw = float(target[StateIndex.YAW])

        distance = float(self.geometry._get_distance(own, target))
        los_az, los_el = self.geometry._get_los_angle(own, target)
        los_az, los_el = float(los_az), float(los_el)
        horizontal_bearing = math.degrees(math.atan2(
            relative_position[1], relative_position[0]
        ))
        horizontal_los_error = wrap180(horizontal_bearing - own_yaw)
        ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
        aa = abs(float(self.geometry._get_aspect_angle(own, target, False)))
        threat_ata = abs(float(self.geometry._get_antenna_train_angle(target, own, False)))

        closure = ata_rate = los_rate = 0.0
        horizontal_los_rate = 0.0
        raw_roll_rate = raw_pitch_rate = own_yaw_rate = target_yaw_rate = 0.0
        vertical_speed = 0.0
        target_climb_rate = 0.0
        if dt:
            if self._prev_distance is not None:
                closure = (self._prev_distance - distance) / dt
            if self._prev_ata is not None:
                ata_rate = (ata - self._prev_ata) / dt
            if self._prev_los_az is not None:
                los_rate = wrap180(los_az - self._prev_los_az) / dt
            if self._prev_horizontal_los_error is not None:
                horizontal_los_rate = wrap180(
                    horizontal_los_error - self._prev_horizontal_los_error
                ) / dt
            if self._prev_own_yaw is not None:
                own_yaw_rate = wrap180(own_yaw - self._prev_own_yaw) / dt
            if self._prev_target_yaw is not None:
                target_yaw_rate = wrap180(target_yaw - self._prev_target_yaw) / dt
            if self._prev_bank is not None:
                raw_roll_rate = wrap180(bank - self._prev_bank) / dt
            if self._prev_pitch is not None:
                raw_pitch_rate = wrap180(pitch - self._prev_pitch) / dt
            if self._prev_alt is not None:
                vertical_speed = (altitude - self._prev_alt) / dt
            if self._prev_target_alt is not None:
                target_climb_rate = (target_altitude - self._prev_target_alt) / dt
            if self._prev_target_position is not None:
                measured_velocity = (target_position - self._prev_target_position) / dt
                velocity_norm = float(np.linalg.norm(measured_velocity))
                if velocity_norm > cfg.target_velocity_clip_mps:
                    measured_velocity *= cfg.target_velocity_clip_mps / velocity_norm
                alpha = cfg.target_velocity_alpha
                self._target_velocity = alpha * measured_velocity + (1.0 - alpha) * self._target_velocity

        self._los_rate = self._ema(self._los_rate, los_rate)
        self._horizontal_los_rate = self._ema(
            self._horizontal_los_rate, horizontal_los_rate
        )
        self._ata_rate = self._ema(self._ata_rate, ata_rate)
        self._own_yaw_rate = self._ema(self._own_yaw_rate, own_yaw_rate)
        previous_filtered_target_yaw_rate = self._target_yaw_rate
        self._target_yaw_rate = self._ema(self._target_yaw_rate, target_yaw_rate)
        if dt:
            raw_target_yaw_accel = (
                self._target_yaw_rate - previous_filtered_target_yaw_rate
            ) / dt
            self._target_yaw_accel = self._ema(
                self._target_yaw_accel, raw_target_yaw_accel
            )
            same_turn = (
                abs(self._target_yaw_rate) >= 0.5
                and abs(previous_filtered_target_yaw_rate) >= 0.5
                and self._target_yaw_rate * previous_filtered_target_yaw_rate > 0.0
            )
            if same_turn:
                self._target_turn_stable_s += dt
            else:
                self._target_turn_stable_s = 0.0
        self._roll_rate = self._ema(self._roll_rate, raw_roll_rate)
        self._pitch_rate = self._ema(self._pitch_rate, raw_pitch_rate)
        previous_target_climb_rate = self._target_climb_rate
        self._target_climb_rate = self._ema(
            self._target_climb_rate, target_climb_rate
        )
        if dt:
            raw_vertical_accel = (
                self._target_climb_rate - previous_target_climb_rate
            ) / dt
            self._target_vertical_accel = self._ema(
                self._target_vertical_accel, raw_vertical_accel
            )
            same_vertical_direction = (
                abs(self._target_climb_rate) >= 3.0
                and abs(previous_target_climb_rate) >= 3.0
                and self._target_climb_rate * previous_target_climb_rate > 0.0
            )
            if same_vertical_direction:
                self._target_vertical_stable_s += dt
            else:
                self._target_vertical_stable_s = 0.0

        previous_state = self._state
        state = self._manager(now, elapsed, ata, threat_ata, distance, closure)
        if (
            cfg.post_defense_conversion_duration_s > 0.0
            and previous_state == "defensive"
            and state != "defensive"
        ):
            if cfg.post_defense_conversion_arm_window_s > 0.0:
                self._post_defense_conversion_armed_until = (
                    now + cfg.post_defense_conversion_arm_window_s
                )
            elif (
                threat_ata >= cfg.post_defense_conversion_min_threat_ata_deg
                and distance <= cfg.post_defense_conversion_max_range_m
            ):
                self._post_defense_conversion_until = (
                    now + cfg.post_defense_conversion_duration_s
                )
        if state == "defensive":
            self._post_defense_conversion_until = 0.0
            self._post_defense_conversion_armed_until = 0.0
        elif (
            now < self._post_defense_conversion_armed_until
            and threat_ata >= cfg.post_defense_conversion_min_threat_ata_deg
            and distance <= cfg.post_defense_conversion_max_range_m
        ):
            self._post_defense_conversion_until = (
                now + cfg.post_defense_conversion_duration_s
            )
            self._post_defense_conversion_armed_until = 0.0
        if (
            now < self._post_defense_conversion_until
            and threat_ata < cfg.post_defense_conversion_min_threat_ata_deg
        ):
            self._post_defense_conversion_until = 0.0
        if distance > cfg.post_defense_conversion_max_range_m:
            self._post_defense_conversion_until = 0.0
        self._set_state(state, now)

        # Predict horizontal target position only.  Vertical extrapolation was
        # explicitly rejected by W12 live data.
        horizontal_relative = relative_position.copy()
        horizontal_relative[2] = 0.0
        horizontal_range_now = max(
            1.0, float(np.hypot(relative_position[0], relative_position[1]))
        )
        world_elevation = math.degrees(math.atan2(
            target_altitude - altitude, horizontal_range_now
        ))
        if self._prev_world_elevation is not None and dt > 1e-6:
            raw_world_elevation_rate = (
                world_elevation - self._prev_world_elevation
            ) / dt
            self._world_elevation_rate = self._ema(
                self._world_elevation_rate, raw_world_elevation_rate
            )
        self._prev_world_elevation = world_elevation
        horizontal_velocity = self._target_velocity.copy()
        horizontal_velocity[2] = 0.0
        horizon = self._intercept_horizon(state, horizontal_relative, horizontal_velocity, own_speed)
        prediction_stable = True
        if cfg.adaptive_turn_prediction:
            prediction_stable = (
                self._target_turn_stable_s >= cfg.turn_prediction_min_stable_s
                and abs(self._target_yaw_accel)
                <= cfg.turn_prediction_yaw_accel_limit_degps2
            )
            if not prediction_stable:
                horizon = min(horizon, cfg.turn_prediction_unstable_horizon_s)
            if abs(self._target_yaw_rate) >= 0.5:
                arc_limited_horizon = (
                    cfg.turn_prediction_max_arc_deg / abs(self._target_yaw_rate)
                )
                horizon = min(horizon, arc_limited_horizon)
        aim_target = np.array(target, copy=True)
        if (
            cfg.use_constant_turn_prediction
            and abs(self._target_yaw_rate) >= 0.5
            and horizon > 0.0
        ):
            omega = math.radians(self._target_yaw_rate)
            angle = omega * horizon
            sin_a, cos_a = math.sin(angle), math.cos(angle)
            vn, ve = horizontal_velocity[0], horizontal_velocity[1]
            delta_n = (vn * sin_a + ve * (cos_a - 1.0)) / omega
            delta_e = (vn * (1.0 - cos_a) + ve * sin_a) / omega
            aim_target[StateIndex.N] = target_position[0] + delta_n
            aim_target[StateIndex.E] = target_position[1] + delta_e
        else:
            aim_target[StateIndex.N] = target_position[0] + horizontal_velocity[0] * horizon
            aim_target[StateIndex.E] = target_position[1] + horizontal_velocity[1] * horizon

        post_defense_conversion_active = (
            cfg.post_defense_conversion_duration_s > 0.0
            and now < self._post_defense_conversion_until
            and state != "defensive"
            and threat_ata >= cfg.post_defense_conversion_min_threat_ata_deg
            and distance <= cfg.post_defense_conversion_max_range_m
        )
        conversion_rear_offset_m = 0.0
        conversion_lateral_offset_m = 0.0
        horizontal_target_speed = float(np.linalg.norm(horizontal_velocity[:2]))
        if (
            post_defense_conversion_active
            and horizontal_target_speed
            >= cfg.post_defense_conversion_min_target_speed_mps
        ):
            track_unit = horizontal_velocity[:2] / horizontal_target_speed
            left_normal = np.array([-track_unit[1], track_unit[0]])
            turn_sign = signed_unit(self._target_yaw_rate, self._turn_sign)
            outside_normal = -turn_sign * left_normal
            conversion_rear_offset_m = cfg.post_defense_conversion_rear_offset_m
            conversion_lateral_offset_m = cfg.post_defense_conversion_lateral_offset_m
            aim_target[StateIndex.N] -= track_unit[0] * conversion_rear_offset_m
            aim_target[StateIndex.E] -= track_unit[1] * conversion_rear_offset_m
            aim_target[StateIndex.N] += outside_normal[0] * conversion_lateral_offset_m
            aim_target[StateIndex.E] += outside_normal[1] * conversion_lateral_offset_m

        terminal_track_enabled = cfg.terminal_track_enter_ata_deg > 0.0
        if terminal_track_enabled:
            if self._terminal_track_latched:
                if (
                    ata >= cfg.terminal_track_exit_ata_deg
                    or distance >= cfg.terminal_track_exit_range_m
                    or threat_ata < 0.5 * cfg.terminal_track_min_threat_ata_deg
                ):
                    self._terminal_track_latched = False
            elif (
                (
                    ata <= cfg.terminal_track_enter_ata_deg
                    and distance <= cfg.terminal_track_enter_range_m
                )
                or (
                    cfg.terminal_track_prelock_ata_deg > 0.0
                    and ata <= cfg.terminal_track_prelock_ata_deg
                    and distance <= cfg.terminal_track_prelock_range_m
                )
            ) and threat_ata >= cfg.terminal_track_min_threat_ata_deg:
                self._terminal_track_latched = True
        else:
            self._terminal_track_latched = False
        terminal_track_active = self._terminal_track_latched
        if terminal_track_active:
            # Inside the gun setup, stop aiming at a lag/intercept proxy.  The
            # live damage cone is only 1/2/3 degrees, so point at the actual
            # aircraft and use rate feed-forward below to keep it there.
            aim_target[StateIndex.N] = target_position[0]
            aim_target[StateIndex.E] = target_position[1]

        # Once nearly aligned, high radial closure calls for lag pursuit rather
        # than an even farther lead point.  Move the aim point behind the
        # target along its measured ground track; disengage immediately if ATA
        # opens so acquisition authority is never sacrificed.
        normal_lag_active = (
            cfg.lag_pursuit_offset_max_m > 0.0
            and ata <= cfg.lag_pursuit_ata_deg
            and distance <= cfg.lag_pursuit_range_m
            and closure >= cfg.lag_pursuit_closure_mps
        )
        vertical_alignment_active = (
            cfg.vertical_alignment_elevation_deg > 0.0
            and abs(world_elevation) >= cfg.vertical_alignment_elevation_deg
            and distance <= cfg.vertical_alignment_range_m
            and closure >= cfg.vertical_alignment_closure_mps
        )
        lag_pursuit_active = (
            (normal_lag_active or vertical_alignment_active)
            and not terminal_track_active
        )
        lag_offset_m = 0.0
        horizontal_target_speed = float(np.linalg.norm(horizontal_velocity[:2]))
        if lag_pursuit_active and horizontal_target_speed > 1.0:
            if normal_lag_active:
                lag_offset_m = float(np.clip(
                    cfg.lag_pursuit_offset_min_m
                    + cfg.lag_pursuit_offset_gain_s
                    * (closure - cfg.lag_pursuit_closure_mps),
                    cfg.lag_pursuit_offset_min_m,
                    cfg.lag_pursuit_offset_max_m,
                ))
            if vertical_alignment_active:
                vertical_lag_offset = cfg.vertical_alignment_lag_offset_m
                if cfg.vertical_alignment_lag_gain_m_per_deg > 0.0:
                    vertical_lag_offset += (
                        cfg.vertical_alignment_lag_gain_m_per_deg
                        * max(
                            0.0,
                            abs(world_elevation)
                            - cfg.vertical_alignment_elevation_deg,
                        )
                    )
                    if cfg.vertical_alignment_lag_max_m > 0.0:
                        vertical_lag_offset = min(
                            vertical_lag_offset,
                            cfg.vertical_alignment_lag_max_m,
                        )
                lag_offset_m = max(lag_offset_m, vertical_lag_offset)
            track_unit = horizontal_velocity[:2] / horizontal_target_speed
            aim_target[StateIndex.N] -= track_unit[0] * lag_offset_m
            aim_target[StateIndex.E] -= track_unit[1] * lag_offset_m
        aim_target[StateIndex.D] = target[StateIndex.D]
        aim_target[StateIndex.ALT] = target[StateIndex.ALT]
        aim_az, aim_el = self.geometry._get_los_angle(own, aim_target)
        aim_az, aim_el = float(aim_az), float(aim_el)

        aim_delta_n = float(aim_target[StateIndex.N] - own_position[0])
        aim_delta_e = float(aim_target[StateIndex.E] - own_position[1])
        aim_world_bearing = math.degrees(math.atan2(aim_delta_e, aim_delta_n))
        aim_course_error = wrap180(aim_world_bearing - own_yaw)
        guidance_az = aim_course_error if cfg.use_horizontal_course_guidance else aim_az
        guidance_los_rate = (
            self._horizontal_los_rate
            if cfg.use_horizontal_course_guidance
            else self._los_rate
        )

        turn_match_enabled = cfg.turn_match_enter_ata_deg > 0.0
        if turn_match_enabled:
            if self._turn_match_latched:
                if (
                    ata >= cfg.turn_match_exit_ata_deg
                    or distance >= cfg.turn_match_exit_range_m
                    or threat_ata < 0.5 * cfg.turn_match_min_threat_ata_deg
                    or abs(guidance_az) >= cfg.turn_match_exit_course_error_deg
                ):
                    self._turn_match_latched = False
                    self._turn_match_candidate_s = 0.0
            entry_candidate = (
                ata <= cfg.turn_match_enter_ata_deg
                and distance <= cfg.turn_match_enter_range_m
                and threat_ata >= cfg.turn_match_min_threat_ata_deg
                and abs(guidance_az) <= cfg.turn_match_enter_course_error_deg
                and closure <= cfg.turn_match_enter_max_closure_mps
                and (
                    not cfg.turn_match_require_stable_prediction
                    or prediction_stable
                )
            )
            if not self._turn_match_latched:
                if entry_candidate:
                    self._turn_match_candidate_s += dt or 0.0
                    if self._turn_match_candidate_s >= cfg.turn_match_entry_hold_s:
                        self._turn_match_latched = True
                else:
                    self._turn_match_candidate_s = 0.0
        else:
            self._turn_match_latched = False
            self._turn_match_candidate_s = 0.0
        turn_match_active = self._turn_match_latched

        desired_sign = signed_unit(guidance_az, self._turn_sign)
        if state == "break_turn":
            if elapsed <= cfg.merge_initial_s + 0.25:
                self._turn_sign = desired_sign
        elif (
            abs(guidance_az) < cfg.guidance_rear_commit_deg
            and desired_sign != self._turn_sign
            and now - self._last_sign_change >= cfg.sign_min_hold_s
        ):
            # Outside break-turn, predicted aimpoint is allowed to recommit.
            self._turn_sign = desired_sign
            self._last_sign_change = now

        rate_limit = self._turn_rate_limit(state)
        rear_committed = abs(guidance_az) >= cfg.guidance_rear_commit_deg
        if (state in ("break_turn", "defensive") and ata > 60.0) or rear_committed:
            desired_turn_rate = self._turn_sign * rate_limit
        else:
            desired_turn_rate = (
                cfg.az_to_turn_rate_gain * guidance_az
                + cfg.los_rate_gain * guidance_los_rate
            )
            desired_turn_rate = float(np.clip(desired_turn_rate, -rate_limit, rate_limit))
        if (
            cfg.acquisition_min_turn_rate_degps > 0.0
            and ata >= cfg.acquisition_min_turn_rate_ata_deg
            and abs(desired_turn_rate) < cfg.acquisition_min_turn_rate_degps
        ):
            desired_turn_rate = (
                signed_unit(desired_turn_rate, self._turn_sign)
                * cfg.acquisition_min_turn_rate_degps
            )
        if turn_match_active:
            desired_turn_rate = float(np.clip(
                cfg.turn_match_yaw_rate_gain * self._target_yaw_rate
                + cfg.turn_match_course_gain * guidance_az,
                -cfg.turn_match_rate_limit_degps,
                cfg.turn_match_rate_limit_degps,
            ))
            desired_turn_rate += cfg.turn_match_los_rate_gain * guidance_los_rate
            desired_turn_rate = float(np.clip(
                desired_turn_rate,
                -cfg.turn_match_rate_limit_degps,
                cfg.turn_match_rate_limit_degps,
            ))
            # Do not roll away from a still-visible horizontal error merely
            # because target yaw-rate feed-forward has the opposite sign.
            # Once the error is nearly zero, release the guard so matching the
            # target circle can maintain the gun solution.
            if (
                abs(guidance_az) >= cfg.turn_match_sign_guard_error_deg
                and desired_turn_rate * guidance_az < 0.0
            ):
                desired_turn_rate = (
                    signed_unit(guidance_az)
                    * min(
                        cfg.turn_match_sign_guard_rate_degps,
                        max(0.5, cfg.turn_match_course_gain * abs(guidance_az)),
                    )
                )
        if terminal_track_active:
            desired_turn_rate = float(np.clip(
                cfg.terminal_track_yaw_rate_gain * self._target_yaw_rate
                + cfg.terminal_track_course_gain * guidance_az
                + cfg.terminal_track_los_rate_gain * guidance_los_rate,
                -cfg.terminal_track_rate_limit_degps,
                cfg.terminal_track_rate_limit_degps,
            ))

        # Convert desired horizontal course rate to coordinated bank demand.
        bank_from_rate = math.degrees(math.atan2(
            own_speed * math.radians(desired_turn_rate), G
        ))
        initial_commit = (
            cfg.initial_commit_duration_s > 0.0
            and elapsed <= cfg.initial_commit_duration_s
            and state == "break_turn"
            and ata > cfg.initial_commit_ata_deg
            and abs(bank) < cfg.initial_commit_release_bank_deg
        )
        if (
            cfg.coarse_turn_bank_deg > 0.0
            and cfg.coarse_turn_reentry_ata_deg > cfg.coarse_turn_ata_deg
        ):
            if not self._coarse_turn_initialized:
                self._coarse_turn_latched = ata >= cfg.coarse_turn_ata_deg
                self._coarse_turn_initialized = True
            elif self._coarse_turn_latched and ata < cfg.coarse_turn_ata_deg:
                self._coarse_turn_latched = False
            elif (
                not self._coarse_turn_latched
                and ata >= cfg.coarse_turn_reentry_ata_deg
            ):
                self._coarse_turn_latched = True
            coarse_turn = self._coarse_turn_latched
        else:
            coarse_turn = (
                cfg.coarse_turn_bank_deg > 0.0
                and ata >= cfg.coarse_turn_ata_deg
            )
        direct_course_bank = cfg.direct_course_bank_full_error_deg > 0.0
        if terminal_track_active or turn_match_active:
            target_bank = float(np.clip(
                bank_from_rate, -cfg.max_bank_deg, cfg.max_bank_deg
            ))
        elif direct_course_bank:
            # Re-evaluate directly from the newest horizontal target error on
            # every synchronized telemetry pair.  Only the tiny +/-180deg
            # ambiguity zone retains the previous side to prevent numerical
            # sign chatter at exactly six o'clock.
            if abs(guidance_az) < cfg.direct_course_rear_ambiguity_deg:
                direct_sign = signed_unit(guidance_az, self._turn_sign)
                self._turn_sign = direct_sign
            else:
                direct_sign = self._turn_sign
            error_fraction = float(np.clip(
                abs(guidance_az) / cfg.direct_course_bank_full_error_deg,
                0.0,
                1.0,
            ))
            target_bank = (
                direct_sign
                * cfg.max_bank_deg
                * error_fraction ** cfg.direct_course_bank_exponent
            )
        elif coarse_turn:
            target_bank = self._turn_sign * cfg.coarse_turn_bank_deg
        elif initial_commit:
            target_bank = self._turn_sign * cfg.initial_commit_bank_deg
        elif state == "break_turn" and ata > 60.0:
            target_bank = self._turn_sign * cfg.break_bank_deg
        else:
            target_bank = float(np.clip(bank_from_rate, -cfg.max_bank_deg, cfg.max_bank_deg))
        vertical_bank_relief_active = (
            cfg.vertical_bank_relief_elevation_deg > 0.0
            and ata <= cfg.vertical_maneuver_ata_gate_deg
            and abs(world_elevation) >= cfg.vertical_bank_relief_elevation_deg
        )
        if vertical_bank_relief_active:
            excess_elevation = (
                abs(world_elevation) - cfg.vertical_bank_relief_elevation_deg
            )
            relief_fraction = float(np.clip(excess_elevation / 35.0, 0.0, 1.0))
            bank_scale = 1.0 - relief_fraction * (
                1.0 - cfg.vertical_bank_relief_min_scale
            )
            target_bank *= bank_scale
        terminal_vertical_unload_active = (
            terminal_track_active
            and cfg.terminal_vertical_unload_el_deg > 0.0
            and abs(world_elevation - pitch)
            >= cfg.terminal_vertical_unload_el_deg
            and abs(world_elevation - pitch) >= (
                cfg.terminal_vertical_unload_ratio * abs(aim_az)
            )
        )
        if terminal_vertical_unload_active:
            # A large bank rotates elevator authority into the horizontal
            # plane.  When the terminal miss is substantially vertical,
            # unload toward wings-level so full pitch command can move the
            # nose in elevation before the high-closure pass completes.
            target_bank *= cfg.terminal_vertical_unload_bank_scale
        if (
            direct_course_bank
            and not terminal_track_active
            and not vertical_bank_relief_active
            and ata >= cfg.direct_course_min_bank_ata_deg
            and cfg.direct_course_min_bank_deg > 0.0
            and abs(target_bank) < cfg.direct_course_min_bank_deg
        ):
            target_bank = (
                signed_unit(target_bank, self._turn_sign)
                * cfg.direct_course_min_bank_deg
            )
        bank_error = wrap180(target_bank - bank)
        desired_roll_rate = float(np.clip(
            cfg.bank_kp * bank_error,
            -cfg.roll_rate_limit_degps,
            cfg.roll_rate_limit_degps,
        ))
        roll_cmd = float(np.clip(
            (desired_roll_rate - self._roll_rate) / cfg.roll_rate_gain_degps_per_unit,
            -cfg.roll_cmd_limit,
            cfg.roll_cmd_limit,
        ))
        bank_reversal_active = (
            cfg.bank_reversal_full_cmd > 0.0
            and target_bank * bank < 0.0
            and abs(bank_error) >= cfg.bank_reversal_error_deg
            and abs(bank) >= cfg.bank_reversal_release_bank_deg
        )
        if bank_reversal_active:
            # During a +80/-80 reversal the ordinary rate cascade removes
            # authority as soon as roll rate builds.  Hold full authority
            # until the old bank is mostly removed; then return to damping.
            roll_cmd = (
                signed_unit(target_bank) * cfg.bank_reversal_full_cmd
            )
        if coarse_turn and cfg.coarse_turn_roll_bias != 0.0:
            roll_cmd = float(np.clip(
                roll_cmd + self._turn_sign * cfg.coarse_turn_roll_bias,
                -cfg.roll_cmd_limit,
                cfg.roll_cmd_limit,
            ))
        if (
            direct_course_bank
            and cfg.direct_course_roll_bias != 0.0
            and abs(target_bank) > 5.0
        ):
            roll_cmd = float(np.clip(
                roll_cmd + signed_unit(target_bank) * cfg.direct_course_roll_bias,
                -cfg.roll_cmd_limit,
                cfg.roll_cmd_limit,
            ))
        if initial_commit and abs(roll_cmd) < cfg.initial_commit_min_roll_cmd:
            roll_cmd = self._turn_sign * cfg.initial_commit_min_roll_cmd
        max_test_active = (
            cfg.max_test_duration_s > 0.0
            and elapsed <= cfg.max_test_duration_s
            and (coarse_turn or not cfg.max_test_requires_coarse_turn)
        )
        if max_test_active:
            abs_bank = abs(bank)
            if abs_bank < cfg.max_test_roll_full_until_deg:
                roll_cmd = self._turn_sign * cfg.max_test_roll_full_cmd
            elif abs_bank < cfg.max_test_roll_taper_until_deg:
                roll_cmd = self._turn_sign * cfg.max_test_roll_taper_cmd
            elif abs_bank > cfg.max_test_roll_brake_above_deg:
                roll_cmd = -self._turn_sign * cfg.max_test_roll_brake_cmd

        horizontal_range = max(1.0, float(np.hypot(relative_position[0], relative_position[1])))
        predicted_target_altitude = target_altitude
        vertical_prediction_horizon = cfg.vertical_prediction_horizon_s
        vertical_prediction_stable = True
        if cfg.use_altitude_rate_vertical_prediction:
            vertical_prediction_stable = (
                self._target_vertical_stable_s
                >= cfg.vertical_prediction_min_stable_s
                and abs(self._target_vertical_accel)
                <= cfg.vertical_prediction_accel_limit_mps2
            )
            vertical_prediction_horizon = (
                cfg.vertical_prediction_stable_horizon_s
                if vertical_prediction_stable
                else cfg.vertical_prediction_unstable_horizon_s
            )
            predicted_target_altitude += (
                self._target_climb_rate * vertical_prediction_horizon
            )
        elif cfg.vertical_prediction_horizon_s > 0.0:
            # State D is down-positive, so positive D-rate means descending.
            predicted_target_altitude -= (
                self._target_velocity[2] * cfg.vertical_prediction_horizon_s
            )
        desired_gamma = math.degrees(math.atan2(
            predicted_target_altitude - altitude, horizontal_range
        ))
        if state in ("offensive_track", "weapons_track"):
            desired_gamma += cfg.fine_vertical_los_blend * aim_el
        gamma_limit = cfg.gamma_limit_deg
        if (
            cfg.vertical_maneuver_gamma_limit_deg > gamma_limit
            and ata <= cfg.vertical_maneuver_ata_gate_deg
        ):
            gamma_limit = cfg.vertical_maneuver_gamma_limit_deg
        defensive_escape_active = (
            cfg.defensive_vertical_escape
            and state == "defensive"
            and threat_ata <= cfg.defensive_escape_threat_ata_deg
            and ata >= cfg.defensive_escape_own_ata_deg
            and distance <= cfg.defensive_escape_range_m
        )
        defensive_escape_sign = 0
        if defensive_escape_active:
            altitude_separation = target_altitude - altitude
            if abs(altitude_separation) >= cfg.defensive_escape_deadband_m:
                # Increase vertical separation: descend from a higher bandit,
                # climb away from one below us.
                defensive_escape_sign = -signed_unit(altitude_separation)
            else:
                # At co-altitude, alternate the escape plane so a persistent
                # horizontal circle is not an easy tracking solution.
                phase = int(elapsed / max(cfg.defensive_escape_switch_s, 0.2))
                defensive_escape_sign = 1 if phase % 2 == 0 else -1
            desired_gamma = (
                defensive_escape_sign * cfg.defensive_escape_gamma_deg
            )
            gamma_limit = max(gamma_limit, cfg.defensive_escape_gamma_deg)
        desired_gamma = float(np.clip(desired_gamma, -gamma_limit, gamma_limit))
        flight_path_angle = math.degrees(math.asin(np.clip(vertical_speed / own_speed, -1.0, 1.0)))
        gamma_error = desired_gamma - flight_path_angle
        pitch_control_error = gamma_error
        pitch_control_kp = cfg.gamma_kp
        if terminal_track_active and cfg.terminal_pitch_attitude_kp > 0.0:
            pitch_control_error = world_elevation - pitch
            pitch_control_kp = cfg.terminal_pitch_attitude_kp
        desired_pitch_rate = float(np.clip(
            pitch_control_kp * pitch_control_error
            + (
                cfg.terminal_pitch_los_rate_gain * self._world_elevation_rate
                if terminal_track_active else 0.0
            ),
            -cfg.pitch_rate_limit_degps,
            cfg.pitch_rate_limit_degps,
        ))
        bank_fraction = float(np.clip(abs(bank) / cfg.max_bank_deg, 0.0, 1.0))
        turn_pull = cfg.turn_pull_at_max_bank * bank_fraction * bank_fraction
        pitch_cmd = float(np.clip(
            cfg.pitch_trim + turn_pull - desired_pitch_rate / cfg.pitch_rate_gain_degps_per_unit,
            -cfg.pitch_cmd_limit,
            cfg.pitch_cmd_limit,
        ))
        if cfg.vertical_speed_damping_gain > 0.0 and not (
            terminal_track_active and cfg.terminal_pitch_attitude_kp > 0.0
        ):
            vertical_speed_error = vertical_speed
            if cfg.vertical_speed_damping_track_desired_gamma:
                desired_vertical_speed = own_speed * math.sin(
                    math.radians(desired_gamma)
                )
                vertical_speed_error -= desired_vertical_speed
            excess_vertical_speed = math.copysign(
                max(0.0, abs(vertical_speed_error)
                    - cfg.vertical_speed_damping_deadband_mps),
                vertical_speed_error,
            )
            # Positive live vertical speed is climb; positive pitch_cmd is
            # nose-down, so this directly brakes a climb (and vice versa).
            pitch_cmd = float(np.clip(
                pitch_cmd
                + cfg.vertical_speed_damping_gain * excess_vertical_speed,
                -cfg.pitch_cmd_limit,
                cfg.pitch_cmd_limit,
            ))
        sustained_pull_cmd = 0.0
        sustained_pull_active = (
            cfg.sustained_pull_rear_cmd < 0.0
            and abs(bank) >= cfg.sustained_pull_min_bank_deg
            and own_speed >= cfg.sustained_pull_min_speed_mps
            and altitude >= cfg.sustained_pull_min_altitude_m
        )
        if sustained_pull_active:
            if ata >= cfg.sustained_pull_rear_ata_deg:
                sustained_pull_cmd = cfg.sustained_pull_rear_cmd
            elif (
                cfg.sustained_pull_mid_cmd < 0.0
                and ata >= cfg.sustained_pull_mid_ata_deg
            ):
                sustained_pull_cmd = cfg.sustained_pull_mid_cmd
            if sustained_pull_cmd < 0.0:
                pitch_cmd = max(-cfg.pitch_cmd_limit, min(pitch_cmd, sustained_pull_cmd))
        max_test_pull_active = (
            max_test_active
            and cfg.max_test_pitch_cmd < 0.0
            and abs(bank) >= cfg.max_test_pitch_bank_gate_deg
            and ata >= cfg.max_test_pitch_ata_gate_deg
            and own_speed >= cfg.max_test_min_speed_mps
            and altitude >= cfg.max_test_min_altitude_m
        )
        if max_test_pull_active:
            if (
                cfg.max_test_pitch_vertical_gain > 0.0
                and cfg.max_test_pitch_relaxed_cmd > cfg.max_test_pitch_cmd
            ):
                test_pitch_cmd = (
                    cfg.max_test_pitch_cmd
                    + cfg.max_test_pitch_vertical_gain
                    * (vertical_speed - cfg.max_test_target_vertical_speed_mps)
                )
                if (
                    cfg.max_test_pitch_bank_relax_gain > 0.0
                    and cfg.coarse_turn_bank_deg > 0.0
                ):
                    test_pitch_cmd += (
                        cfg.max_test_pitch_bank_relax_gain
                        * max(0.0, cfg.coarse_turn_bank_deg - abs(bank))
                    )
                pitch_cmd = float(np.clip(
                    test_pitch_cmd,
                    cfg.max_test_pitch_cmd,
                    cfg.max_test_pitch_relaxed_cmd,
                ))
            else:
                pitch_cmd = max(-cfg.pitch_cmd_limit, min(pitch_cmd, cfg.max_test_pitch_cmd))

        if state == "merge":
            target_throttle = cfg.throttle_merge
        elif state == "defensive":
            target_throttle = cfg.throttle_defensive
        elif state == "weapons_track":
            target_throttle = cfg.throttle_weapons
        elif state == "offensive_track":
            desired_speed = float(np.clip(target_speed + 15.0, 230.0, 330.0))
            target_throttle = cfg.corner_throttle_base + cfg.corner_speed_gain * (desired_speed - own_speed)
        else:
            target_throttle = cfg.corner_throttle_base + cfg.corner_speed_gain * (
                cfg.corner_speed_mps - own_speed
            )
        desired_closure = None
        if (
            cfg.closure_throttle_gain > 0.0
            and ata <= cfg.closure_throttle_ata_deg
            and distance <= cfg.closure_throttle_range_m
        ):
            if distance >= cfg.closure_throttle_far_range_m:
                desired_closure = cfg.closure_target_far_mps
            elif distance >= cfg.closure_throttle_near_range_m:
                desired_closure = cfg.closure_target_mid_mps
            else:
                desired_closure = cfg.closure_target_near_mps
            if vertical_alignment_active:
                desired_closure = min(
                    desired_closure,
                    cfg.vertical_alignment_target_closure_mps,
                )
            target_throttle = (
                cfg.closure_throttle_base
                - cfg.closure_throttle_gain * (closure - desired_closure)
            )
        energy_recovery_active = (
            cfg.minimum_energy_speed_mps > 0.0
            and own_speed < cfg.minimum_energy_speed_mps
        )
        if energy_recovery_active:
            target_throttle = max(
                target_throttle, cfg.minimum_energy_throttle
            )
        target_throttle = float(np.clip(target_throttle, cfg.throttle_min, cfg.throttle_max))
        if dt:
            max_step = cfg.throttle_slew_per_s * dt
            self._throttle += float(np.clip(target_throttle - self._throttle, -max_step, max_step))
        else:
            self._throttle = target_throttle
        throttle = float(np.clip(self._throttle, cfg.throttle_min, cfg.throttle_max))

        action = np.array([roll_cmd, pitch_cmd, 0.0, throttle], dtype=np.float32)
        info = {
            "state": state,
            "ata": ata,
            "aa": aa,
            "threat_ata": threat_ata,
            "distance": distance,
            "closure_rate": closure,
            "los_az": los_az,
            "los_el": los_el,
            "aim_az": guidance_az,
            "aim_course_error": aim_course_error,
            "aim_el": aim_el,
            "los_rate": guidance_los_rate,
            "ata_rate": self._ata_rate,
            "own_yaw_rate": self._own_yaw_rate,
            "target_yaw_rate": self._target_yaw_rate,
            "target_yaw_accel": self._target_yaw_accel,
            "target_turn_stable_s": self._target_turn_stable_s,
            "prediction_stable": prediction_stable,
            "intercept_horizon": horizon,
            "lag_pursuit_active": lag_pursuit_active,
            "lag_offset_m": lag_offset_m,
            "vertical_alignment_active": vertical_alignment_active,
            "turn_match_active": turn_match_active,
            "turn_match_candidate_s": self._turn_match_candidate_s,
            "terminal_track_active": terminal_track_active,
            "terminal_vertical_unload_active": terminal_vertical_unload_active,
            "desired_turn_rate": desired_turn_rate,
            "target_bank": target_bank,
            "current_bank": bank,
            "bank_error": bank_error,
            "bank_reversal_active": bank_reversal_active,
            "desired_roll_rate": desired_roll_rate,
            "measured_roll_rate": self._roll_rate,
            "desired_gamma": desired_gamma,
            "world_elevation": world_elevation,
            "world_elevation_rate": self._world_elevation_rate,
            "target_climb_rate": self._target_climb_rate,
            "target_vertical_accel": self._target_vertical_accel,
            "target_vertical_stable_s": self._target_vertical_stable_s,
            "vertical_prediction_horizon": vertical_prediction_horizon,
            "vertical_prediction_stable": vertical_prediction_stable,
            "vertical_bank_relief_active": vertical_bank_relief_active,
            "defensive_escape_active": defensive_escape_active,
            "defensive_escape_sign": defensive_escape_sign,
            "post_defense_conversion_active": post_defense_conversion_active,
            "post_defense_conversion_armed": (
                now < self._post_defense_conversion_armed_until
            ),
            "conversion_rear_offset_m": conversion_rear_offset_m,
            "conversion_lateral_offset_m": conversion_lateral_offset_m,
            "flight_path_angle": flight_path_angle,
            "gamma_error": gamma_error,
            "desired_pitch_rate": desired_pitch_rate,
            "measured_pitch_rate": self._pitch_rate,
            "turn_pitch_feedforward": turn_pull,
            "own_pitch": pitch,
            "own_alt": altitude,
            "vertical_speed": vertical_speed,
            "target_speed": target_speed,
            "speed_error": cfg.corner_speed_mps - own_speed,
            "target_throttle": target_throttle,
            "desired_closure": desired_closure,
            "energy_recovery_active": energy_recovery_active,
            "roll_cmd": roll_cmd,
            "initial_commit": initial_commit,
            "coarse_turn": coarse_turn,
            "max_test_active": max_test_active,
            "pitch_cmd": pitch_cmd,
            "sustained_pull_active": sustained_pull_cmd < 0.0,
            "sustained_pull_cmd": sustained_pull_cmd,
            "max_test_pull_active": max_test_pull_active,
        }

        self._prev_time = now
        self._prev_distance = distance
        self._prev_ata = ata
        self._prev_los_az = los_az
        self._prev_horizontal_los_error = horizontal_los_error
        self._prev_own_yaw = own_yaw
        self._prev_target_yaw = target_yaw
        self._prev_bank = bank
        self._prev_pitch = pitch
        self._prev_alt = altitude
        self._prev_target_alt = target_altitude
        self._prev_target_position = target_position.copy()
        self.state_log.append(state)
        self.info_log.append(info)
        return ActionResult(
            action=action,
            source=f"{cfg.controller_name}[{state}]",
            confidence=1.0,
            info=info,
        )
