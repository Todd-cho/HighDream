"""Simple lead-pursuit pull handoff: a from-scratch, deliberately minimal
rule that only fires when the target is FAR (distance-triggered, not
ATA-triggered) -- bank+pull toward the target's PREDICTED future position
(current position + estimated velocity * lead_time), so a turn can go
left/right or up/down and actually leads a maneuvering target instead of
just pointing at where it is right now. Once distance comes back inside
the engagement band, this hands straight back to the raw RL v8 policy
(no blending) -- "안정적 비행" only happens there, not in the rule.

Revision history:
- v1 (2026-08-25 morning): ATA-triggered hard handoff, pure-pursuit
  (aimed at the target's CURRENT position). JSBSim smoke test: mean
  distance blew up to ~17-18km, wez_episode_rate=0%, despite decent
  instantaneous ATA convergence 30% of the time. Diagnosis: aiming at
  the current position of a maneuvering target is pure pursuit, which is
  known to fail to close range against a target that is itself turning
  toward you (this project's own scripted_pursuit opponent does exactly
  that) -- this is the same problem `lead_pursuit_scale` in
  my_reward_delta_v1.py and W108's "LOS-rate acceleration vector" were
  built to address.
- v2: (a) predicts the target's future position from a finite-difference
  velocity estimate (native VP getters return 0 in JSBSim, same reason
  observation.py's tactical19 builder computes it manually) and banks/
  pulls toward THAT point instead of the target's current position; (b)
  the RL<->rule trigger is DISTANCE-based (simple: far -> pull, close ->
  hand back to RL) instead of ATA-based, per explicit user instruction --
  ATA-based hysteresis kept firing the rule even when the aircraft was
  already pointed correctly but simply far away, which was part of why
  v1 never closed range. JSBSim smoke test: mean distance improved
  (18km -> 12km) but rule_step_fraction=94% -- the aircraft spent almost
  the whole flight "far", meaning it wasn't actually closing range even
  while the rule had full authority.
- v3 (this revision): two BFM pieces that already existed elsewhere in
  this codebase (run_unreal_inference.py's w104/w105 modes) but were
  never carried into this handoff:
  (a) corner-speed energy management -- w104/w105's own comment records a
  live plant audit (run0191) where a 195 m/s target cut throttle to 0.27
  while the opponent sustained 214-224 m/s; they settled on 220 m/s
  (`high_bank_target_speed_mps`/`lag_pursuit_energy_target_speed_mps`).
  Reused here verbatim: below corner speed, pull magnitude is throttled
  back so the turn doesn't bleed the last of the aircraft's energy.
  (b) lag pursuit -- classic BFM doctrine (and w104/105's own
  `lag_pursuit_energy_target_speed_mps` field) uses lead pursuit to close
  range efficiently, then deliberately drops to a LAG aim point (behind
  the target, not ahead) once closure rate is high and range is short, to
  bleed off overtake speed instead of flying through the merge. v1/v2's
  failure signature (a brief opponent-driven close pass, then a sustained
  ~12-18km divergence for the rest of the flight) is consistent with
  exactly the overshoot lag pursuit exists to prevent -- v1/v2 kept
  pulling lead the entire time, including in the merge itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


@dataclass
class LeadPullConfig:
    # Distance-based hard handoff (hysteresis so a boundary-straddling
    # episode doesn't flap RL<->rule every frame).
    enter_far_m: float = 2200.0   # switch to rule once distance exceeds this
    exit_far_m: float = 1400.0    # switch back to RL once distance drops below this

    # Lead prediction: aim at target_pos + target_velocity * lead_time_s,
    # not the target's current position.
    lead_time_s: float = 3.0
    velocity_clip_mps: float = 60.0  # clamp finite-difference noise

    # Bank law: target_bank_deg = atan2(los_az, los_el) to the PREDICTED
    # point, so pull direction tracks both azimuth and elevation at once.
    bank_error_norm_deg: float = 30.0
    roll_delta_limit: float = 0.20
    turn_sign_flip_deg: float = 25.0  # commit hysteresis (2026-08-25 v1 lesson)

    # Pull magnitude scales with angle-off to the predicted point.
    min_pull: float = 0.25
    max_pull: float = 0.90
    pull_angle_full_deg: float = 90.0
    pitch_delta_limit: float = 0.20

    # World-frame pitch/altitude envelope -- kept firm on purpose (the
    # P-series' own history: unbounded nose-up pull turned pursuits into
    # zoom climbs that never closed). SafetyOverrideCommandPolicy is the
    # real safety net; this just stops the rule from actively fighting it.
    climb_hard_deg: float = 35.0
    dive_hard_deg: float = -35.0
    altitude_floor_m: float = 900.0
    altitude_recovery_m: float = 1400.0

    throttle_rule: float = 1.0

    # Corner-speed energy management (live-measured value reused from
    # w104/w105's high_bank_target_speed_mps / lag_pursuit_energy_target_speed_mps).
    corner_speed_mps: float = 220.0
    energy_pull_floor: float = 0.55  # never cut pull below this fraction even far under corner speed
    energy_speed_deficit_full_mps: float = 60.0  # deficit at which pull is cut to energy_pull_floor

    # Lag pursuit: switch the aim point from lead to lag once closing fast
    # and inside lag_range_m, to bleed overtake speed instead of overshooting.
    lag_closure_trigger_mps: float = 60.0
    lag_range_m: float = 1500.0
    lag_time_s: float = 2.0
    lag_throttle: float = 0.55


class LeadPullHandoffActionProvider(ActionProvider):
    """Distance-triggered hard handoff. Rule = lead-pursuit pull toward the
    target's predicted position. RL = raw, unblended, whenever distance is
    inside the engagement band."""

    def __init__(self, rl_provider: ActionProvider, config: LeadPullConfig | None = None) -> None:
        self.rl_provider = rl_provider
        self.cfg = config or LeadPullConfig()
        self.geometry = GeometryInfo()
        self._far = False
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self._bank_commit: float | None = None
        self._prev_target_pos: np.ndarray | None = None
        self._prev_sim_time: float | None = None
        self._prev_distance: float | None = None
        self._prev_distance_time: float | None = None
        self.state_log: list[str] = []
        self.info_log: list[dict] = []

    def reset(self, context: ActionContext | None = None) -> None:
        self.rl_provider.reset(context)
        self._far = False
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self._bank_commit = None
        self._prev_target_pos = None
        self._prev_sim_time = None
        self._prev_distance = None
        self._prev_distance_time = None
        self.state_log = []
        self.info_log = []

    def _estimate_target_velocity(self, target: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        pos = np.asarray(target[0:3], dtype=np.float64)
        sim_time = float(target[StateIndex.SIM_TIME]) if len(target) > StateIndex.SIM_TIME else None
        velocity = np.zeros(3, dtype=np.float64)
        if (
            self._prev_target_pos is not None
            and self._prev_sim_time is not None
            and sim_time is not None
            and sim_time > self._prev_sim_time
        ):
            dt = sim_time - self._prev_sim_time
            velocity = (pos - self._prev_target_pos) / dt
            velocity = np.clip(velocity, -cfg.velocity_clip_mps, cfg.velocity_clip_mps)
        self._prev_target_pos = pos
        self._prev_sim_time = sim_time
        return velocity

    def compute_action(self, context: ActionContext) -> ActionResult:
        cfg = self.cfg
        rl_result = self.rl_provider.compute_action(context)
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            self.state_log.append("rl")
            return ActionResult(rl_result.action, "lead_pull_passthrough", rl_result.confidence)

        distance = float(self.geometry._get_distance(own, target))
        target_velocity = self._estimate_target_velocity(target)

        own_sim_time = float(own[StateIndex.SIM_TIME])
        closure_rate = 0.0
        if self._prev_distance is not None and self._prev_distance_time is not None:
            dt = own_sim_time - self._prev_distance_time
            if dt > 1e-3:
                closure_rate = (self._prev_distance - distance) / dt
        self._prev_distance = distance
        self._prev_distance_time = own_sim_time

        if self._far:
            self._far = distance >= cfg.exit_far_m
        else:
            self._far = distance >= cfg.enter_far_m

        if not self._far:
            self.state_log.append("rl")
            info = {"mode": "rl", "distance": distance}
            self.info_log.append(info)
            return ActionResult(rl_result.action, f"lead_pull[rl]<{rl_result.source}>", rl_result.confidence, info)

        # Lag pursuit: closing fast and already inside lag range -> aim
        # BEHIND the target (negative time) to bleed overtake speed instead
        # of flying through the merge with a lead solution the whole time.
        in_lag = closure_rate >= cfg.lag_closure_trigger_mps and distance <= cfg.lag_range_m
        aim_time = -cfg.lag_time_s if in_lag else cfg.lead_time_s

        target_pred = np.array(target, dtype=np.float64, copy=True)
        target_pred[0:3] = target_pred[0:3] + target_velocity * aim_time

        ata_pred = abs(float(self.geometry._get_antenna_train_angle(own, target_pred, False)))
        az, el = self.geometry._get_los_angle(own, target_pred)
        az, el = float(az), float(el)

        # Lift-vector pointing: bank so a pull closes BOTH az and el at once.
        raw_bank = float(np.degrees(np.arctan2(az, el)))
        if self._bank_commit is None:
            self._bank_commit = raw_bank
        else:
            delta = _wrap180(raw_bank - self._bank_commit)
            if abs(delta) >= cfg.turn_sign_flip_deg:
                self._bank_commit = raw_bank
        target_bank = self._bank_commit

        current_bank = _wrap180(float(own[StateIndex.ROLL]))
        bank_error = _wrap180(target_bank - current_bank)
        roll_raw = float(np.clip(bank_error / cfg.bank_error_norm_deg, -1.0, 1.0))
        self._roll_cmd += float(np.clip(
            roll_raw - self._roll_cmd, -cfg.roll_delta_limit, cfg.roll_delta_limit
        ))

        pull_mag = cfg.min_pull + float(np.clip(ata_pred / cfg.pull_angle_full_deg, 0.0, 1.0)) * (
            cfg.max_pull - cfg.min_pull
        )

        # Corner-speed energy management (220 m/s, live-measured in
        # w104/w105): pulling hard while well below corner speed just bleeds
        # the last of the aircraft's energy, so taper pull magnitude down
        # (never below energy_pull_floor) as the speed deficit grows.
        own_speed = float(own[StateIndex.KCAS])
        speed_deficit = max(0.0, cfg.corner_speed_mps - own_speed)
        energy_scale = 1.0 - float(np.clip(
            speed_deficit / max(1e-3, cfg.energy_speed_deficit_full_mps), 0.0, 1.0
        )) * (1.0 - cfg.energy_pull_floor)
        pull_mag *= energy_scale

        pitch_raw = -pull_mag

        own_pitch = float(own[StateIndex.PITCH])
        if own_pitch >= cfg.climb_hard_deg:
            pitch_raw = max(pitch_raw, 0.0)
        elif own_pitch <= cfg.dive_hard_deg:
            pitch_raw = min(pitch_raw, 0.0)

        altitude = float(own[StateIndex.ALT])
        if altitude <= cfg.altitude_floor_m:
            pitch_raw = min(pitch_raw, -0.65)
            self._roll_cmd = float(np.clip(self._roll_cmd, -0.3, 0.3))
        elif altitude < cfg.altitude_recovery_m:
            recovery = (cfg.altitude_recovery_m - altitude) / (
                cfg.altitude_recovery_m - cfg.altitude_floor_m
            )
            pitch_raw = min(pitch_raw, -0.65 * recovery)

        self._pitch_cmd += float(np.clip(
            pitch_raw - self._pitch_cmd, -cfg.pitch_delta_limit, cfg.pitch_delta_limit
        ))

        throttle = cfg.lag_throttle if in_lag else cfg.throttle_rule
        action = np.array(
            [self._roll_cmd, self._pitch_cmd, 0.0, throttle], dtype=np.float32
        )
        action = np.clip(action, [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]).astype(np.float32)

        self.state_log.append("rule_lag" if in_lag else "rule_lead")
        info = {
            "mode": "rule_lag" if in_lag else "rule_lead",
            "distance": distance,
            "closure_rate": closure_rate,
            "ata_pred": ata_pred,
            "los_az_pred": az,
            "los_el_pred": el,
            "target_velocity": target_velocity.tolist(),
            "target_bank": target_bank,
            "current_bank": current_bank,
            "bank_error": bank_error,
            "roll_cmd": self._roll_cmd,
            "pitch_cmd": self._pitch_cmd,
            "own_speed": own_speed,
            "energy_scale": energy_scale,
        }
        self.info_log.append(info)
        return ActionResult(action, "lead_pull[rule]", rl_result.confidence, info)

    def close(self) -> None:
        self.rl_provider.close()
