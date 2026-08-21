"""Stable bank-to-turn pursuit controller for live 1-v-1 evaluation.

The controller deliberately keeps the learned policy in the loop only near a
firing solution.  At large ATA it uses a conventional two-loop structure:

* outer loop: choose a target bank from LOS azimuth;
* inner loop: hold that bank and pull only after the bank is established;
* envelope loop: prevent an unbounded zoom/dive using world pitch and altitude;
* fine loop: blend back toward the learned policy as ATA approaches the WEZ.

Unlike the old tactical wrapper, it never commands a fixed roll input for a
fixed time and never derives sustained pull solely from body LOS elevation.
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
class PursuitControllerConfig:
    # Geometry / hand-off.
    fine_ata_deg: float = 8.0
    track_ata_deg: float = 3.0
    weapon_range_m: float = 1219.2
    rule_weight_far: float = 0.95
    rule_weight_fine: float = 0.65

    # Bank-to-turn outer/inner loops.
    max_bank_deg: float = 70.0
    min_turn_bank_deg: float = 25.0
    full_bank_az_deg: float = 60.0
    turn_sign_deadband_deg: float = 4.0
    turn_sign_flip_deg: float = 20.0
    bank_error_norm_deg: float = 28.0
    roll_ema_alpha: float = 0.45
    roll_delta_limit: float = 0.22

    # Pull loop.  Negative pitch command is nose-up in the live protocol.
    pull_ata_full_deg: float = 70.0
    max_pull: float = 0.85
    min_pull: float = 0.18
    bank_alignment_full_deg: float = 15.0
    bank_alignment_zero_deg: float = 65.0
    pitch_delta_limit: float = 0.18

    # World-frame pitch/altitude envelope.  These are intentionally firm:
    # run19-21 climbed from 4.6 km to 8 km while still missing the target.
    climb_soft_deg: float = 18.0
    climb_hard_deg: float = 30.0
    dive_soft_deg: float = -18.0
    dive_hard_deg: float = -30.0
    altitude_floor_m: float = 900.0
    altitude_recovery_m: float = 1400.0

    # Energy / closure management.
    throttle_turn: float = 0.92
    throttle_track: float = 0.62
    high_closure_mps: float = 350.0


class PursuitControllerActionProvider(ActionProvider):
    """Rule-guided pursuit with a learned-policy fine-tracking blend."""

    def __init__(self, inner: ActionProvider, config: PursuitControllerConfig | None = None):
        self.inner = inner
        self.cfg = config or PursuitControllerConfig()
        self.geometry = GeometryInfo()
        self._turn_sign = 0
        self._prev_distance: float | None = None
        self._prev_time: float | None = None
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self.state_log: list[str] = []

    def reset(self, context: ActionContext | None = None) -> None:
        self.inner.reset(context)
        self._turn_sign = 0
        self._prev_distance = None
        self._prev_time = None
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self.state_log = []

    def compute_action(self, context: ActionContext) -> ActionResult:
        learned = self.inner.compute_action(context)
        rl = np.asarray(learned.action, dtype=np.float32)
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            return ActionResult(rl, "pursuit_passthrough", learned.confidence)

        cfg = self.cfg
        now = float(own[StateIndex.SIM_TIME])
        distance = float(self.geometry._get_distance(own, target))
        ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
        az, el = self.geometry._get_los_angle(own, target)
        az, el = float(az), float(el)
        closure = 0.0
        if self._prev_distance is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 1e-3:
                closure = (self._prev_distance - distance) / dt

        # Commit to a turn side, but permit a reversal only when the target is
        # clearly across the nose.  There is no fixed-time/full-roll command.
        desired_sign = 1 if az >= 0.0 else -1
        if self._turn_sign == 0:
            self._turn_sign = desired_sign
        elif desired_sign != self._turn_sign and abs(az) >= cfg.turn_sign_flip_deg:
            self._turn_sign = desired_sign
        elif abs(az) <= cfg.turn_sign_deadband_deg:
            desired_sign = self._turn_sign

        az_fraction = float(np.clip(abs(az) / cfg.full_bank_az_deg, 0.0, 1.0))
        target_bank_mag = cfg.min_turn_bank_deg + az_fraction * (
            cfg.max_bank_deg - cfg.min_turn_bank_deg
        )
        if ata <= cfg.fine_ata_deg:
            target_bank_mag *= float(np.clip(ata / cfg.fine_ata_deg, 0.0, 1.0))
        target_bank = self._turn_sign * target_bank_mag
        current_bank = _wrap180(float(own[StateIndex.ROLL]))
        bank_error = _wrap180(target_bank - current_bank)
        roll_raw = float(np.clip(bank_error / cfg.bank_error_norm_deg, -1.0, 1.0))
        roll_filtered = cfg.roll_ema_alpha * roll_raw + (1.0 - cfg.roll_ema_alpha) * self._roll_cmd
        self._roll_cmd += float(np.clip(
            roll_filtered - self._roll_cmd, -cfg.roll_delta_limit, cfg.roll_delta_limit
        ))

        # Pull only when the lift vector is reasonably aligned with the turn
        # plane.  This avoids the v12-v14 failure: hard pull while still
        # banking, which converted the intercept into a zoom climb.
        alignment = 1.0 - float(np.clip(
            (abs(bank_error) - cfg.bank_alignment_full_deg)
            / (cfg.bank_alignment_zero_deg - cfg.bank_alignment_full_deg),
            0.0,
            1.0,
        ))
        pull_mag = cfg.min_pull + float(np.clip(ata / cfg.pull_ata_full_deg, 0.0, 1.0)) * (
            cfg.max_pull - cfg.min_pull
        )
        pitch_raw = -pull_mag * alignment

        # When nearly wings-level, body elevation is trustworthy and permits
        # a direct push for a target below instead of always pulling upward.
        if abs(current_bank) < 20.0 and abs(el) > 4.0:
            pitch_raw = float(np.clip(-el / 35.0, -cfg.max_pull, cfg.max_pull))

        own_pitch = float(own[StateIndex.PITCH])
        if own_pitch > cfg.climb_soft_deg:
            allowed_pull = cfg.max_pull * float(np.clip(
                (cfg.climb_hard_deg - own_pitch)
                / (cfg.climb_hard_deg - cfg.climb_soft_deg), 0.0, 1.0
            ))
            pitch_raw = max(pitch_raw, -allowed_pull)
        if own_pitch < cfg.dive_soft_deg:
            allowed_push = cfg.max_pull * float(np.clip(
                (own_pitch - cfg.dive_hard_deg)
                / (cfg.dive_soft_deg - cfg.dive_hard_deg), 0.0, 1.0
            ))
            pitch_raw = min(pitch_raw, allowed_push)

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
        # Envelope limits are safety constraints, not tracking requests: do
        # not let the command-rate limiter delay their application.
        if own_pitch >= cfg.climb_hard_deg:
            self._pitch_cmd = max(self._pitch_cmd, 0.0)
        elif own_pitch <= cfg.dive_hard_deg:
            self._pitch_cmd = min(self._pitch_cmd, 0.0)

        if ata <= cfg.track_ata_deg and distance <= cfg.weapon_range_m:
            state = "weapons_track"
            rule_weight = cfg.rule_weight_fine
            throttle = cfg.throttle_track
            if closure > cfg.high_closure_mps:
                throttle = 0.35
        elif ata <= cfg.fine_ata_deg:
            state = "fine_pursuit"
            rule_weight = cfg.rule_weight_fine
            throttle = cfg.throttle_track
        else:
            state = "turn_pursuit"
            rule_weight = cfg.rule_weight_far
            throttle = cfg.throttle_turn

        rule = np.array([self._roll_cmd, self._pitch_cmd, 0.0, throttle], dtype=np.float32)
        action = rule_weight * rule + (1.0 - rule_weight) * rl
        action = np.clip(action, [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]).astype(np.float32)

        self._prev_distance = distance
        self._prev_time = now
        self.state_log.append(state)
        return ActionResult(
            action=action,
            source=f"pursuit_controller[{state}]",
            confidence=learned.confidence,
            info={
                "state": state,
                "ata": ata,
                "distance": distance,
                "closure_rate": closure,
                "los_az": az,
                "los_el": el,
                "target_bank": target_bank,
                "bank_error": bank_error,
            },
        )
