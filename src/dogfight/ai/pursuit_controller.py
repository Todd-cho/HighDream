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

    # Ablation switches (2026-08-21, P0-P3 diagnostic per user's protocol
    # after JSBSim showed mean_distance=16.1km / wez_rate=16.7% -- the
    # aircraft essentially never closes. Each isolates ONE candidate cause
    # instead of retuning magnitudes blind, per the user's explicit
    # instruction not to touch rule_weight_far/pull size yet.
    disable_alignment_gate: bool = False  # P1: alignment forced to 1.0
    disable_pitch_envelope: bool = False  # P2: skip climb/dive world-attitude caps
    flip_bank_sign: bool = False  # P3: target_bank negated before bank_error
    world_frame_pitch_gate: bool = False  # P4: no nose-up pull unless target is actually above (world frame)
    world_frame_pitch_gate_margin_m: float = 0.0
    # P5: additive turn_pull + altitude_correction + vertical_damping
    # decomposition, replacing P4's hard clamp -- see the compute_action
    # comment where turn_pull_decomposition is consumed for the full
    # rationale (user diagnosis, 2026-08-21: P4 zeroed the coordinated-turn
    # back-pressure along with the excess climb pull it was meant to stop).
    turn_pull_decomposition: bool = False
    turn_pull_max_bank_deg: float = 62.0
    turn_pull_gain: float = 0.4
    alt_correction_gain: float = 0.0002
    alt_correction_cap_m: float = 500.0
    vertical_damping_gain: float = 0.004
    # P6: vertical-speed PRIORITY gate on turn_pull itself (not just an
    # additive opposing term) -- see compute_action comment where
    # turn_pull_priority_gate is consumed for the full rationale.
    turn_pull_priority_gate: bool = False
    vspeed_soft_mps: float = 15.0
    vspeed_hard_mps: float = 40.0
    turn_pull_scale_delta_limit: float = 0.08

    # 2026-08-25: bank law generalized to 2-D LOS (az AND el) instead of az
    # alone, so a target well above/below the nose gets pulled toward
    # directly (bank = atan2(az, el)) instead of being flattened into a
    # horizontal-only turn. Uses the same "commit, only flip on a confident
    # reversal" hysteresis as the az-only turn_sign law above, generalized
    # to the full signed bank angle. rule_weight_far/rule_weight_fine let
    # the caller turn this into a hard handoff (1.0/0.0) instead of a blend.
    omnidirectional_bank: bool = False


class PursuitControllerActionProvider(ActionProvider):
    """Rule-guided pursuit with a learned-policy fine-tracking blend."""

    def __init__(self, inner: ActionProvider, config: PursuitControllerConfig | None = None):
        self.inner = inner
        self.cfg = config or PursuitControllerConfig()
        self.geometry = GeometryInfo()
        self._turn_sign = 0
        self._prev_distance: float | None = None
        self._prev_time: float | None = None
        self._prev_own_alt: float | None = None
        self._turn_pull_scale = 1.0
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self._bank_commit: float | None = None
        self.state_log: list[str] = []
        self.info_log: list[dict] = []

    def reset(self, context: ActionContext | None = None) -> None:
        self.inner.reset(context)
        self._turn_sign = 0
        self._prev_distance = None
        self._prev_time = None
        self._prev_own_alt = None
        self._turn_pull_scale = 1.0
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self._bank_commit = None
        self.state_log = []
        self.info_log = []

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
        aa = abs(float(self.geometry._get_aspect_angle(own, target, False)))
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

        max_bank_deg = cfg.turn_pull_max_bank_deg if cfg.turn_pull_decomposition else cfg.max_bank_deg
        if cfg.omnidirectional_bank:
            raw_bank = float(np.clip(
                float(np.degrees(np.arctan2(az, el))), -max_bank_deg, max_bank_deg
            ))
            if self._bank_commit is None:
                self._bank_commit = raw_bank
            else:
                delta = _wrap180(raw_bank - self._bank_commit)
                if abs(delta) >= cfg.turn_sign_flip_deg:
                    self._bank_commit = raw_bank
            target_bank = self._bank_commit
            if ata <= cfg.fine_ata_deg:
                target_bank *= float(np.clip(ata / cfg.fine_ata_deg, 0.0, 1.0))
        else:
            az_fraction = float(np.clip(abs(az) / cfg.full_bank_az_deg, 0.0, 1.0))
            target_bank_mag = cfg.min_turn_bank_deg + az_fraction * (
                max_bank_deg - cfg.min_turn_bank_deg
            )
            if ata <= cfg.fine_ata_deg:
                target_bank_mag *= float(np.clip(ata / cfg.fine_ata_deg, 0.0, 1.0))
            target_bank = self._turn_sign * target_bank_mag
        if cfg.flip_bank_sign:  # P3 ablation
            target_bank = -target_bank
        current_bank = _wrap180(float(own[StateIndex.ROLL]))
        bank_error = _wrap180(target_bank - current_bank)
        roll_raw = float(np.clip(bank_error / cfg.bank_error_norm_deg, -1.0, 1.0))
        roll_filtered = cfg.roll_ema_alpha * roll_raw + (1.0 - cfg.roll_ema_alpha) * self._roll_cmd
        self._roll_cmd += float(np.clip(
            roll_filtered - self._roll_cmd, -cfg.roll_delta_limit, cfg.roll_delta_limit
        ))

        own_pitch = float(own[StateIndex.PITCH])
        alignment = float("nan")  # not used by P5's pitch law; kept for info_log shape

        if cfg.turn_pull_decomposition:
            # P5 (2026-08-21, user diagnosis of P4's live failure at 91deg):
            # P4's hard clamp (pitch_raw=max(pitch_raw,0) whenever the
            # target isn't above in world frame) killed ALL nose-up pull in
            # that condition -- including the coordinated-turn back-pressure
            # every banked turn needs just to hold altitude and actually
            # turn the velocity vector. Without it the aircraft banks
            # (confirmed live, run0043 ang091: roll held ~70-74deg the whole
            # flight) but heading barely tracks the LOS, so ATA never
            # converges even though bank/altitude look perfectly stable.
            # P5 replaces the all-or-nothing gate with an additive
            # decomposition: a coordinated-turn pull that scales with bank
            # angle alone (present regardless of target position, since
            # holding ANY bank needs it) plus a small world-frame altitude
            # correction and a vertical-rate damping term, instead of one
            # hard floor. Bank is also capped lower here
            # (turn_pull_max_bank_deg, default 62 vs the 70 used elsewhere)
            # since the coordinated-turn pull itself grows sharply near
            # 90deg (1/cos blows up).
            bank_rad = float(np.radians(min(abs(current_bank), 89.0)))
            turn_pull = -cfg.turn_pull_gain * (1.0 / np.cos(bank_rad) - 1.0)
            turn_pull = float(np.clip(turn_pull, -cfg.max_pull, 0.0))

            target_alt = float(target[StateIndex.ALT])
            own_alt_now = float(own[StateIndex.ALT])
            alt_diff = float(np.clip(
                target_alt - own_alt_now, -cfg.alt_correction_cap_m, cfg.alt_correction_cap_m
            ))
            altitude_correction = -cfg.alt_correction_gain * alt_diff

            vertical_speed = 0.0
            if self._prev_own_alt is not None and self._prev_time is not None:
                dt = now - self._prev_time
                if dt > 1e-3:
                    vertical_speed = (own_alt_now - self._prev_own_alt) / dt
            vertical_damping = cfg.vertical_damping_gain * vertical_speed

            if cfg.turn_pull_priority_gate:
                # P6 (2026-08-21, user diagnosis of P5b's 60s failure): P5b's
                # gain (vertical_damping_gain=0.008, computed from P5's own
                # logged turn_pull/vspeed ratio) DOES work within a single
                # turn commitment (own_pitch went positive->negative,
                # altitude briefly stabilized ~t=37-43s) -- but every time
                # turn_sign flips (a new bank-direction commit), the whole
                # climb/recover cycle restarts from scratch, so vertical_speed
                # never gets a chance to actually settle before the next
                # climb begins. Simple additive damping can't fix this
                # because it always concedes turn_pull PRIORITY -- vspeed
                # only decays as a side-effect, never overrides. This gate
                # gives vertical_speed direct priority instead: once climbing
                # too fast, turn_pull itself is throttled down (not just
                # opposed by an additive term), and only restored once
                # vspeed is back under control. A delta limit on the scale
                # itself avoids reintroducing scale-chattering across
                # bank-direction flips.
                target_scale = 1.0 - float(np.clip(
                    (vertical_speed - cfg.vspeed_soft_mps)
                    / max(1.0, cfg.vspeed_hard_mps - cfg.vspeed_soft_mps),
                    0.0, 1.0,
                ))
                self._turn_pull_scale += float(np.clip(
                    target_scale - self._turn_pull_scale,
                    -cfg.turn_pull_scale_delta_limit, cfg.turn_pull_scale_delta_limit,
                ))
                turn_pull *= self._turn_pull_scale

            pitch_raw = float(np.clip(
                turn_pull + altitude_correction + vertical_damping, -cfg.max_pull, cfg.max_pull
            ))
            self._prev_own_alt = own_alt_now
        else:
            # Pull only when the lift vector is reasonably aligned with the
            # turn plane. This avoids the v12-v14 failure: hard pull while
            # still banking, which converted the intercept into a zoom climb.
            if cfg.disable_alignment_gate:  # P1 ablation
                alignment = 1.0
            else:
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

            # When nearly wings-level, body elevation is trustworthy and
            # permits a direct push for a target below instead of always
            # pulling upward.
            if abs(current_bank) < 20.0 and abs(el) > 4.0:
                pitch_raw = float(np.clip(-el / 35.0, -cfg.max_pull, cfg.max_pull))

            if not cfg.disable_pitch_envelope:  # P2 ablation skips this whole block
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

            # P4 ablation (2026-08-21, 30s diagnostic on P0 pinpointed the
            # actual failure mode): own_pitch settles at a stable ~27-29deg
            # (held there by the climb envelope above) and NEVER comes back
            # down for the entire 30s window, because pitch_raw's floor is
            # -min_pull (always some nose-up) for every ata>fine_ata_deg,
            # and the only branch that can push down (wings-level el-trust)
            # needs |current_bank|<20deg, which never happens mid-turn (bank
            # stays 60-73deg) -- so ATA genuinely converges (89->46deg over
            # 12s) while distance keeps growing the entire time
            # (761m->2467m), because a sustained shallow climb is bleeding
            # forward progress into altitude instead of closure. This gate
            # uses the WORLD-FRAME (not body-frame, so it's valid at any
            # bank) altitude relationship: if the target isn't actually
            # above us, there's no justification for continued nose-up pull
            # regardless of what ata says. SUPERSEDED BY P5 (see above) --
            # kept only for A/B reference, P5 is the recommended candidate.
            if cfg.world_frame_pitch_gate:
                target_alt = float(target[StateIndex.ALT])
                own_alt_now = float(own[StateIndex.ALT])
                if target_alt - own_alt_now <= cfg.world_frame_pitch_gate_margin_m:
                    pitch_raw = max(pitch_raw, 0.0)

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
        if not cfg.disable_pitch_envelope:  # P2 ablation skips this too
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
        info = {
            "sim_time": now,
            "state": state,
            "ata": ata,
            "aa": aa,
            "distance": distance,
            "closure_rate": closure,
            "los_az": az,
            "los_el": el,
            "target_bank": target_bank,
            "current_bank": current_bank,
            "bank_error": bank_error,
            "alignment": alignment,
            "pitch_raw": pitch_raw,
            "turn_pull_scale": self._turn_pull_scale,
            "own_speed": float(own[StateIndex.KCAS]),
            "own_alt": float(own[StateIndex.ALT]),
            "own_pitch": own_pitch,
            "roll_cmd": float(self._roll_cmd),
            "pitch_cmd": float(self._pitch_cmd),
        }
        self.info_log.append(info)
        return ActionResult(
            action=action,
            source=f"pursuit_controller[{state}]",
            confidence=learned.confidence,
            info=info,
        )
