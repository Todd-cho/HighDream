"""W1: from-scratch minimal approach controller sized from MEASURED live
plant response (2026-08-21, step 4 of the post-pursuit_controller pivot).

Every gain in tactical_wrapper.py (W1-W14) and pursuit_controller.py (P0-P7)
was tuned against JSBSim behavior or against symptoms re-read from live logs
after the fact -- never against a direct, controlled measurement of the real
live plant's actual response. run0044/run0045 (open-loop fixed-command pulse
tests, src/dogfight/ai/pulse_test_provider.py) measured it directly, and a
second pass over those same two logs (user, 2026-08-21) found three things
the first pass missed:

  ROLL:  ~160deg/s per unit roll_cmd -- roughly 7-10x more responsive than
         pitch. roll_cmd=0.0 does NOT hold wings-level: it settles at a
         stable +45.7deg bank (23s window, stdev 1.85deg) -- a genuine trim
         equilibrium (engine torque/gyroscopic, reproduced independently in
         both logs), which a P-only law can never fully zero (it's an
         attractor, not a one-off offset -- needs an I term). The plant also
         has strong inertia: an ended pulse keeps rolling for a while, so a
         command-magnitude delta-limiter alone (bounding how fast roll_cmd
         itself changes) is not the same as real rate feedback -- a D term
         on the MEASURED roll rate is needed too, or corrections overshoot.
  PITCH: ~+24deg/s per unit nose-down (pitch_cmd>0) vs ~+14deg/s per unit
         nose-up (pitch_cmd<0) -- asymmetric. Also strong inertia (ending a
         dive pulse at t=10 left pitch still recovering from -69deg to
         -32deg five seconds later on its own). And: pitch_cmd=0.0 with
         throttle=0.7 is NOT level flight either -- both pulse-test logs
         show pitch drifting 0deg->-6.9deg and speed climbing 200->225m/s
         over the first quiet 5s before any pulse fired. This is smaller
         than roll's bias but real, and needs an explicit nose-up trim
         term, not just a P term driven off LOS elevation.

W1 reuses the same overall geometry (bank toward LOS azimuth via
GeometryInfo, target-bank-from-az sizing, turn-direction commit with P7's
tighter flip threshold) -- that part was never the diagnosed problem. What
changes is the control LAW: roll is now PID (P+I+D) instead of P(+I), and
pitch is trim+P+D using the measured asymmetric gain, with a
flight-path-angle (velocity-vector angle, not Euler pitch) based altitude
safety check -- Euler pitch alone doesn't reflect actual climb/dive at high
bank (see tactical_wrapper.py's fixed _climb_floor() abs() bug, found from
this same pulse data, for a concrete example of that mistake).
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
class W1Config:
    # Geometry / hand-off (unchanged from pursuit_controller.py -- never the
    # diagnosed problem).
    fine_ata_deg: float = 8.0
    track_ata_deg: float = 3.0
    weapon_range_m: float = 1219.2

    # Bank-to-turn outer loop.
    max_bank_deg: float = 60.0
    min_turn_bank_deg: float = 20.0
    full_bank_az_deg: float = 60.0
    turn_sign_deadband_deg: float = 4.0
    turn_sign_flip_deg: float = 10.0  # P7 finding: tighter flip = faster re-commit

    # Roll PID, sized from the measured ~160deg/s-per-unit rate and the
    # measured +45.7deg neutral-command equilibrium.
    # - roll_kp: bank_error_norm_deg=60 means a 60deg error commands full
    #   authority (~160deg/s) -- errors close fast at that rate, so this is
    #   intentionally not aggressive.
    # - roll_ki: integrates over several seconds specifically to cancel the
    #   trim equilibrium (a pure P term cannot, since it's a stable
    #   attractor, not a one-off offset); clamped so it can't run away.
    # - roll_kd: NEW (2026-08-21 second pass) -- damps against the
    #   plant's own measured roll rate (not just a delta-limit on our own
    #   command), since the pulse logs showed strong post-input inertia
    #   that a command-only rate limiter doesn't see or react to.
    bank_error_norm_deg: float = 60.0
    roll_kp: float = 1.0
    roll_ki: float = 0.012
    roll_integral_clamp: float = 0.6
    roll_kd: float = 0.004  # per deg/s of measured roll rate
    roll_delta_limit: float = 0.15

    # Pitch: trim + asymmetric P (LOS-elevation-driven aim, only trusted
    # near wings-level same as pursuit_controller.py) + D on measured pitch
    # rate. level_pitch_trim=-0.145 is MEASURED (2026-08-21, run0046 pitch
    # trim sweep, 3s legs at pitch_cmd in [0,-0.05,-0.10,-0.15,-0.20,0]):
    # vertical_speed crossed zero between the -0.10 leg (-18.5m/s) and the
    # -0.15 leg (+2.25m/s); linear interpolation puts the true zero-crossing
    # at -0.145. Superseded the earlier -0.10 estimate (drift-rate/gain
    # division, never actually measured). Caveat: own_speed climbed
    # 207->269m/s across the sweep (not fully speed-stabilized before
    # starting), so this is a good working value, not a final one.
    level_pitch_trim: float = -0.145
    pitch_norm_down_deg: float = 45.0
    pitch_norm_up_deg: float = 26.25  # 45 * 14/24, so equal error -> equal RATE either direction
    pitch_kd: float = 0.006  # per deg/s of measured pitch rate
    pitch_delta_limit: float = 0.15
    max_pitch_cmd: float = 0.7

    # Altitude safety backstop -- keyed on flight_path_angle (velocity
    # vector angle, asin(vertical_speed/speed)), not Euler own_pitch, so a
    # banked turn that's still genuinely diving is caught correctly (the
    # abs(own_pitch_deg) bug in tactical_wrapper.py's _climb_floor(),
    # found from this same pulse data, is exactly the mistake this avoids).
    dive_flight_path_floor_deg: float = -20.0
    altitude_floor_m: float = 900.0
    altitude_recovery_m: float = 1400.0

    throttle_turn: float = 0.85
    throttle_track: float = 0.6


class W1ControllerActionProvider(ActionProvider):
    """Minimal bank-to-turn approach controller with gains measured from a
    live open-loop pulse test, not guessed from JSBSim. RL is NOT blended in
    at all -- W1 is meant to stand alone for the approach-only live test
    (step 5 of the pivot plan); an inner RL/track policy is a later step."""

    def __init__(self, config: W1Config | None = None):
        self.cfg = config or W1Config()
        self.geometry = GeometryInfo()
        self._turn_sign = 0
        self._roll_integral = 0.0
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self._prev_time: float | None = None
        self._prev_bank: float | None = None
        self._prev_pitch: float | None = None
        self._prev_alt: float | None = None
        self.state_log: list[str] = []
        self.info_log: list[dict] = []

    def reset(self, context: ActionContext | None = None) -> None:
        self._turn_sign = 0
        self._roll_integral = 0.0
        self._roll_cmd = 0.0
        self._pitch_cmd = 0.0
        self._prev_time = None
        self._prev_bank = None
        self._prev_pitch = None
        self._prev_alt = None
        self.state_log = []
        self.info_log = []

    def compute_action(self, context: ActionContext) -> ActionResult:
        cfg = self.cfg
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            action = np.array([0.0, 0.0, 0.0, cfg.throttle_turn], dtype=np.float32)
            return ActionResult(action, "w1_passthrough", 1.0)

        now = float(own[StateIndex.SIM_TIME])
        az, el = self.geometry._get_los_angle(own, target)
        az, el = float(az), float(el)
        ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
        distance = float(self.geometry._get_distance(own, target))
        current_bank = _wrap180(float(own[StateIndex.ROLL]))
        current_pitch = float(own[StateIndex.PITCH])
        altitude = float(own[StateIndex.ALT])
        speed = float(own[StateIndex.KCAS])

        dt = None
        if self._prev_time is not None:
            candidate = now - self._prev_time
            if candidate > 1e-3:
                dt = candidate
        roll_rate = _wrap180(current_bank - self._prev_bank) / dt if dt and self._prev_bank is not None else 0.0
        pitch_rate = (current_pitch - self._prev_pitch) / dt if dt and self._prev_pitch is not None else 0.0
        vertical_speed = (altitude - self._prev_alt) / dt if dt and self._prev_alt is not None else 0.0
        flight_path_angle = float(np.degrees(np.arcsin(
            np.clip(vertical_speed / speed, -1.0, 1.0) if speed > 1e-3 else 0.0
        )))

        # Bank-direction commit (unchanged shape; only the flip threshold
        # changed, per P7's finding).
        desired_sign = 1 if az >= 0.0 else -1
        if self._turn_sign == 0:
            self._turn_sign = desired_sign
        elif desired_sign != self._turn_sign and abs(az) >= cfg.turn_sign_flip_deg:
            self._turn_sign = desired_sign
        elif abs(az) <= cfg.turn_sign_deadband_deg:
            desired_sign = self._turn_sign

        az_fraction = float(np.clip(abs(az) / cfg.full_bank_az_deg, 0.0, 1.0))
        target_bank_mag = cfg.min_turn_bank_deg + az_fraction * (cfg.max_bank_deg - cfg.min_turn_bank_deg)
        if ata <= cfg.fine_ata_deg:
            target_bank_mag *= float(np.clip(ata / cfg.fine_ata_deg, 0.0, 1.0))
        target_bank = self._turn_sign * target_bank_mag

        # Roll PID.
        bank_error = _wrap180(target_bank - current_bank)
        if dt:
            self._roll_integral = float(np.clip(
                self._roll_integral + bank_error * dt,
                -cfg.roll_integral_clamp / max(cfg.roll_ki, 1e-6),
                cfg.roll_integral_clamp / max(cfg.roll_ki, 1e-6),
            ))
        roll_raw = float(np.clip(
            cfg.roll_kp * bank_error / cfg.bank_error_norm_deg
            + cfg.roll_ki * self._roll_integral
            - cfg.roll_kd * roll_rate,
            -1.0, 1.0,
        ))
        self._roll_cmd += float(np.clip(roll_raw - self._roll_cmd, -cfg.roll_delta_limit, cfg.roll_delta_limit))

        # Pitch: trim + asymmetric-P aim (LOS elevation, trustworthy only
        # near wings-level) + D on measured pitch rate.
        pitch_error = el  # positive el = target above = need nose up = negative pitch_cmd
        norm = cfg.pitch_norm_down_deg if pitch_error < 0 else cfg.pitch_norm_up_deg
        pitch_raw = cfg.level_pitch_trim + float(np.clip(-pitch_error / norm, -cfg.max_pitch_cmd, cfg.max_pitch_cmd))
        pitch_raw -= cfg.pitch_kd * pitch_rate
        pitch_raw = float(np.clip(pitch_raw, -1.0, 1.0))
        self._pitch_cmd += float(np.clip(pitch_raw - self._pitch_cmd, -cfg.pitch_delta_limit, cfg.pitch_delta_limit))

        # Altitude safety backstop -- flight_path_angle based (velocity
        # vector), not Euler own_pitch, so a banked genuine dive is caught
        # even if body-frame pitch attitude looks moderate.
        diving = flight_path_angle <= cfg.dive_flight_path_floor_deg
        if altitude <= cfg.altitude_floor_m or (diving and altitude < cfg.altitude_recovery_m):
            self._pitch_cmd = min(self._pitch_cmd, -0.6)
            self._roll_cmd = float(np.clip(self._roll_cmd, -0.3, 0.3))
        elif altitude < cfg.altitude_recovery_m:
            recovery = (cfg.altitude_recovery_m - altitude) / (cfg.altitude_recovery_m - cfg.altitude_floor_m)
            self._pitch_cmd = min(self._pitch_cmd, -0.6 * recovery)

        if ata <= cfg.track_ata_deg and distance <= cfg.weapon_range_m:
            state = "weapons_track"
            throttle = cfg.throttle_track
        elif ata <= cfg.fine_ata_deg:
            state = "fine_pursuit"
            throttle = cfg.throttle_track
        else:
            state = "turn_pursuit"
            throttle = cfg.throttle_turn

        action = np.clip(
            np.array([self._roll_cmd, self._pitch_cmd, 0.0, throttle], dtype=np.float32),
            [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0],
        ).astype(np.float32)

        self._prev_time = now
        self._prev_bank = current_bank
        self._prev_pitch = current_pitch
        self._prev_alt = altitude

        self.state_log.append(state)
        info = {
            "state": state, "ata": ata, "distance": distance,
            "los_az": az, "los_el": el,
            "target_bank": target_bank, "current_bank": current_bank, "bank_error": bank_error,
            "roll_rate": roll_rate, "pitch_rate": pitch_rate,
            "roll_integral": self._roll_integral,
            "own_pitch": current_pitch, "own_alt": altitude,
            "vertical_speed": vertical_speed, "flight_path_angle": flight_path_angle,
            "roll_cmd": float(self._roll_cmd), "pitch_cmd": float(self._pitch_cmd),
        }
        self.info_log.append(info)
        return ActionResult(action=action, source=f"w1[{state}]", confidence=1.0, info=info)
