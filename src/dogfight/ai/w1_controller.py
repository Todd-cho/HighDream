"""W1: from-scratch minimal approach controller, gains measured from live
pulse tests (2026-08-21, step 4 of the post-pursuit_controller pivot).

Second revision, per a more careful re-analysis of the same two pulse logs
(run0046 pitch-trim sweep, run0047 roll-inertia step/brake -- see
Desktop/전술.txt) that corrected the first pass on two points:

  1. PITCH TRIM is about ATTITUDE, not altitude. The first pass found the
     pitch_cmd where vertical_speed crossed zero (-0.145) -- but that
     conflates attitude trim with leftover descent momentum from earlier in
     the same test. The correct read is the pitch_cmd where PITCH RATE (not
     vertical speed) goes to ~0: -0.05. Altitude/vertical-speed control is a
     SEPARATE outer loop (flight-path-angle tracking), not something a
     single trim constant should try to absorb.
  2. ROLL HAS NO FIXED BIAS. The first pass's "+45.7deg equilibrium" from
     run0044 and the "-84..-90deg settling" from run0047 are NOT the same
     underlying constant -- a fresh start (no prior roll command) shows
     essentially zero drift (0deg -> -0.01deg over 2s). What looked like a
     bias was really unbraked angular momentum coasting to a stop wherever
     the plant's own damping happened to arrest it, biased toward whichever
     direction was most recently commanded. A PID with an integral term
     (the first revision's fix) is therefore solving the wrong problem --
     what's actually needed is a proper RATE-COMMAND cascade: convert
     attitude error to a desired rate, then command based on
     (desired_rate - measured_rate), so the loop actively brakes to a stop
     at the target instead of assuming a constant offset needs to be
     integrated away.

Architecture (both axes): outer loop (attitude error -> desired rate) then
inner loop (desired_rate - measured_rate -> surface command), with EMA
smoothing on the measured rate (it's finite-differenced from live state and
noisy). This is the standard fly-by-wire rate-command/attitude-hold
structure, not a PID directly on attitude error.

  ROLL:  desired_roll_rate = clip(bank_kp * bank_error, -rate_limit, +rate_limit)
         roll_cmd = clip((desired_roll_rate - measured_roll_rate) / rate_gain, -cmd_limit, +cmd_limit)
  PITCH: desired_gamma = atan2(target_alt - own_alt, horizontal_distance)  # world-frame, not body-frame el
         gamma_error = desired_gamma - current_flight_path_angle
         desired_pitch_rate = clip(gamma_kp * gamma_error, -rate_limit, +rate_limit)
         pitch_cmd = attitude_trim - desired_pitch_rate / rate_gain

Before pointing this at a live pursuit (91deg), validate the roll cascade in
isolation with bank_step_test=True (a scripted +30->0->-30->0deg target-bank
sequence, no target/geometry involved at all) -- confirm it actually settles
at each target without overshoot before trusting it to fly toward a moving
target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


# (duration_s, target_bank_deg) -- the pre-pursuit isolated validation
# sequence from 전술.txt's closing recommendation.
BANK_STEP_TEST_SEQUENCE: list[tuple[float, float]] = [
    (5.0, 30.0),
    (5.0, 0.0),
    (5.0, -30.0),
    (5.0, 0.0),
]


@dataclass
class W1Config:
    # Geometry / hand-off (unchanged from pursuit_controller.py).
    fine_ata_deg: float = 8.0
    track_ata_deg: float = 3.0
    weapon_range_m: float = 1219.2

    # Bank-to-turn outer loop (az -> target bank), used only when
    # bank_step_test is False.
    max_bank_deg: float = 45.0  # 전술.txt's target_bank_limit_deg
    min_turn_bank_deg: float = 15.0
    full_bank_az_deg: float = 60.0
    turn_sign_deadband_deg: float = 4.0
    turn_sign_flip_deg: float = 10.0  # P7 finding

    # Roll rate-cascade, values from 전술.txt's measured re-analysis:
    # rate_gain=140deg/s per unit (re-estimated from the clean short
    # roll_inertia test, both directions averaged), desired-rate capped at
    # 30deg/s, output capped at +-0.25 for this first live validation.
    bank_kp: float = 1.0  # deg of desired-rate per deg of bank error
    roll_rate_gain_degps_per_unit: float = 140.0
    roll_desired_rate_limit_degps: float = 30.0
    roll_command_limit: float = 0.25
    roll_rate_smoothing_alpha: float = 0.25

    # Pitch rate-cascade. attitude_trim=-0.05 measured where PITCH RATE (not
    # vertical speed) crossed zero in the run0046 sweep -- altitude control
    # is the separate gamma (flight-path-angle) loop below, not folded into
    # this constant. rate_gain=25deg/s per unit from the near-normal-range
    # legs of the same sweep.
    gamma_kp: float = 1.0  # deg of desired pitch-rate per deg of gamma error
    pitch_attitude_trim: float = -0.05
    pitch_rate_gain_degps_per_unit: float = 25.0
    pitch_desired_rate_limit_degps: float = 8.0
    pitch_command_limit: float = 0.30
    pitch_rate_smoothing_alpha: float = 0.25

    # Altitude safety backstop (unchanged spirit -- flight_path_angle based,
    # not Euler pitch, so a banked genuine dive is still caught correctly).
    dive_flight_path_floor_deg: float = -20.0
    altitude_floor_m: float = 900.0
    altitude_recovery_m: float = 1400.0

    throttle_turn: float = 0.85
    throttle_track: float = 0.6

    # Pre-pursuit isolated validation (전술.txt's closing recommendation):
    # when True, target_bank comes from BANK_STEP_TEST_SEQUENCE instead of
    # az, target/geometry is ignored entirely, and pitch just holds
    # gamma_error=0 (flat, no target altitude reference) so only the roll
    # cascade is under test.
    bank_step_test: bool = False


class W1ControllerActionProvider(ActionProvider):
    """Rate-command/attitude-hold controller with gains measured from a live
    open-loop pulse test. RL is NOT blended in -- W1 stands alone for the
    approach-only live test; an inner RL/track policy is a later step."""

    def __init__(self, config: W1Config | None = None):
        self.cfg = config or W1Config()
        self.geometry = GeometryInfo()
        self._turn_sign = 0
        self._roll_rate_smoothed = 0.0
        self._pitch_rate_smoothed = 0.0
        self._prev_time: float | None = None
        self._prev_bank: float | None = None
        self._prev_pitch: float | None = None
        self._prev_alt: float | None = None
        self._start_time: float | None = None
        self.state_log: list[str] = []
        self.info_log: list[dict] = []

    def reset(self, context: ActionContext | None = None) -> None:
        self._turn_sign = 0
        self._roll_rate_smoothed = 0.0
        self._pitch_rate_smoothed = 0.0
        self._prev_time = None
        self._prev_bank = None
        self._prev_pitch = None
        self._prev_alt = None
        self._start_time = None
        self.state_log = []
        self.info_log = []

    def compute_action(self, context: ActionContext) -> ActionResult:
        cfg = self.cfg
        own = context.ownship_state
        if own is None:
            action = np.array([0.0, 0.0, 0.0, cfg.throttle_turn], dtype=np.float32)
            return ActionResult(action, "w1_passthrough", 1.0)
        target = context.target_state

        now = float(own[StateIndex.SIM_TIME])
        if self._start_time is None:
            self._start_time = now
        current_bank = _wrap180(float(own[StateIndex.ROLL]))
        current_pitch = float(own[StateIndex.PITCH])
        altitude = float(own[StateIndex.ALT])
        speed = float(own[StateIndex.KCAS])

        dt = None
        if self._prev_time is not None:
            candidate = now - self._prev_time
            if candidate > 1e-3:
                dt = candidate
        raw_roll_rate = _wrap180(current_bank - self._prev_bank) / dt if dt and self._prev_bank is not None else 0.0
        raw_pitch_rate = (current_pitch - self._prev_pitch) / dt if dt and self._prev_pitch is not None else 0.0
        vertical_speed = (altitude - self._prev_alt) / dt if dt and self._prev_alt is not None else 0.0
        a = cfg.roll_rate_smoothing_alpha
        self._roll_rate_smoothed = a * raw_roll_rate + (1.0 - a) * self._roll_rate_smoothed
        b = cfg.pitch_rate_smoothing_alpha
        self._pitch_rate_smoothed = b * raw_pitch_rate + (1.0 - b) * self._pitch_rate_smoothed
        flight_path_angle = float(np.degrees(np.arcsin(
            np.clip(vertical_speed / speed, -1.0, 1.0) if speed > 1e-3 else 0.0
        )))

        ata = 0.0
        distance = 0.0
        az = el = 0.0
        if cfg.bank_step_test:
            elapsed = now - self._start_time
            cum = 0.0
            target_bank = BANK_STEP_TEST_SEQUENCE[-1][1]
            for duration, bank in BANK_STEP_TEST_SEQUENCE:
                cum += duration
                if elapsed < cum:
                    target_bank = bank
                    break
            desired_gamma = 0.0  # hold flat -- isolating the roll loop only
        elif target is not None:
            az, el = self.geometry._get_los_angle(own, target)
            az, el = float(az), float(el)
            ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
            distance = float(self.geometry._get_distance(own, target))

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

            # World-frame flight-path-angle target (not body-frame el) --
            # point the velocity vector at the target's altitude over the
            # horizontal range, per 전술.txt.
            own_n, own_e = float(own[StateIndex.N]), float(own[StateIndex.E])
            tgt_n, tgt_e = float(target[StateIndex.N]), float(target[StateIndex.E])
            horizontal_distance = max(1.0, float(np.hypot(tgt_n - own_n, tgt_e - own_e)))
            target_alt = float(target[StateIndex.ALT])
            desired_gamma = float(np.degrees(np.arctan2(target_alt - altitude, horizontal_distance)))
        else:
            target_bank = 0.0
            desired_gamma = 0.0

        # Roll rate-cascade.
        bank_error = _wrap180(target_bank - current_bank)
        desired_roll_rate = float(np.clip(
            cfg.bank_kp * bank_error, -cfg.roll_desired_rate_limit_degps, cfg.roll_desired_rate_limit_degps
        ))
        roll_cmd = float(np.clip(
            (desired_roll_rate - self._roll_rate_smoothed) / cfg.roll_rate_gain_degps_per_unit,
            -cfg.roll_command_limit, cfg.roll_command_limit,
        ))

        # Pitch rate-cascade.
        gamma_error = desired_gamma - flight_path_angle
        desired_pitch_rate = float(np.clip(
            cfg.gamma_kp * gamma_error, -cfg.pitch_desired_rate_limit_degps, cfg.pitch_desired_rate_limit_degps
        ))
        # Sign is deliberately inverted: MORE nose-up needed (positive
        # desired_pitch_rate, since gamma should increase) -> NEGATIVE
        # pitch_cmd (live convention: positive=nose-down).
        pitch_cmd = float(np.clip(
            cfg.pitch_attitude_trim - desired_pitch_rate / cfg.pitch_rate_gain_degps_per_unit,
            -cfg.pitch_command_limit, cfg.pitch_command_limit,
        ))

        # Altitude safety backstop.
        diving = flight_path_angle <= cfg.dive_flight_path_floor_deg
        if altitude <= cfg.altitude_floor_m or (diving and altitude < cfg.altitude_recovery_m):
            pitch_cmd = min(pitch_cmd, -0.6)
            roll_cmd = float(np.clip(roll_cmd, -0.3, 0.3))
        elif altitude < cfg.altitude_recovery_m:
            recovery = (cfg.altitude_recovery_m - altitude) / (cfg.altitude_recovery_m - cfg.altitude_floor_m)
            pitch_cmd = min(pitch_cmd, -0.6 * recovery)

        if cfg.bank_step_test:
            state = "bank_step_test"
            throttle = cfg.throttle_turn
        elif ata <= cfg.track_ata_deg and distance <= cfg.weapon_range_m:
            state = "weapons_track"
            throttle = cfg.throttle_track
        elif ata <= cfg.fine_ata_deg:
            state = "fine_pursuit"
            throttle = cfg.throttle_track
        else:
            state = "turn_pursuit"
            throttle = cfg.throttle_turn

        action = np.clip(
            np.array([roll_cmd, pitch_cmd, 0.0, throttle], dtype=np.float32),
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
            "desired_roll_rate": desired_roll_rate, "measured_roll_rate": self._roll_rate_smoothed,
            "desired_gamma": desired_gamma, "flight_path_angle": flight_path_angle, "gamma_error": gamma_error,
            "desired_pitch_rate": desired_pitch_rate, "measured_pitch_rate": self._pitch_rate_smoothed,
            "own_pitch": current_pitch, "own_alt": altitude, "vertical_speed": vertical_speed,
            "roll_cmd": roll_cmd, "pitch_cmd": pitch_cmd,
        }
        self.info_log.append(info)
        return ActionResult(action=action, source=f"w1[{state}]", confidence=1.0, info=info)
