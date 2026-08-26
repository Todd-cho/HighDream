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
    controller_name: str = "w1"
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
    # W3-only rear-hemisphere commitment gate. LOS azimuth wraps from
    # +180deg to -180deg although the physical direction barely changes.
    # W1/W2 interpreted that representation wrap as a real side change
    # (run0051: +179.8deg/+45deg -> -173.6deg/-45deg). Values below 180
    # enable the gate; 180 preserves the recorded W1/W2 behavior.
    rear_commit_az_deg: float = 180.0
    # Optional W6 release rule for the rear commitment.  None preserves the
    # permanent W3/W4/W5 commitment exactly.  When enabled, keep the current
    # turn while ATA is making progress; if the best ATA has not improved by
    # rear_progress_deg for rear_stall_flip_s, allow one recommit toward the
    # LOS-indicated side.  This prevents an endless same-direction orbit.
    rear_stall_flip_s: float | None = None
    rear_progress_deg: float = 1.0
    # W7 trend-based recommit.  Unlike W6, this does not require LOS azimuth
    # to cross sign: when the target is in the rear hemisphere, range is
    # closing, and ATA has worsened across this window, reverse the committed
    # turn once and restart the window.  None preserves W1-W6 behavior.
    rear_trend_flip_s: float | None = None
    rear_trend_worsen_deg: float = 3.0
    rear_trend_min_hold_s: float = 0.0

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

    # W2-only coordinated-turn feed-forward.  The live run0050 test proved
    # that the roll cascade can hold 45deg bank accurately, but with the
    # unmodified flight-path loop the aircraft only changed course by about
    # 14deg in 10s while the target LOS crossed roughly 90deg.  A banked
    # aircraft needs back-pressure even when desired_gamma is zero; without
    # it the bank is merely held and does not generate useful turn rate.
    #
    # Keep the W1 baseline unchanged at 0.0.  --mode w2 selects -0.12 at
    # 45deg bank, ramped quadratically from turn_pitch_bank_start_deg.  The
    # existing gamma loop remains the final feedback path: if this feed-
    # forward starts a climb, positive gamma error reduces/cancels the pull.
    turn_pitch_feedforward_at_45_deg: float = 0.0
    turn_pitch_bank_start_deg: float = 10.0
    turn_pitch_reference_bank_deg: float = 45.0

    # Altitude safety backstop (unchanged spirit -- flight_path_angle based,
    # not Euler pitch, so a banked genuine dive is still caught correctly).
    dive_flight_path_floor_deg: float = -20.0
    altitude_floor_m: float = 900.0
    altitude_recovery_m: float = 1400.0

    throttle_turn: float = 0.85
    throttle_track: float = 0.6
    # Optional rear-hemisphere cornering throttle. None preserves W1-W8.
    # W9 uses this to shed excess speed/radius while the target remains aft.
    throttle_rear: float | None = None
    # W10 relative-energy throttle schedule. None preserves W1-W9.  Reduction
    # is continuous in both misalignment and own-minus-target speed, so a
    # faster target immediately restores full turn throttle.
    dynamic_throttle_min: float | None = None
    dynamic_speed_excess_start_mps: float = 10.0
    dynamic_speed_excess_full_mps: float = 70.0
    dynamic_throttle_slew_per_s: float = 0.30
    # Optional absolute-speed pressure for W11.  This prevents both aircraft
    # accelerating together from defeating the relative-speed controller.
    dynamic_absolute_speed_start_mps: float | None = None
    dynamic_absolute_speed_full_mps: float = 360.0
    # Defensive override: do not stay slow while a faster target is rapidly
    # closing inside this range with large ATA.
    threat_override_range_m: float = 1800.0
    threat_override_closure_mps: float = 20.0
    threat_override_target_advantage_mps: float = 15.0
    # W12 dynamic intercept aim point.  Zero preserves W1-W11.  The horizon
    # is distance / own_speed, bounded below/above, using finite-differenced
    # target position rather than an opponent-specific fixed time.
    intercept_horizon_max_s: float = 0.0
    intercept_horizon_min_s: float = 0.5
    intercept_velocity_clip_mps: float = 400.0
    intercept_predict_vertical: bool = True

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
        self._rear_best_ata: float | None = None
        self._rear_last_progress_time: float | None = None
        self._rear_trend_samples: list[tuple[float, float, float]] = []
        self._rear_last_flip_time: float | None = None
        self._throttle_smoothed = self.cfg.throttle_turn
        self._prev_target_position: np.ndarray | None = None
        self._prev_target_time: float | None = None
        self._roll_rate_smoothed = 0.0
        self._pitch_rate_smoothed = 0.0
        self._prev_time: float | None = None
        self._prev_bank: float | None = None
        self._prev_pitch: float | None = None
        self._prev_alt: float | None = None
        self._prev_distance: float | None = None
        self._start_time: float | None = None
        self.state_log: list[str] = []
        self.info_log: list[dict] = []

    def reset(self, context: ActionContext | None = None) -> None:
        self._turn_sign = 0
        self._rear_best_ata = None
        self._rear_last_progress_time = None
        self._rear_trend_samples = []
        self._rear_last_flip_time = None
        self._throttle_smoothed = self.cfg.throttle_turn
        self._prev_target_position = None
        self._prev_target_time = None
        self._roll_rate_smoothed = 0.0
        self._pitch_rate_smoothed = 0.0
        self._prev_time = None
        self._prev_bank = None
        self._prev_pitch = None
        self._prev_alt = None
        self._prev_distance = None
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
        closure_rate = 0.0
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
            guidance_target = target
            current_target_position = np.asarray(
                target[StateIndex.N : StateIndex.D + 1], dtype=np.float64
            )
            actual_distance = float(self.geometry._get_distance(own, target))
            if (
                cfg.intercept_horizon_max_s > 0.0
                and self._prev_target_position is not None
                and self._prev_target_time is not None
            ):
                target_dt = now - self._prev_target_time
                if target_dt > 1e-3:
                    target_velocity = (
                        current_target_position - self._prev_target_position
                    ) / target_dt
                    target_velocity_norm = float(np.linalg.norm(target_velocity))
                    if target_velocity_norm > cfg.intercept_velocity_clip_mps:
                        target_velocity *= cfg.intercept_velocity_clip_mps / target_velocity_norm
                    if not cfg.intercept_predict_vertical:
                        # W13: horizontal intercept only. Instantaneous live
                        # vertical velocity is too noisy/aggressive to
                        # extrapolate for several seconds (W12 commanded a
                        # roughly -69deg gamma and lost over 3km altitude).
                        target_velocity[2] = 0.0
                    horizon = float(np.clip(
                        actual_distance / max(speed, 1.0),
                        cfg.intercept_horizon_min_s,
                        cfg.intercept_horizon_max_s,
                    ))
                    guidance_target = np.array(target, copy=True)
                    guidance_target[StateIndex.N : StateIndex.D + 1] = (
                        current_target_position + target_velocity * horizon
                    )
                    # Live Unreal position.z and StateIndex.ALT are both
                    # up-positive altitude; keep them consistent.
                    if cfg.intercept_predict_vertical:
                        guidance_target[StateIndex.ALT] = guidance_target[StateIndex.D]
                    else:
                        guidance_target[StateIndex.D] = target[StateIndex.D]
                        guidance_target[StateIndex.ALT] = target[StateIndex.ALT]

            az, el = self.geometry._get_los_angle(own, guidance_target)
            az, el = float(az), float(el)
            ata = abs(float(self.geometry._get_antenna_train_angle(own, guidance_target, False)))
            distance = actual_distance
            if dt and self._prev_distance is not None:
                closure_rate = (self._prev_distance - distance) / dt

            desired_sign = 1 if az >= 0.0 else -1
            if self._turn_sign == 0:
                self._turn_sign = desired_sign
            elif abs(az) >= cfg.rear_commit_az_deg:
                # Keep turning through the ambiguous +180/-180 wrap.  W6 can
                # release this commitment only after ATA has stopped making
                # measurable progress for a configured interval.
                if cfg.rear_trend_flip_s is not None:
                    hold_active = (
                        self._rear_last_flip_time is not None
                        and now - self._rear_last_flip_time < cfg.rear_trend_min_hold_s
                    )
                    if hold_active:
                        self._rear_trend_samples = []
                    else:
                        self._rear_trend_samples.append((now, ata, distance))
                        cutoff = now - cfg.rear_trend_flip_s
                        while len(self._rear_trend_samples) > 1 and self._rear_trend_samples[1][0] <= cutoff:
                            self._rear_trend_samples.pop(0)
                        oldest_t, oldest_ata, oldest_distance = self._rear_trend_samples[0]
                        window_ready = now - oldest_t >= cfg.rear_trend_flip_s * 0.9
                        ata_worsening = ata >= oldest_ata + cfg.rear_trend_worsen_deg
                        range_closing = distance < oldest_distance
                        if window_ready and ata_worsening and range_closing:
                            self._turn_sign *= -1
                            self._rear_last_flip_time = now
                            self._rear_trend_samples = []
                    desired_sign = self._turn_sign
                elif desired_sign == self._turn_sign:
                    self._rear_best_ata = None
                    self._rear_last_progress_time = None
                elif cfg.rear_stall_flip_s is None:
                    desired_sign = self._turn_sign
                else:
                    progress = max(0.0, cfg.rear_progress_deg)
                    if self._rear_best_ata is None or ata <= self._rear_best_ata - progress:
                        self._rear_best_ata = ata
                        self._rear_last_progress_time = now
                    if self._rear_last_progress_time is None:
                        self._rear_last_progress_time = now
                    stalled_for = now - self._rear_last_progress_time
                    if stalled_for >= cfg.rear_stall_flip_s:
                        self._turn_sign = desired_sign
                        self._rear_best_ata = None
                        self._rear_last_progress_time = None
                    else:
                        desired_sign = self._turn_sign
            elif desired_sign != self._turn_sign and abs(az) >= cfg.turn_sign_flip_deg:
                self._turn_sign = desired_sign
                self._rear_best_ata = None
                self._rear_last_progress_time = None
                self._rear_trend_samples = []
            elif abs(az) <= cfg.turn_sign_deadband_deg:
                desired_sign = self._turn_sign
            else:
                self._rear_best_ata = None
                self._rear_last_progress_time = None
                self._rear_trend_samples = []

            az_fraction = float(np.clip(abs(az) / cfg.full_bank_az_deg, 0.0, 1.0))
            target_bank_mag = cfg.min_turn_bank_deg + az_fraction * (cfg.max_bank_deg - cfg.min_turn_bank_deg)
            if ata <= cfg.fine_ata_deg:
                target_bank_mag *= float(np.clip(ata / cfg.fine_ata_deg, 0.0, 1.0))
            target_bank = self._turn_sign * target_bank_mag

            # World-frame flight-path-angle target (not body-frame el) --
            # point the velocity vector at the target's altitude over the
            # horizontal range, per 전술.txt.
            own_n, own_e = float(own[StateIndex.N]), float(own[StateIndex.E])
            tgt_n, tgt_e = float(guidance_target[StateIndex.N]), float(guidance_target[StateIndex.E])
            horizontal_distance = max(1.0, float(np.hypot(tgt_n - own_n, tgt_e - own_e)))
            target_alt = float(guidance_target[StateIndex.ALT])
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
        turn_pitch_feedforward = 0.0
        if not cfg.bank_step_test and cfg.turn_pitch_feedforward_at_45_deg != 0.0:
            bank_excess = max(0.0, abs(current_bank) - cfg.turn_pitch_bank_start_deg)
            bank_span = max(1.0, cfg.turn_pitch_reference_bank_deg - cfg.turn_pitch_bank_start_deg)
            bank_fraction = float(np.clip(bank_excess / bank_span, 0.0, 1.0))
            turn_pitch_feedforward = cfg.turn_pitch_feedforward_at_45_deg * bank_fraction ** 2

        pitch_cmd = float(np.clip(
            cfg.pitch_attitude_trim
            + turn_pitch_feedforward
            - desired_pitch_rate / cfg.pitch_rate_gain_degps_per_unit,
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
            if cfg.dynamic_throttle_min is not None and target is not None:
                target_speed = float(target[StateIndex.KCAS])
                speed_excess = speed - target_speed
                speed_span = max(
                    1.0,
                    cfg.dynamic_speed_excess_full_mps - cfg.dynamic_speed_excess_start_mps,
                )
                speed_pressure = float(np.clip(
                    (speed_excess - cfg.dynamic_speed_excess_start_mps) / speed_span,
                    0.0,
                    1.0,
                ))
                absolute_pressure = 0.0
                if cfg.dynamic_absolute_speed_start_mps is not None:
                    absolute_span = max(
                        1.0,
                        cfg.dynamic_absolute_speed_full_mps - cfg.dynamic_absolute_speed_start_mps,
                    )
                    absolute_pressure = float(np.clip(
                        (speed - cfg.dynamic_absolute_speed_start_mps) / absolute_span,
                        0.0,
                        1.0,
                    ))
                # No hard rear-angle switch: ramp from zero reduction at
                # ATA=60deg to full authority at ATA=120deg.
                misalignment = float(np.clip((ata - 60.0) / 60.0, 0.0, 1.0))
                energy_pressure = max(speed_pressure, absolute_pressure)
                target_throttle = cfg.throttle_turn - (
                    cfg.throttle_turn - cfg.dynamic_throttle_min
                ) * energy_pressure * misalignment
                target_faster = target_speed - speed >= cfg.threat_override_target_advantage_mps
                immediate_threat = (
                    ata >= 90.0
                    and distance <= cfg.threat_override_range_m
                    and closure_rate >= cfg.threat_override_closure_mps
                    and target_faster
                )
                if immediate_threat:
                    target_throttle = cfg.throttle_turn
                if dt is None:
                    self._throttle_smoothed = target_throttle
                else:
                    max_step = cfg.dynamic_throttle_slew_per_s * dt
                    self._throttle_smoothed += float(np.clip(
                        target_throttle - self._throttle_smoothed,
                        -max_step,
                        max_step,
                    ))
                throttle = float(np.clip(
                    self._throttle_smoothed,
                    cfg.dynamic_throttle_min,
                    cfg.throttle_turn,
                ))
            elif cfg.throttle_rear is not None and ata >= cfg.rear_commit_az_deg:
                throttle = cfg.throttle_rear
            else:
                throttle = cfg.throttle_turn

        action = np.clip(
            np.array([roll_cmd, pitch_cmd, 0.0, throttle], dtype=np.float32),
            [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0],
        ).astype(np.float32)

        self._prev_time = now
        self._prev_bank = current_bank
        self._prev_pitch = current_pitch
        self._prev_alt = altitude
        self._prev_distance = distance if target is not None else None
        if target is not None:
            self._prev_target_position = np.asarray(
                target[StateIndex.N : StateIndex.D + 1], dtype=np.float64
            ).copy()
            self._prev_target_time = now
        else:
            self._prev_target_position = None
            self._prev_target_time = None

        self.state_log.append(state)
        info = {
            "state": state, "ata": ata, "distance": distance, "closure_rate": closure_rate,
            "los_az": az, "los_el": el,
            "target_bank": target_bank, "current_bank": current_bank, "bank_error": bank_error,
            "desired_roll_rate": desired_roll_rate, "measured_roll_rate": self._roll_rate_smoothed,
            "desired_gamma": desired_gamma, "flight_path_angle": flight_path_angle, "gamma_error": gamma_error,
            "desired_pitch_rate": desired_pitch_rate, "measured_pitch_rate": self._pitch_rate_smoothed,
            "turn_pitch_feedforward": turn_pitch_feedforward,
            "own_pitch": current_pitch, "own_alt": altitude, "vertical_speed": vertical_speed,
            "roll_cmd": roll_cmd, "pitch_cmd": pitch_cmd,
        }
        self.info_log.append(info)
        return ActionResult(action=action, source=f"{cfg.controller_name}[{state}]", confidence=1.0, info=info)
