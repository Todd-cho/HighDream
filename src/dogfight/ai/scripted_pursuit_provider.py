from __future__ import annotations

import math

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


class ScriptedPursuitActionProvider(ActionProvider):
    """Simple, non-learned bank-to-turn pursuit controller for the target
    aircraft. Computes raw roll/pitch/rudder/throttle directly rather than
    using the sim's own step_autopilot(): that built-in control loop was
    found (2026-08-11) to crash the target aircraft almost immediately in
    this env/scenario regardless of the commanded heading/altitude/speed
    (confirmed with both a static heading_cmd and a dynamic pursuit
    heading_cmd -- the instability was in step_autopilot itself, not in the
    commanded values). Behavior tree mode is also unusable: its only action
    node (Task_Empty) never actually maneuvers (confirmed from the AIP_DCS
    project source: only BFM_Mode *checks* are implemented, no BFM *tasks*).

    This provider gives a genuinely active, closing opponent for validating
    the ownship policy against something other than the passive "loiter"
    target every prior evaluation this session used.

    Sign conventions verified empirically 2026-08-11 (held a constant action
    for 30 steps on the ownship, watched StateIndex.PITCH/ROLL): action[1]=+1.0
    pitches the nose DOWN, action[0]=+1.0 rolls right/positive. Assumed
    (not re-verified) to hold identically for the target aircraft, since both
    sides step the same FighterSim.JSBSim.step(action) interface.

    Structure: nested P-controllers, not a single-shot proportional-on-error
    law -- a heading error maps to a *desired bank angle* (clamped), then a
    bank-angle error maps to a roll-axis command; similarly altitude error
    maps to a *desired pitch angle* (clamped, plus a bank-induced-lift-loss
    compensation term), then a pitch-angle error maps to a pitch-axis
    command. This two-stage structure is what keeps it stable -- an earlier
    attempt that fed heading/altitude error straight into step_autopilot()
    (a black-box control loop) diverged into a crash.
    """

    def __init__(
        self,
        cruise_altitude_m: float = 7000.0,
        cruise_speed_kcas: float = 260.0,
        max_bank_deg: float = 70.0,
        heading_to_bank_gain: float = 1.5,
        bank_rate_gain: float = 0.035,
        max_pitch_deg: float = 25.0,
        altitude_to_pitch_gain: float = 0.02,
        bank_pitch_compensation_gain: float = 0.15,
        pitch_rate_gain: float = 0.05,
        speed_gain: float = 0.01,
        base_throttle: float = 0.65,
        confidence: float = 0.9,
    ):
        self.cruise_altitude_m = cruise_altitude_m
        self.cruise_speed_kcas = cruise_speed_kcas
        self.max_bank_deg = max_bank_deg
        self.heading_to_bank_gain = heading_to_bank_gain
        self.bank_rate_gain = bank_rate_gain
        self.max_pitch_deg = max_pitch_deg
        self.altitude_to_pitch_gain = altitude_to_pitch_gain
        self.bank_pitch_compensation_gain = bank_pitch_compensation_gain
        self.pitch_rate_gain = pitch_rate_gain
        self.speed_gain = speed_gain
        self.base_throttle = base_throttle
        self.confidence = confidence

    def reset(self, context: ActionContext | None = None) -> None:
        return None

    @staticmethod
    def _wrap180(deg: float) -> float:
        return (deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def compute_action(self, context: ActionContext) -> ActionResult:
        my = context.ownship_state
        opp = context.target_state

        my_n, my_e = float(my[StateIndex.N]), float(my[StateIndex.E])
        opp_n, opp_e = float(opp[StateIndex.N]), float(opp[StateIndex.E])
        my_alt = float(my[StateIndex.ALT])
        my_roll = float(my[StateIndex.ROLL])
        my_pitch = float(my[StateIndex.PITCH])
        my_yaw = float(my[StateIndex.YAW])
        my_kcas = float(my[StateIndex.KCAS])

        # --- roll axis: heading error -> desired bank -> roll command ---
        bearing_deg = math.degrees(math.atan2(opp_e - my_e, opp_n - my_n))
        heading_error = self._wrap180(bearing_deg - my_yaw)
        desired_bank = self._clip(
            heading_error * self.heading_to_bank_gain, -self.max_bank_deg, self.max_bank_deg
        )
        bank_error = self._wrap180(desired_bank - my_roll)
        roll_action = self._clip(bank_error * self.bank_rate_gain, -1.0, 1.0)

        # --- pitch axis: altitude error (+ bank compensation) -> desired
        # pitch -> pitch command. action[1]=+1 pitches DOWN, so a positive
        # pitch_error (need more nose-up) requires a *negative* action.
        altitude_error = self.cruise_altitude_m - my_alt
        desired_pitch = altitude_error * self.altitude_to_pitch_gain
        desired_pitch += self.bank_pitch_compensation_gain * abs(my_roll)
        desired_pitch = self._clip(desired_pitch, -self.max_pitch_deg, self.max_pitch_deg)
        pitch_error = desired_pitch - my_pitch
        pitch_action = self._clip(-pitch_error * self.pitch_rate_gain, -1.0, 1.0)

        # --- throttle: hold cruise_speed_kcas ---
        speed_error = self.cruise_speed_kcas - my_kcas
        throttle_action = self._clip(self.base_throttle + speed_error * self.speed_gain, 0.0, 1.0)

        action = clip_action([roll_action, pitch_action, 0.0, throttle_action])
        return ActionResult(action=action, source="scripted_pursuit", confidence=self.confidence)


__all__ = ["ScriptedPursuitActionProvider"]
