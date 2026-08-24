"""EP1: output-equivalent W111 controller with explicit phase/owner telemetry."""

from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.control_arbitration import diagnose_control_ownership
from dogfight.ai.phase_manager import classify_phase
from dogfight.sim.state_schema import StateIndex


class ExplicitPhaseBFMController(ActionProvider):
    """Preserve a proven controller while exposing hidden arbitration decisions."""

    def __init__(self, legacy: ActionProvider, name: str = "ep1") -> None:
        self.legacy = legacy
        self.name = name

    def reset(self, context: ActionContext | None = None) -> None:
        self.legacy.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.legacy.compute_action(context)
        info = dict(result.info)
        info["engagement_phase"] = classify_phase(info).value
        info.update(diagnose_control_ownership(info, result.action))
        return ActionResult(
            action=result.action,
            source=f"{self.name}<{result.source}>",
            confidence=result.confidence,
            info=info,
        )

    def close(self) -> None:
        self.legacy.close()


class ExplicitPhaseBFMControllerV2(ExplicitPhaseBFMController):
    """EP2: give post-merge reacquisition one stable bank-axis owner.

    W111 repeatedly reverses a full +/-82 degree bank when instantaneous LOS
    crosses the nose.  EP2 preserves the committed turn and only permits a
    reversal after the measured target turn has stayed opposite long enough.
    All other axes and phases remain byte-identical to W111.
    """

    def __init__(self, legacy: ActionProvider) -> None:
        super().__init__(legacy, name="ep2")
        self._commit_sign = 0
        self._commit_since = 0.0
        self._candidate_sign = 0
        self._candidate_since = 0.0
        self._initial_candidate_sign = 0
        self._initial_candidate_since = 0.0

    def reset(self, context: ActionContext | None = None) -> None:
        super().reset(context)
        self._commit_sign = 0
        self._commit_since = 0.0
        self._candidate_sign = 0
        self._candidate_since = 0.0
        self._initial_candidate_sign = 0
        self._initial_candidate_since = 0.0

    @staticmethod
    def _sign(value: float, deadband: float = 1e-6) -> int:
        return 1 if value > deadband else -1 if value < -deadband else 0

    @staticmethod
    def _wrap180(value: float) -> float:
        return (value + 180.0) % 360.0 - 180.0

    def compute_action(self, context: ActionContext) -> ActionResult:
        observed = super().compute_action(context)
        info = dict(observed.info)
        phase = info["engagement_phase"]
        now = (
            float(context.ownship_state[StateIndex.SIM_TIME])
            if context.ownship_state is not None else 0.0
        )
        target_bank = float(info.get("target_bank", 0.0))
        target_turn_rate = float(info.get("target_yaw_rate", 0.0))

        # Do not latch the bank on the exact merge-crossing frame.  Live logs
        # show that this frame still carries the pre-pass +82deg target and the
        # geometrically correct post-pass target becomes -82deg one tick later.
        # Observe a stable post-merge course-bank sign first; EP2's previous
        # immediate latch blocked this necessary initial reversal.
        if (
            self._commit_sign == 0
            and bool(info.get("first_merge_passed", False))
            and phase == "reacquire"
        ):
            initial_sign = self._sign(target_bank, deadband=30.0)
            if initial_sign != 0:
                if self._initial_candidate_sign != initial_sign:
                    self._initial_candidate_sign = initial_sign
                    self._initial_candidate_since = now
                elif now - self._initial_candidate_since >= 0.25:
                    self._commit_sign = initial_sign
                    self._commit_since = now
            else:
                self._initial_candidate_sign = 0

        reversal_candidate = self._sign(target_turn_rate, deadband=6.0)
        target_bank_sign = self._sign(target_bank, deadband=30.0)
        stable_target_turn = float(info.get("target_turn_stable_s", 0.0)) >= 0.8
        reversal_allowed = (
            self._commit_sign != 0
            and reversal_candidate == -self._commit_sign
            and target_bank_sign == reversal_candidate
            and stable_target_turn
            and now - self._commit_since >= 6.0
        )
        if reversal_allowed:
            if self._candidate_sign != reversal_candidate:
                self._candidate_sign = reversal_candidate
                self._candidate_since = now
            elif now - self._candidate_since >= 1.0:
                self._commit_sign = reversal_candidate
                self._commit_since = now
                self._candidate_sign = 0
        else:
            self._candidate_sign = 0

        action = np.asarray(observed.action, dtype=np.float32).copy()
        bank_override_active = phase == "reacquire" and self._commit_sign != 0
        if bank_override_active:
            desired_bank = 82.0 * self._commit_sign
            current_bank = float(info.get("current_bank", 0.0))
            roll_rate = float(info.get("measured_roll_rate", 0.0))
            bank_error = self._wrap180(desired_bank - current_bank)
            roll_cmd = float(np.clip(0.012 * bank_error - 0.006 * roll_rate, -0.8, 0.8))
            action[0] = roll_cmd
            info.update({
                "bank_owner": "ep2_reacquire_commit",
                "bank_authority": 1.0,
                "target_bank": desired_bank,
                "bank_error": bank_error,
                "proposal_conflict": False,
            })

        info.update({
            "explicit_bank_override_active": bank_override_active,
            "explicit_commit_sign": self._commit_sign,
            "explicit_commit_age_s": max(0.0, now - self._commit_since),
            "explicit_reversal_candidate": self._candidate_sign,
            "explicit_reversal_candidate_age_s": (
                max(0.0, now - self._candidate_since) if self._candidate_sign else 0.0
            ),
            "explicit_initial_candidate": self._initial_candidate_sign,
            "explicit_initial_candidate_age_s": (
                max(0.0, now - self._initial_candidate_since)
                if self._commit_sign == 0 and self._initial_candidate_sign else 0.0
            ),
        })
        return ActionResult(
            action=action,
            source=f"ep2<{observed.source}>",
            confidence=observed.confidence,
            info=info,
        )


class ExplicitPhaseBFMControllerV3(ExplicitPhaseBFMControllerV2):
    """EP3: moderate roll-compatible opening pull on top of corrected EP2.

    W112's fixed -0.82 pulse improved only the first second and then slowed
    roll buildup.  EP3 fills only the weak portion of the quadratic bank gate:
    it ramps from -0.38 to -0.62 while bank grows from 30 to 60 degrees, and
    relinquishes pitch as soon as the legacy scheduler is already stronger.
    """

    def __init__(self, legacy: ActionProvider) -> None:
        super().__init__(legacy)
        self.name = "ep3"
        self._opening_start_time: float | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        super().reset(context)
        self._opening_start_time = None

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = super().compute_action(context)
        info = dict(result.info)
        now = (
            float(context.ownship_state[StateIndex.SIM_TIME])
            if context.ownship_state is not None else 0.0
        )
        if self._opening_start_time is None:
            self._opening_start_time = now
        elapsed = now - self._opening_start_time
        bank = abs(float(info.get("current_bank", 0.0)))
        ata = float(info.get("ata", 0.0))
        own_speed = (
            float(context.ownship_state[StateIndex.KCAS])
            if context.ownship_state is not None else 0.0
        )
        active = (
            info.get("engagement_phase") == "opening_merge"
            and not bool(info.get("first_merge_passed", False))
            and elapsed <= 4.0
            and 30.0 <= bank <= 60.0
            and ata >= 80.0
            and own_speed >= 180.0
        )
        action = np.asarray(result.action, dtype=np.float32).copy()
        scheduled_pitch = 0.0
        if active:
            fraction = float(np.clip((bank - 30.0) / 30.0, 0.0, 1.0))
            scheduled_pitch = -(0.38 + 0.24 * fraction)
            # Negative pitch is the measured pull direction. Only fill missing
            # pull authority; never weaken a stronger legacy command.
            action[1] = min(float(action[1]), scheduled_pitch)
            info.update({
                "pitch_owner": "ep3_opening_pull_fill",
                "pitch_authority": 1.0,
            })
        info.update({
            "explicit_opening_pull_active": active,
            "explicit_opening_pull_cmd": scheduled_pitch,
        })
        return ActionResult(
            action=action,
            source=f"ep3<{result.source}>",
            confidence=result.confidence,
            info=info,
        )
