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

    def reset(self, context: ActionContext | None = None) -> None:
        super().reset(context)
        self._commit_sign = 0
        self._commit_since = 0.0
        self._candidate_sign = 0
        self._candidate_since = 0.0

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

        if self._commit_sign == 0 and bool(info.get("first_merge_passed", False)):
            self._commit_sign = self._sign(target_bank) or self._sign(target_turn_rate) or 1
            self._commit_since = now

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
        })
        return ActionResult(
            action=action,
            source=f"ep2<{observed.source}>",
            confidence=observed.confidence,
            info=info,
        )
