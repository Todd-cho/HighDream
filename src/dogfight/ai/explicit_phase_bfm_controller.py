"""EP1: output-equivalent W111 controller with explicit phase/owner telemetry."""

from __future__ import annotations

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.control_arbitration import diagnose_control_ownership
from dogfight.ai.phase_manager import classify_phase


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
