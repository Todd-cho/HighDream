"""Hand off between an RL policy and a live-proven rule controller by ATA.

2026-08-25: tactical_wrapper.py's own "neutral" state rule engine (a
from-scratch lead-pursuit implementation) was live-verified to never
converge ATA from a 91deg start -- ata stayed 60-170deg for 87s straight and
the sustained pull-up it commanded put the aircraft in a ~56deg nose-up
attitude, climbing from 4566m to 8673m. Live logs from today's W110/W111/
W112/W113 sessions show none of them recover cleanly either in isolation,
but they are the controllers this project has actually flown live
repeatedly today, so they are the better-understood fallback -- unlike
tactical_wrapper's own rule path, which had never been live-tested at all
before today's failure.

This module does NOT touch tactical_wrapper.py. It's a separate, minimal
composition: call the rule controller (W112 by default -- see
run_unreal_inference.py's w112 branch, which is W111 plus the opening-pull
fix and a wider terminal-track/attack-conversion aspect gate) whenever ATA
is outside the RL policy's trained comfort zone (roughly 0-22deg per
tactical_wrapper.py's own docstring), and let the RL policy fly once ATA is
actually small. Only the currently-selected provider is called each frame
(not both) so this doesn't add extra per-frame compute on top of the
multiprocess-transport fix.
"""
from __future__ import annotations

from GeoMathUtil import GeometryInfo

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult


class RLRuleHandoffActionProvider(ActionProvider):
    def __init__(
        self,
        rl_provider: ActionProvider,
        rule_provider: ActionProvider,
        enter_rl_ata_deg: float = 20.0,
        exit_rl_ata_deg: float = 28.0,
    ) -> None:
        if exit_rl_ata_deg < enter_rl_ata_deg:
            raise ValueError("exit_rl_ata_deg must be >= enter_rl_ata_deg (hysteresis)")
        self.rl_provider = rl_provider
        self.rule_provider = rule_provider
        self.enter_rl_ata_deg = enter_rl_ata_deg
        self.exit_rl_ata_deg = exit_rl_ata_deg
        self.geometry = GeometryInfo()
        self._using_rl = False

    def reset(self, context: ActionContext | None = None) -> None:
        self.rl_provider.reset(context)
        self.rule_provider.reset(context)
        self._using_rl = False

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            result = self.rule_provider.compute_action(context)
            info = dict(result.info)
            info["handoff_mode"] = "rule"
            info["handoff_ata_deg"] = None
            return ActionResult(
                action=result.action,
                source=f"handoff[rule]<{result.source}>",
                confidence=result.confidence,
                info=info,
            )

        ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
        # Hysteresis so a converged solution doesn't flap RL<->rule around a
        # single threshold -- same pattern tactical_wrapper.py itself uses
        # for its own state transitions.
        if self._using_rl:
            self._using_rl = ata <= self.exit_rl_ata_deg
        else:
            self._using_rl = ata <= self.enter_rl_ata_deg

        if self._using_rl:
            result = self.rl_provider.compute_action(context)
            mode = "rl"
        else:
            result = self.rule_provider.compute_action(context)
            mode = "rule"

        info = dict(result.info)
        info["handoff_mode"] = mode
        info["handoff_ata_deg"] = ata
        return ActionResult(
            action=result.action,
            source=f"handoff[{mode}]<{result.source}>",
            confidence=result.confidence,
            info=info,
        )

    def close(self) -> None:
        self.rl_provider.close()
        self.rule_provider.close()
