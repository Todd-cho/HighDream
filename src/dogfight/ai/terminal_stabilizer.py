"""RL-first terminal stabilizer for live Unreal inference.

The inner policy keeps full authority during acquisition.  Once it has already
created a close firing opportunity, this layer only removes axis saturation and
command chatter; it does not replace the RL aim point with another pursuit law.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action


@dataclass
class TerminalStabilizerConfig:
    enter_ata_deg: float = 22.0
    exit_ata_deg: float = 35.0
    enter_distance_m: float = 2200.0
    exit_distance_m: float = 3000.0
    # Retain enough authority to follow a turning target, but prevent the
    # +-1 discontinuities seen in run0218 (65/83/80% axis saturation).
    roll_limit: float = 0.78
    pitch_limit: float = 0.68
    yaw_limit: float = 0.42
    smoothing_alpha: float = 0.38
    max_delta_roll: float = 0.30
    max_delta_pitch: float = 0.24
    max_delta_yaw: float = 0.20
    terminal_throttle_floor: float = 0.72


class TerminalStabilizerActionProvider(ActionProvider):
    def __init__(self, inner: ActionProvider, config: TerminalStabilizerConfig | None = None):
        self.inner = inner
        self.cfg = config or TerminalStabilizerConfig()
        self.geometry = GeometryInfo()
        self._active = False
        self._previous: np.ndarray | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self.inner.reset(context)
        self._active = False
        self._previous = None

    def close(self) -> None:
        self.inner.close()

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.inner.compute_action(context)
        raw = clip_action(result.action)
        own = context.ownship_state
        target = context.target_state
        if own is None or target is None:
            self._previous = raw.copy()
            return result

        distance = float(self.geometry._get_distance(own, target))
        ata = abs(float(self.geometry._get_antenna_train_angle(own, target, False)))
        if self._active:
            self._active = ata < self.cfg.exit_ata_deg and distance < self.cfg.exit_distance_m
        else:
            self._active = ata <= self.cfg.enter_ata_deg and distance <= self.cfg.enter_distance_m

        if not self._active:
            # Do not smooth acquisition: run0218 proved raw v8 can reach the
            # firing cone while every rule-dominant acquisition candidate did not.
            self._previous = raw.copy()
            return ActionResult(
                action=raw,
                source="terminal_stabilizer[rl_acquisition]",
                confidence=result.confidence,
                info={**result.info, "terminal_stabilizer_active": False, "ata": ata, "distance": distance},
            )

        limited = raw.copy()
        limits = np.asarray(
            [self.cfg.roll_limit, self.cfg.pitch_limit, self.cfg.yaw_limit], dtype=np.float32
        )
        limited[:3] = np.clip(limited[:3], -limits, limits)
        limited[3] = max(float(limited[3]), self.cfg.terminal_throttle_floor)

        if self._previous is not None:
            alpha = self.cfg.smoothing_alpha
            filtered = self._previous + alpha * (limited - self._previous)
            max_delta = np.asarray(
                [self.cfg.max_delta_roll, self.cfg.max_delta_pitch, self.cfg.max_delta_yaw, 0.25],
                dtype=np.float32,
            )
            limited = self._previous + np.clip(filtered - self._previous, -max_delta, max_delta)
        limited = clip_action(limited)
        self._previous = limited.copy()
        return ActionResult(
            action=limited,
            source="terminal_stabilizer[terminal]",
            confidence=result.confidence,
            info={**result.info, "terminal_stabilizer_active": True, "ata": ata, "distance": distance},
        )


__all__ = ["TerminalStabilizerActionProvider", "TerminalStabilizerConfig"]
