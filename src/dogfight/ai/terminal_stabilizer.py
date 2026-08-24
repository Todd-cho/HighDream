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
from dogfight.sim.state_schema import StateIndex


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
    precision_blend: float = 0.72
    precision_full_ata_deg: float = 8.0
    az_norm_deg: float = 7.0
    el_norm_deg: float = 6.0
    rate_norm_degps: float = 24.0
    az_rate_gain: float = 0.20
    el_rate_gain: float = 0.16
    los_rate_alpha: float = 0.32


class TerminalStabilizerActionProvider(ActionProvider):
    def __init__(self, inner: ActionProvider, config: TerminalStabilizerConfig | None = None):
        self.inner = inner
        self.cfg = config or TerminalStabilizerConfig()
        self.geometry = GeometryInfo()
        self._active = False
        self._previous: np.ndarray | None = None
        self._previous_time: float | None = None
        self._previous_az: float | None = None
        self._previous_el: float | None = None
        self._az_rate = 0.0
        self._el_rate = 0.0

    def reset(self, context: ActionContext | None = None) -> None:
        self.inner.reset(context)
        self._active = False
        self._previous = None
        self._previous_time = None
        self._previous_az = None
        self._previous_el = None
        self._az_rate = 0.0
        self._el_rate = 0.0

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
        az, el = self.geometry._get_los_angle(own, target)
        az, el = float(az), float(el)
        now = float(own[StateIndex.SIM_TIME])
        if self._previous_time is not None and self._previous_az is not None:
            dt = now - self._previous_time
            if dt > 1e-3:
                az_delta = ((az - self._previous_az + 180.0) % 360.0) - 180.0
                raw_az_rate = az_delta / dt
                raw_el_rate = (el - self._previous_el) / dt
                alpha_rate = self.cfg.los_rate_alpha
                self._az_rate = alpha_rate * raw_az_rate + (1.0 - alpha_rate) * self._az_rate
                self._el_rate = alpha_rate * raw_el_rate + (1.0 - alpha_rate) * self._el_rate
        self._previous_time = now
        self._previous_az = az
        self._previous_el = el
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

        # Actual LOS PD loop.  Unlike the old tactical wrapper this does not
        # invent a lead/lag aim point or select a manoeuvre.  It only drives
        # the measured body-frame LOS error and its rate toward zero after RL
        # has acquired the target.
        precision = raw.copy()
        precision[0] = float(np.clip(
            az / self.cfg.az_norm_deg
            + self.cfg.az_rate_gain * self._az_rate / self.cfg.rate_norm_degps,
            -self.cfg.roll_limit, self.cfg.roll_limit,
        ))
        # Live convention: negative pitch command pulls the nose up.
        precision[1] = float(np.clip(
            -el / self.cfg.el_norm_deg
            - self.cfg.el_rate_gain * self._el_rate / self.cfg.rate_norm_degps,
            -self.cfg.pitch_limit, self.cfg.pitch_limit,
        ))
        precision[2] = float(np.clip(
            0.55 * az / self.cfg.az_norm_deg
            + 0.10 * self._az_rate / self.cfg.rate_norm_degps,
            -self.cfg.yaw_limit, self.cfg.yaw_limit,
        ))
        # Gradually give the precision loop authority, reaching the configured
        # blend by 8deg.  This avoids a discontinuity at the 22deg entry gate.
        convergence = float(np.clip(
            (self.cfg.enter_ata_deg - ata)
            / max(1e-3, self.cfg.enter_ata_deg - self.cfg.precision_full_ata_deg),
            0.0, 1.0,
        ))
        precision_weight = self.cfg.precision_blend * convergence
        limited = (1.0 - precision_weight) * raw + precision_weight * precision
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
            info={
                **result.info,
                "terminal_stabilizer_active": True,
                "terminal_precision_weight": precision_weight,
                "terminal_los_az_deg": az,
                "terminal_los_el_deg": el,
                "terminal_los_az_rate_degps": self._az_rate,
                "terminal_los_el_rate_degps": self._el_rate,
                "ata": ata,
                "distance": distance,
            },
        )


__all__ = ["TerminalStabilizerActionProvider", "TerminalStabilizerConfig"]
