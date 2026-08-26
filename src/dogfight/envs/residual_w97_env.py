"""Residual SAC environment around the frozen W97 3-D rule controller."""
from __future__ import annotations

from typing import Any

from dogfight.ai.integrated_bfm_controller import IntegratedBFMController
from dogfight.ai.rule_profiles import build_w97_config
from dogfight.envs.residual_w56_env import ResidualW56Env


DEFAULT_RESIDUAL_W97_CONFIG = {
    "max_roll_scale": 0.10,
    "max_pitch_scale": 0.20,
    "max_throttle_scale": 0.10,
    # Learn the live failure band, not only an already solved gun track.
    "gate_ata_deg": 45.0,
    "gate_range_m": 2500.0,
    "gate_min_threat_ata_deg": 30.0,
    "residual_magnitude_penalty_scale": 0.012,
    "residual_delta_penalty_scale": 0.018,
    "w56_overrides": {},
}


class ResidualW97Env(ResidualW56Env):
    """Keep W56's checkpoint-compatible 36-observation/3-action contract."""

    def __init__(self, env_config: dict | None = None, **kwargs: Any):
        cfg = dict(env_config or {})
        residual = {
            **DEFAULT_RESIDUAL_W97_CONFIG,
            **dict(cfg.get("residual_w97", {}) or {}),
        }
        # Reuse the thoroughly tested residual composition/telemetry path.
        cfg["residual_w56"] = residual
        super().__init__(env_config=cfg, **kwargs)
        self._w56 = IntegratedBFMController(build_w97_config())
        self.config["observation_mode"] = "residual_w97_36"
        self.config["observation_summary"] = {
            "mode": "residual_w97_36",
            "size": self.num_observation,
            "description": "tactical19 + W97 action/rates/state + previous residual",
        }


__all__ = ["DEFAULT_RESIDUAL_W97_CONFIG", "ResidualW97Env"]
