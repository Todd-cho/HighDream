"""Live Unreal action provider for the frozen W56 residual SAC policy."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import copy

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle
from dogfight.ai.integrated_bfm_controller import IntegratedBFMController
from dogfight.ai.rule_profiles import build_w56_config
from dogfight.envs.observation import build_observation, reset_tactical19_history
from dogfight.sim.state_schema import StateIndex
from GeoMathUtil import GeometryInfo


DEFAULT_WEZ = {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4,
}


def compose_w56_residual_action(
    rule_action: np.ndarray,
    raw_residual: np.ndarray,
    active: bool,
    max_roll_scale: float,
    max_pitch_scale: float,
    max_throttle_scale: float,
) -> np.ndarray:
    """Compose the bounded 3-axis residual while W56 keeps yaw control."""
    rule = np.asarray(rule_action, dtype=np.float32)
    residual = np.asarray(raw_residual, dtype=np.float32)
    if rule.shape != (4,) or residual.shape != (3,):
        raise ValueError(
            f"Expected rule (4,) and residual (3,), got {rule.shape} and {residual.shape}"
        )
    combined = rule.copy()
    if active:
        bounded = np.clip(residual, -1.0, 1.0)
        combined[0] += bounded[0] * max(0.0, float(max_roll_scale))
        combined[1] += bounded[1] * max(0.0, float(max_pitch_scale))
        combined[3] += bounded[2] * max(0.0, float(max_throttle_scale))
    return clip_action(combined)


class W56ResidualActionProvider(ActionProvider):
    """Compose a 3-axis residual bundle over the frozen W56 rule action."""

    def __init__(
        self,
        bundle_dir: str,
        algorithm_factory: Callable[[dict], object],
        policy_id: str = "default_policy",
        roll_scale: float = 0.15,
        pitch_scale: float = 0.20,
        throttle_scale: float = 0.15,
        terminal_pitch_los_rate_gain: float = 1.5,
        gate_ata_deg: float = 20.0,
        gate_range_m: float = 3000.0,
        gate_min_threat_ata_deg: float = 40.0,
        force_zero_residual: bool = False,
        safety_altitude_m: float = 500.0,
        safety_pitch_deg: float = -5.0,
        safety_time_horizon_s: float = 15.0,
        safety_hard_floor_m: float = 400.0,
        min_altitude_m: float = 304.8,
        rule_provider: ActionProvider | None = None,
    ):
        self.policy_id = policy_id
        self.roll_scale = float(roll_scale)
        self.pitch_scale = float(pitch_scale)
        self.throttle_scale = float(throttle_scale)
        self.gate_ata_deg = float(gate_ata_deg)
        self.gate_range_m = float(gate_range_m)
        self.gate_min_threat_ata_deg = float(gate_min_threat_ata_deg)
        self.force_zero_residual = bool(force_zero_residual)
        self.safety_altitude_m = float(safety_altitude_m)
        self.safety_pitch_deg = float(safety_pitch_deg)
        self.safety_time_horizon_s = float(safety_time_horizon_s)
        self.safety_hard_floor_m = float(safety_hard_floor_m)
        self.min_altitude_m = float(min_altitude_m)
        self.bundle_dir = str(Path(bundle_dir).resolve())

        if rule_provider is None:
            cfg = build_w56_config()
            cfg.terminal_pitch_los_rate_gain = float(terminal_pitch_los_rate_gain)
            self.rule = IntegratedBFMController(cfg)
        else:
            self.rule = rule_provider
        self.geometry = GeometryInfo()
        self._previous_raw_residual = np.zeros(3, dtype=np.float32)
        self._previous_altitude: float | None = None
        self._previous_time: float | None = None

        loaded_metadata, weights = load_lightweight_policy_bundle(Path(bundle_dir))
        self.metadata = self._w56_inference_metadata(loaded_metadata)
        self.algorithm = algorithm_factory(self.metadata)
        module = self.algorithm.get_module(policy_id)
        if module is None:
            module = self.algorithm.get_module()
        if module is None or not hasattr(module, "set_state"):
            raise RuntimeError("Unable to locate the W56 residual RLModule")
        module.set_state(weights)

    def reset(self, context: ActionContext | None = None) -> None:
        self.rule.reset()
        self._previous_raw_residual.fill(0.0)
        self._previous_altitude = None
        self._previous_time = None
        reset_tactical19_history()

    @staticmethod
    def _w56_inference_metadata(metadata: dict) -> dict:
        """Repair stale generic bundle spaces for the residual W56 policy.

        The frozen bundle contains a 36-input/3-output RLModule, while legacy
        summary fields still describe the pre-residual tactical16 controller.
        RLlib builds layer shapes from env_config, so force the spaces that are
        intrinsic to ResidualW56Env before constructing the inference module.
        """
        restored = copy.deepcopy(metadata)
        algorithm_config = restored.setdefault("algorithm_config", {})
        env_config = algorithm_config.setdefault("env_config", {})
        env_config["observation_mode"] = "residual_w56_36"
        env_config["observation_size"] = 36
        env_config["observation_summary"] = {
            "mode": "residual_w56_36",
            "size": 36,
        }
        env_config["action_size"] = 3
        return restored

    def _raw_module_action(self, observation: np.ndarray) -> np.ndarray:
        import torch
        from ray.rllib.core.columns import Columns

        module = self.algorithm.get_module(self.policy_id)
        if module is None:
            module = self.algorithm.get_module()
        batch = {Columns.OBS: torch.as_tensor(observation[None, :], dtype=torch.float32)}
        with torch.no_grad():
            output = module.forward_inference(batch)
        action = output.get(Columns.ACTIONS)
        if action is None:
            distribution = module.get_inference_action_dist_cls().from_logits(
                output[Columns.ACTION_DIST_INPUTS]
            )
            action = distribution.to_deterministic().sample()
        raw = np.asarray(action.detach().cpu().numpy()[0], dtype=np.float32)
        if raw.shape != (3,):
            raise ValueError(f"W56 residual bundle must output shape (3,), got {raw.shape}")
        return np.clip(raw, -1.0, 1.0)

    def _danger_active(self, ownship_state: np.ndarray) -> bool:
        altitude = float(ownship_state[StateIndex.ALT])
        pitch = float(ownship_state[StateIndex.PITCH])
        now = float(ownship_state[StateIndex.SIM_TIME])
        vertical_rate = None
        if self._previous_altitude is not None and self._previous_time is not None and now > self._previous_time:
            vertical_rate = (altitude - self._previous_altitude) / (now - self._previous_time)
        self._previous_altitude = altitude
        self._previous_time = now
        if altitude < self.safety_hard_floor_m:
            return True
        if altitude < self.safety_altitude_m and pitch <= self.safety_pitch_deg:
            return True
        if vertical_rate is not None and vertical_rate < 0.0:
            time_to_impact = (altitude - self.min_altitude_m) / max(1.0, -vertical_rate)
            return time_to_impact < self.safety_time_horizon_s
        return False

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float32)
        target = np.asarray(context.target_state, dtype=np.float32)
        rule_result = self.rule.compute_action(context)
        rule_info = dict(rule_result.info)
        danger = self._danger_active(own)
        distance = float(rule_info.get("distance", float("inf")))
        ata = abs(float(rule_info.get("ata", 180.0)))
        threat_ata = abs(float(rule_info.get("threat_ata", 180.0)))
        gate_active = (
            ata <= self.gate_ata_deg
            and distance <= self.gate_range_m
            and threat_ata >= self.gate_min_threat_ata_deg
            and not danger
        )

        obs = np.zeros(36, dtype=np.float32)
        obs[:19] = build_observation("tactical19", own, target, self.geometry, DEFAULT_WEZ)
        obs[19:22] = np.clip(rule_result.action[:3], -1.0, 1.0)
        obs[22] = np.clip(2.0 * float(rule_result.action[3]) - 1.0, -1.0, 1.0)
        obs[23] = np.clip(float(rule_info.get("measured_roll_rate", 0.0)) / 90.0, -1.0, 1.0)
        obs[24] = np.clip(float(rule_info.get("measured_pitch_rate", 0.0)) / 45.0, -1.0, 1.0)
        obs[25] = np.clip(float(rule_info.get("own_yaw_rate", 0.0)) / 45.0, -1.0, 1.0)
        obs[26] = np.clip(float(rule_info.get("target_yaw_rate", 0.0)) / 45.0, -1.0, 1.0)
        obs[27] = np.clip(float(rule_info.get("los_rate", 0.0)) / 45.0, -1.0, 1.0)
        obs[28] = np.clip(float(rule_info.get("closure_rate", 0.0)) / 600.0, -1.0, 1.0)
        obs[29] = 1.0 if rule_info.get("terminal_track_active", False) else -1.0
        obs[30] = 1.0 if rule_info.get("turn_match_active", False) else -1.0
        obs[31] = np.clip(threat_ata / 180.0, -1.0, 1.0)
        obs[32] = 1.0 if gate_active else -1.0
        obs[33:36] = self._previous_raw_residual
        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)

        raw = np.zeros(3, dtype=np.float32) if self.force_zero_residual else self._raw_module_action(obs)
        combined = compose_w56_residual_action(
            rule_result.action, raw, gate_active,
            self.roll_scale, self.pitch_scale, self.throttle_scale,
        )
        scaled = np.zeros(3, dtype=np.float32)
        if gate_active:
            scaled = raw * np.asarray([self.roll_scale, self.pitch_scale, self.throttle_scale], dtype=np.float32)
        self._previous_raw_residual = raw.copy()
        info = {
            **rule_info,
            "rule_action": np.asarray(rule_result.action, dtype=np.float32).tolist(),
            "raw_residual": raw.tolist(),
            "scaled_residual": scaled.tolist(),
            "final_action": combined.tolist(),
            "residual_gate_active": gate_active,
            "residual_gate_block_reason": "" if gate_active else (
                "safety" if danger else "ata" if ata > self.gate_ata_deg else
                "range" if distance > self.gate_range_m else "threat_ata"
            ),
            "observation_mode": "residual_w56_36",
            "observation_dimension": 36,
            "bundle_id": self.bundle_dir,
            "safety_override_active": danger,
        }
        return ActionResult(action=combined, source="residual_rl", confidence=1.0, info=info)

    def close(self) -> None:
        if hasattr(self.algorithm, "stop"):
            self.algorithm.stop()
