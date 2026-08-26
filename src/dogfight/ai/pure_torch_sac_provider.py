from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action


class PureTorchSACActionProvider(ActionProvider):
    """Deterministic SAC actor inference without importing Ray/RLlib."""

    def __init__(self, bundle_dir: str | Path):
        weights_path = Path(bundle_dir) / "policy_weights.pkl.gz"
        with gzip.open(weights_path, "rb") as handle:
            weights = pickle.load(handle)
        self.w0 = np.asarray(weights["pi_encoder.net.mlp.0.weight"], dtype=np.float32)
        self.b0 = np.asarray(weights["pi_encoder.net.mlp.0.bias"], dtype=np.float32)
        self.w1 = np.asarray(weights["pi_encoder.net.mlp.2.weight"], dtype=np.float32)
        self.b1 = np.asarray(weights["pi_encoder.net.mlp.2.bias"], dtype=np.float32)
        self.w2 = np.asarray(weights["pi.net.mlp.0.weight"], dtype=np.float32)
        self.b2 = np.asarray(weights["pi.net.mlp.0.bias"], dtype=np.float32)
        if self.w2.shape[0] != 8 or self.w0.shape[1] != 19:
            raise ValueError(f"Unsupported v8 actor layout: w0={self.w0.shape} w2={self.w2.shape}")

    def reset(self, context: ActionContext | None = None) -> None:
        return None

    def compute_raw_action(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        hidden = np.maximum(self.w0 @ obs + self.b0, 0.0)
        hidden = np.maximum(self.w1 @ hidden + self.b1, 0.0)
        logits = self.w2 @ hidden + self.b2
        # RLActionProvider uses the deterministic SAC distribution mean and
        # clamps the resulting training-space command before throttle remap.
        # Applying tanh here a second time weakens every non-saturated command
        # and no longer reproduces the live RLlib V8 policy.
        return np.clip(logits[:4], -1.0, 1.0).astype(np.float32)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.observation is None:
            raise ValueError("PureTorchSACActionProvider requires an observation")
        action = self.compute_raw_action(context.observation)
        action[3] = (action[3] + 1.0) / 2.0
        return ActionResult(action=clip_action(action), source="pure_torch_sac_v8", confidence=0.9)
