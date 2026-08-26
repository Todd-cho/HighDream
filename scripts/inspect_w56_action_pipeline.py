"""Separate SAC network/distribution/action clipping/scaling stages for W56."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env

import train_rllib


def inference_stages(algorithm, observation: np.ndarray) -> dict:
    module = algorithm.get_module("default_policy") or algorithm.get_module()
    batch = {Columns.OBS: torch.as_tensor(observation[None, :], dtype=torch.float32)}
    with torch.no_grad():
        output = module.forward_inference(batch)
    dist_inputs = output.get(Columns.ACTION_DIST_INPUTS)
    direct = output.get(Columns.ACTIONS)
    if direct is not None:
        deterministic = direct
        action_source = Columns.ACTIONS
    else:
        dist = module.get_inference_action_dist_cls().from_logits(dist_inputs)
        deterministic = dist.to_deterministic().sample()
        action_source = "deterministic_distribution_sample"
    deterministic_np = np.asarray(deterministic.detach().cpu().numpy()[0], dtype=np.float32)
    clipped = np.clip(deterministic_np, -1.0, 1.0)
    return {
        "output_keys": sorted(str(key) for key in output),
        "action_source": action_source,
        "network_action_dist_inputs": None if dist_inputs is None else np.asarray(dist_inputs.detach().cpu().numpy()[0], dtype=np.float32),
        "direct_actions": None if direct is None else np.asarray(direct.detach().cpu().numpy()[0], dtype=np.float32),
        "deterministic_action": deterministic_np,
        "action_space_clipped": clipped,
        "scaled_residual": clipped * np.asarray([0.15, 0.20, 0.15], dtype=np.float32),
    }


def main() -> None:
    experiment = yaml.safe_load((ROOT / "experiments/residual_w56_gain15_wezcurr_200iter.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(experiment["env_config"])
    cfg.update(experiment["env"])
    cfg["initial_scenario"] = {
        "mode": "two_circle_headon", "alpha_deg": 0.0,
        "center_n_m": 3500.0, "altitude_m": 3048.0,
        "turn_diameter_ft": 6000.0, "separation_jitter_ft": [3000.0, 6000.0],
        "speed_mps_range": [230.0, 280.0], "roll_range_deg": [0.0, 20.0],
        "vertical_pitch_choices_deg": [0.0, 10.0, -10.0],
    }
    cfg.pop("holdout_initial_scenario", None)
    cfg.pop("episode_summary_path", None)
    register_env("dogfight-single-agent-v0", train_rllib.env_creator)
    train_rllib._ensure_ray_runtime_env()
    algorithm = Algorithm.from_checkpoint(str((ROOT / "artifacts/checkpoints/highdream/residual_w56_gain15_wezcurr_200iter_seed01/checkpoint_000050").resolve()))
    env = train_rllib.env_creator(cfg)
    obs, _ = env.reset(seed=2026082999)
    records = []
    try:
        for step in range(300):
            stages = inference_stages(algorithm, obs)
            if step in (0, 1, 2, 10, 50, 100, 200, 299):
                records.append({
                    "step": step,
                    **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in stages.items()},
                })
            obs, _, terminated, truncated, _ = env.step(stages["deterministic_action"])
            if terminated or truncated:
                break
    finally:
        env.close()
        algorithm.stop()
    deterministic = np.asarray([row["deterministic_action"] for row in records], dtype=np.float64)
    payload = {
        "finding": "Environment action_space clipping is the authoritative boundary before residual scaling.",
        "samples": records,
        "sampled_deterministic_abs_max_by_axis": np.max(np.abs(deterministic), axis=0).tolist(),
        "any_deterministic_outside_action_space": bool(np.any(np.abs(deterministic) > 1.0)),
    }
    output = ROOT / "artifacts/evaluations/w56_action_pipeline_inspection.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
