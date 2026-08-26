"""Diagnose the frozen final-test scripted-pursuit cases without tuning."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

import train_rllib


def main() -> None:
    experiment = yaml.safe_load((ROOT / "experiments/residual_w56_gain15_wezcurr_200iter.yaml").read_text(encoding="utf-8"))
    base = copy.deepcopy(experiment["env_config"])
    base.update(experiment["env"])
    base.pop("holdout_initial_scenario", None)
    base.pop("episode_summary_path", None)
    residual = base.setdefault("residual_w56", {})
    residual.update({"max_roll_scale": 0.15, "max_pitch_scale": 0.20, "max_throttle_scale": 0.15})
    residual.setdefault("w56_overrides", {})["terminal_pitch_los_rate_gain"] = 1.5
    base.update({"max_engage_time": 180.0, "episode_step_limit": 1801})

    cases = []
    seed_base = 2026082300
    for alpha in (0.0, 91.0):
        for repeat in range(4):
            cases.append({
                "case_id": f"scripted_pursuit_a{int(alpha):02d}_r{repeat}",
                "alpha_deg": alpha,
                "seed": seed_base + 200 + int(alpha) * 10 + repeat,
            })

    register_env("dogfight-single-agent-v0", train_rllib.env_creator)
    train_rllib._ensure_ray_runtime_env()
    checkpoint = ROOT / "artifacts/checkpoints/highdream/residual_w56_gain15_wezcurr_200iter_seed01/checkpoint_000050"
    algorithm = Algorithm.from_checkpoint(str(checkpoint.resolve()))
    rows = []
    try:
        for case in cases:
            cfg = copy.deepcopy(base)
            cfg["initial_scenario"] = {
                "mode": "two_circle_headon", "alpha_deg": case["alpha_deg"],
                "center_n_m": 3500.0, "altitude_m": 3048.0,
                "turn_diameter_ft": 6000.0, "separation_jitter_ft": [3000.0, 6000.0],
                "speed_mps_range": [230.0, 280.0], "roll_range_deg": [0.0, 20.0],
                "vertical_pitch_choices_deg": [0.0, 10.0, -10.0],
            }
            cfg["opponent_forced_sequence"] = [{"type": "scripted_pursuit", "name": "scripted_pursuit", "param_seed": case["seed"]}]
            env = train_rllib.env_creator(cfg)
            obs, _ = env.reset(seed=case["seed"])
            minima = {"ata_deg": math.inf, "distance_m": math.inf, "threat_ata_deg": math.inf}
            entries = {"distance_le_3000": None, "distance_le_2000": None, "distance_le_1219": None, "ata_le_20": None, "ata_le_3": None, "gate_on": None}
            closure_negative_steps = closure_positive_steps = gate_steps = 0
            raw_abs_sum = np.zeros(3, dtype=np.float64)
            raw_max = np.zeros(3, dtype=np.float64)
            steps = 0
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                rule_info = env._get_rule_result().info
                distance = float(rule_info.get("distance", math.inf))
                ata = abs(float(rule_info.get("ata", 180.0)))
                threat = abs(float(rule_info.get("threat_ata", 0.0)))
                closure = float(rule_info.get("closure_rate", 0.0))
                # obs[32] is the environment-owned gate decision for this
                # exact control step.  Do not call _gate_active() here because
                # its safety predictor intentionally updates private history.
                gate = bool(float(obs[32]) > 0.0)
                minima["ata_deg"] = min(minima["ata_deg"], ata)
                minima["distance_m"] = min(minima["distance_m"], distance)
                minima["threat_ata_deg"] = min(minima["threat_ata_deg"], threat)
                for key, condition in (
                    ("distance_le_3000", distance <= 3000.0), ("distance_le_2000", distance <= 2000.0),
                    ("distance_le_1219", distance <= 1219.0), ("ata_le_20", ata <= 20.0),
                    ("ata_le_3", ata <= 3.0), ("gate_on", gate),
                ):
                    if condition and entries[key] is None:
                        entries[key] = steps * 0.1
                closure_negative_steps += int(closure < 0.0)
                closure_positive_steps += int(closure > 0.0)
                gate_steps += int(gate)
                raw = np.asarray(train_rllib._raw_module_action(algorithm, "default_policy", obs), dtype=np.float32)
                raw_abs_sum += np.abs(raw)
                raw_max = np.maximum(raw_max, np.abs(raw))
                obs, _, terminated, truncated, info = env.step(raw)
                steps += 1
            env.close()
            row = {
                **case, "outcome": info.get("outcome"), "end_condition": info.get("end_condition"), "steps": steps,
                "target_health_loss": 1.0 - float(info.get("target_health", 1.0)),
                "ownship_health_loss": 1.0 - float(info.get("ownship_health", 1.0)),
                "minima": minima, "first_entry_seconds": entries,
                "gate_active_rate": gate_steps / steps,
                "closure_negative_rate": closure_negative_steps / steps,
                "closure_positive_rate": closure_positive_steps / steps,
                "residual_abs_mean": (raw_abs_sum / steps).tolist(), "residual_abs_max": raw_max.tolist(),
                "wez_steps": int(info.get("ep_wez_steps", 0)),
                "safety_override_steps": int(info.get("ep_safety_override_steps", 0)),
                "min_altitude_m": float(info.get("ep_min_altitude_m", math.nan)),
                "final_distance_m": float(info.get("final_distance_m", math.nan)),
                "final_ata_deg": float(info.get("final_ata_deg", math.nan)),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    finally:
        algorithm.stop()
    output = ROOT / "artifacts/evaluations/w56_scripted_pursuit_diagnosis_8ep.json"
    output.write_text(json.dumps({"candidate": "residual_pitch200", "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
