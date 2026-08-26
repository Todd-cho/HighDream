"""One-shot paired final holdout for frozen W56 rule/residual candidates.

The episode matrix is explicit: 4 opponents x 2 angles x 4 fresh seeds.  Do
not tune against this output; create a new development set if follow-up work is
needed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from collections import defaultdict
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


OPPONENTS = ("straight_level", "steady_turn", "scripted_pursuit", "vertical_weave")
CANDIDATES = {
    "rule_gain100": {"gain": 1.0, "pitch": 0.0, "policy": False},
    "rule_gain150": {"gain": 1.5, "pitch": 0.0, "policy": False},
    "residual_pitch200": {"gain": 1.5, "pitch": 0.20, "policy": True},
    "residual_pitch225": {"gain": 1.5, "pitch": 0.225, "policy": True},
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def bootstrap_paired_ci(deltas: list[float], seed: int = 20260823) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(deltas, dtype=np.float64)
    means = np.empty(10000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_episode(algorithm, base_config: dict, case: dict, candidate: dict) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["max_engage_time"] = 180.0
    cfg["episode_step_limit"] = 1801
    cfg["initial_scenario"] = {
        "mode": "two_circle_headon",
        "alpha_deg": float(case["alpha_deg"]),
        "center_n_m": 3500.0,
        "altitude_m": 3048.0,
        "turn_diameter_ft": 6000.0,
        "separation_jitter_ft": [3000.0, 6000.0],
        "speed_mps_range": [230.0, 280.0],
        "roll_range_deg": [0.0, 20.0],
        "vertical_pitch_choices_deg": [0.0, 10.0, -10.0],
    }
    cfg["opponent_forced_sequence"] = [{
        "type": case["opponent"],
        "name": case["opponent"],
        "param_seed": int(case["seed"]),
    }]
    residual = cfg.setdefault("residual_w56", {})
    residual["max_roll_scale"] = 0.15 if candidate["policy"] else 0.0
    residual["max_pitch_scale"] = float(candidate["pitch"])
    residual["max_throttle_scale"] = 0.15 if candidate["policy"] else 0.0
    residual.setdefault("w56_overrides", {})["terminal_pitch_los_rate_gain"] = float(candidate["gain"])
    cfg.pop("holdout_initial_scenario", None)
    cfg.pop("episode_summary_path", None)

    env = train_rllib.env_creator(cfg)
    obs, reset_info = env.reset(seed=int(case["seed"]))
    terminated = truncated = False
    info = dict(reset_info)
    raw_abs_sum = np.zeros(3, dtype=np.float64)
    raw_max = np.zeros(3, dtype=np.float64)
    saturation_steps = gate_steps = steps = 0
    try:
        while not (terminated or truncated):
            raw = (
                train_rllib._raw_module_action(algorithm, "default_policy", obs)
                if candidate["policy"]
                else np.zeros(3, dtype=np.float32)
            )
            raw = np.asarray(raw, dtype=np.float32)
            raw_abs_sum += np.abs(raw)
            raw_max = np.maximum(raw_max, np.abs(raw))
            saturation_steps += int(np.any(np.abs(raw) >= 0.95))
            obs, _reward, terminated, truncated, info = env.step(raw)
            gate_steps += int(bool(info.get("gate_active", False)))
            steps += 1
    finally:
        env.close()
    target_loss = 1.0 - float(info.get("target_health", 1.0))
    own_loss = 1.0 - float(info.get("ownship_health", 1.0))
    return {
        **case,
        "outcome": info.get("outcome", "other"),
        "end_condition": info.get("end_condition"),
        "steps": steps,
        "target_health_loss": target_loss,
        "ownship_health_loss": own_loss,
        "damage_exchange": target_loss - own_loss,
        "gate_active_steps": gate_steps,
        "gate_active_rate": gate_steps / max(1, steps),
        "residual_abs_mean": (raw_abs_sum / max(1, steps)).tolist(),
        "residual_abs_max": raw_max.tolist(),
        "residual_saturation_rate": saturation_steps / max(1, steps),
        "safety_override_steps": int(info.get("ep_safety_override_steps", 0)),
        "min_altitude_m": float(info.get("ep_min_altitude_m", math.nan)),
        "wez_steps": int(info.get("ep_wez_steps", 0)),
    }


def aggregate(rows: list[dict]) -> dict:
    exchanges = [row["damage_exchange"] for row in rows]
    outcomes = [row["outcome"] for row in rows]
    axes = np.asarray([row["residual_abs_mean"] for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "win_rate": outcomes.count("win") / len(rows),
        "loss_rate": outcomes.count("loss") / len(rows),
        "draw_rate": outcomes.count("draw") / len(rows),
        "crash_rate": outcomes.count("crash") / len(rows),
        "mean_target_health_loss": statistics.fmean(row["target_health_loss"] for row in rows),
        "mean_ownship_health_loss": statistics.fmean(row["ownship_health_loss"] for row in rows),
        "mean_damage_exchange": statistics.fmean(exchanges),
        "median_damage_exchange": statistics.median(exchanges),
        "worst_quartile_threshold": percentile(exchanges, 0.25),
        "mean_gate_active_rate": statistics.fmean(row["gate_active_rate"] for row in rows),
        "gate_active_episode_rate": sum(row["gate_active_steps"] > 0 for row in rows) / len(rows),
        "mean_residual_abs_by_axis": axes.mean(axis=0).tolist(),
        "mean_residual_saturation_rate": statistics.fmean(row["residual_saturation_rate"] for row in rows),
        "total_safety_override_steps": sum(row["safety_override_steps"] for row in rows),
        "minimum_altitude_m": min(row["min_altitude_m"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiments/residual_w56_gain15_wezcurr_200iter.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/highdream/residual_w56_gain15_wezcurr_200iter_seed01/checkpoint_000050")
    parser.add_argument("--seed-base", type=int, default=2026082300)
    parser.add_argument("--output", default="artifacts/evaluations/w56_final_holdout_32ep_20260823.json")
    args = parser.parse_args()

    experiment = yaml.safe_load((ROOT / args.experiment).read_text(encoding="utf-8"))
    base_config = copy.deepcopy(experiment["env_config"])
    base_config.update(experiment["env"])
    cases = []
    for opponent_index, opponent in enumerate(OPPONENTS):
        for alpha in (0.0, 91.0):
            for repeat in range(4):
                cases.append({
                    "case_id": f"{opponent}_a{int(alpha):02d}_r{repeat}",
                    "opponent": opponent,
                    "alpha_deg": alpha,
                    "seed": args.seed_base + opponent_index * 100 + int(alpha) * 10 + repeat,
                })

    register_env("dogfight-single-agent-v0", train_rllib.env_creator)
    train_rllib._ensure_ray_runtime_env()
    algorithm = Algorithm.from_checkpoint(str((ROOT / args.checkpoint).resolve()))
    candidate_rows: dict[str, list[dict]] = {}
    try:
        for name, candidate in CANDIDATES.items():
            rows = []
            for index, case in enumerate(cases, 1):
                row = run_episode(algorithm, base_config, case, candidate)
                rows.append(row)
                print(
                    f"[{name}] {index:02d}/32 {case['case_id']} "
                    f"outcome={row['outcome']} exchange={row['damage_exchange']:.6f}"
                )
            candidate_rows[name] = rows
    finally:
        algorithm.stop()

    summaries = {name: aggregate(rows) for name, rows in candidate_rows.items()}
    paired = {}
    for left, right in (("residual_pitch200", "rule_gain150"), ("residual_pitch225", "rule_gain150"), ("residual_pitch225", "residual_pitch200")):
        deltas = [a["damage_exchange"] - b["damage_exchange"] for a, b in zip(candidate_rows[left], candidate_rows[right])]
        paired[f"{left}_minus_{right}"] = {
            "mean_delta": statistics.fmean(deltas),
            "median_delta": statistics.median(deltas),
            "bootstrap_95pct_ci": bootstrap_paired_ci(deltas),
            "positive_cases": sum(delta > 0 for delta in deltas),
            "negative_cases": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        }
    by_angle = {}
    by_opponent = {}
    for name, rows in candidate_rows.items():
        by_angle[name] = {str(int(alpha)): aggregate([row for row in rows if row["alpha_deg"] == alpha]) for alpha in (0.0, 91.0)}
        by_opponent[name] = {opponent: aggregate([row for row in rows if row["opponent"] == opponent]) for opponent in OPPONENTS}

    payload = {
        "final_test_warning": "One-shot final set. Do not tune against these seeds.",
        "seed_base": args.seed_base,
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "cases": cases,
        "summaries": summaries,
        "paired_comparisons": paired,
        "by_angle": by_angle,
        "by_opponent": by_opponent,
        "episode_rows": candidate_rows,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "paired_comparisons": paired}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
