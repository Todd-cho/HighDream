from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from dogfight.envs.initial_scenario import validate_scenario_pool
from student.reward_search_curriculum import CONFIRM_SCENARIO_POOL
from scripts.run_altitude_reward_search import (
    PARAMETER_KEYS,
    build_experiment,
    dashboard_path,
    load_yaml,
    read_csv,
    run_experiment,
    summarize,
    write_csv,
    write_validation_summary,
    write_yaml,
)


CANDIDATES: dict[str, dict[str, float]] = {
    "C05": {
        "altitude_soft_floor_m": 3000.0,
        "altitude_hard_floor_m": 1000.0,
        "low_altitude_penalty": 0.90,
        "very_low_altitude_penalty": 2.88,
        "altitude_bonus_high_min_m": 3000.0,
        "altitude_bonus_high_max_m": 9000.0,
        "altitude_bonus_mid_min_m": 1000.0,
        "altitude_bonus_high": 0.45,
        "altitude_bonus_mid": 0.20,
        "nose_down_altitude_m": 3300.0,
        "nose_down_pitch_deg": -8.0,
        "nose_down_penalty": -0.98,
        "crash_penalty": -280.0,
    },
    "C07": {
        "altitude_soft_floor_m": 2600.0,
        "altitude_hard_floor_m": 1800.0,
        "low_altitude_penalty": 1.68,
        "very_low_altitude_penalty": 4.65,
        "altitude_bonus_high_min_m": 2600.0,
        "altitude_bonus_high_max_m": 9000.0,
        "altitude_bonus_mid_min_m": 1800.0,
        "altitude_bonus_high": 0.45,
        "altitude_bonus_mid": 0.35,
        "nose_down_altitude_m": 2900.0,
        "nose_down_pitch_deg": -8.0,
        "nose_down_penalty": -0.85,
        "crash_penalty": -200.0,
    },
    "C11": {
        "altitude_soft_floor_m": 2600.0,
        "altitude_hard_floor_m": 1000.0,
        "low_altitude_penalty": 0.95,
        "very_low_altitude_penalty": 2.84,
        "altitude_bonus_high_min_m": 2600.0,
        "altitude_bonus_high_max_m": 9000.0,
        "altitude_bonus_mid_min_m": 1000.0,
        "altitude_bonus_high": 0.67,
        "altitude_bonus_mid": 0.21,
        "nose_down_altitude_m": 3500.0,
        "nose_down_pitch_deg": -8.0,
        "nose_down_penalty": -1.43,
        "crash_penalty": -240.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirm C05, C07, and C11.")
    parser.add_argument("--base", default="experiments/altitude_reward_search_base.yaml")
    parser.add_argument("--campaign", default="altitude_c050711_confirm_v1")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_experiment(
    base: dict[str, Any], tag: str, iterations: int, params: dict[str, float]
) -> dict[str, Any]:
    experiment = build_experiment(base, tag, iterations, params, validation=True)
    experiment.setdefault("env", {})["target_mode"] = "loiter"
    experiment.setdefault("env_config", {})["initial_scenario"] = copy.deepcopy(
        CONFIRM_SCENARIO_POOL
    )
    engagement = experiment.setdefault("engagement_log", {})
    engagement.update(
        {"enabled": True, "interval": 20, "steps": 1200, "episodes": 2, "print": False}
    )
    return experiment


def main() -> int:
    args = parse_args()
    if args.iterations < 1 or args.repeats < 1:
        raise ValueError("iterations and repeats must be positive")
    validate_scenario_pool(CONFIRM_SCENARIO_POOL)
    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign
    results_path = work / "confirmation_results.csv"
    existing = read_csv(results_path) if args.resume else []
    completed = {row["trial"] for row in existing if row.get("status") == "success"}
    results: list[dict[str, Any]] = list(existing)

    jobs = [
        (candidate, repeat, params)
        for candidate, params in CANDIDATES.items()
        for repeat in range(1, args.repeats + 1)
    ]
    for index, (candidate, repeat, params) in enumerate(jobs, 1):
        trial = f"{index:03d}"
        tag = f"{args.campaign}_{candidate}_r{repeat}_{args.iterations}iter"
        yaml_path = work / "generated" / f"{trial}_{candidate}_r{repeat}.yaml"
        experiment = make_experiment(base, tag, args.iterations, params)
        write_yaml(yaml_path, experiment)
        if trial in completed:
            print(f"[resume] skip completed {trial} {candidate} repeat {repeat}")
            continue
        code = run_experiment(
            yaml_path, work / "logs" / f"{trial}_{candidate}_r{repeat}.log", args.dry_run
        )
        if args.dry_run:
            continue
        row = {
            "trial": trial,
            "candidate": candidate,
            "repeat": repeat,
            "label": "targeted_confirm",
            "tag": tag,
            "status": "success" if code == 0 else "failed",
            "return_code": code,
            **params,
            **summarize(dashboard_path(experiment)),
        }
        results = [old for old in results if old.get("trial") != trial]
        results.append(row)
        results.sort(key=lambda item: int(item["trial"]))
        write_csv(results_path, results)

    if not args.dry_run:
        write_validation_summary(work / "confirmation_summary.csv", results)
        print(f"[done] summary: {work / 'confirmation_summary.csv'}")
    print(f"[done] results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
