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
from scripts.run_altitude_reward_search import (
    PARAMETER_KEYS,
    add_episode_summary_metrics,
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
from student.reward_search_curriculum import CONFIRM_SCENARIO_POOL


R08_BUNDLE = (
    "artifacts/models/highdream/"
    "altitude_c05_refine_v1_validation_R08_r1_150iter"
)


def params(
    *,
    soft: float,
    hard: float,
    low: float,
    very_low: float,
    high_bonus: float,
    mid_bonus: float,
    nose_alt: float,
    nose_pitch: float,
    nose_penalty: float,
    crash: float,
) -> dict[str, float]:
    return {
        "altitude_soft_floor_m": soft,
        "altitude_hard_floor_m": hard,
        "low_altitude_penalty": low,
        "very_low_altitude_penalty": very_low,
        "altitude_bonus_high_min_m": soft,
        "altitude_bonus_high_max_m": 9000.0,
        "altitude_bonus_mid_min_m": hard,
        "altitude_bonus_high": high_bonus,
        "altitude_bonus_mid": mid_bonus,
        "nose_down_altitude_m": nose_alt,
        "nose_down_pitch_deg": nose_pitch,
        "nose_down_penalty": nose_penalty,
        "crash_penalty": crash,
    }


CANDIDATES = {
    "R08": (
        "historical_R08_reference",
        params(
            soft=3000.0, hard=1000.0, low=0.90, very_low=2.88,
            high_bonus=0.60, mid_bonus=0.25, nose_alt=3800.0,
            nose_pitch=-8.0, nose_penalty=-1.40, crash=-280.0,
        ),
    ),
    "C03": (
        "latest_screen_winner",
        params(
            soft=2600.0, hard=1800.0, low=1.34, very_low=5.74,
            high_bonus=0.72, mid_bonus=0.22, nose_alt=2900.0,
            nose_pitch=-8.0, nose_penalty=-1.11, crash=-280.0,
        ),
    ),
    "C10": (
        "latest_screen_runner_up",
        params(
            soft=3400.0, hard=1800.0, low=1.15, very_low=3.45,
            high_bonus=0.50, mid_bonus=0.31, nose_alt=4000.0,
            nose_pitch=-10.0, nose_penalty=-1.13, crash=-280.0,
        ),
    ),
    "C01": (
        "old_final04_reference",
        params(
            soft=3800.0, hard=1800.0, low=1.76, very_low=5.49,
            high_bonus=0.72, mid_bonus=0.43, nose_alt=3200.0,
            nose_pitch=-6.0, nose_penalty=-0.90, crash=-260.0,
        ),
    ),
    "C04": (
        "latest_screen_auxiliary",
        params(
            soft=3400.0, hard=1400.0, low=1.06, very_low=3.94,
            high_bonus=0.43, mid_bonus=0.13, nose_alt=4000.0,
            nose_pitch=-10.0, nose_penalty=-1.29, crash=-280.0,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair loiter-target validation of R08 and latest altitude candidates."
    )
    parser.add_argument(
        "--base",
        default="experiments/altitude_reward_search_base.yaml",
    )
    parser.add_argument(
        "--campaign",
        default="altitude_loiter_validation_v1",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260801)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_experiment(
    base: dict[str, Any],
    tag: str,
    iterations: int,
    seed: int,
    candidate_params: dict[str, float],
) -> dict[str, Any]:
    experiment = build_experiment(
        base,
        tag,
        iterations,
        candidate_params,
        validation=False,
    )
    experiment.setdefault("env", {})["target_mode"] = "loiter"
    experiment.setdefault("env_config", {})["initial_scenario"] = copy.deepcopy(
        CONFIRM_SCENARIO_POOL
    )
    runtime = experiment.setdefault("runtime", {})
    runtime["seed"] = seed
    runtime["init_bundle"] = R08_BUNDLE
    experiment.setdefault("engagement_log", {})["enabled"] = False
    experiment["notes"] = (
        "R08-initialized fair loiter validation; same seed per repeat across candidates."
    )
    return experiment


def main() -> int:
    args = parse_args()
    if args.iterations < 1 or args.repeats < 1:
        raise ValueError("iterations and repeats must be positive")
    scenarios = validate_scenario_pool(CONFIRM_SCENARIO_POOL)
    if not all(item.get("target_mode") == "loiter" for item in scenarios):
        raise RuntimeError("All validation scenarios must use loiter target mode")

    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign
    results_path = work / "validation_results.csv"
    existing = read_csv(results_path) if args.resume else []
    completed = {
        row["trial"] for row in existing if row.get("status") == "success"
    }
    results: list[dict[str, Any]] = list(existing)

    trial_number = 0
    for repeat in range(1, args.repeats + 1):
        repeat_seed = args.seed + repeat - 1
        for candidate, (label, candidate_params) in CANDIDATES.items():
            trial_number += 1
            trial = f"{trial_number:03d}"
            if trial in completed:
                print(f"[resume] skip {trial} {candidate} repeat {repeat}")
                continue
            tag = (
                f"{args.campaign}_{candidate}_r{repeat}_"
                f"seed{repeat_seed}_{args.iterations}iter"
            )
            yaml_path = (
                work / "generated" / f"{trial}_{candidate}_r{repeat}.yaml"
            )
            experiment = make_experiment(
                base,
                tag,
                args.iterations,
                repeat_seed,
                candidate_params,
            )
            write_yaml(yaml_path, experiment)
            code = run_experiment(
                yaml_path,
                work / "logs" / f"{trial}.log",
                args.dry_run,
            )
            if args.dry_run:
                continue
            metrics = add_episode_summary_metrics(
                summarize(dashboard_path(experiment)),
                experiment,
            )
            row = {
                "trial": trial,
                "candidate": candidate,
                "repeat": repeat,
                "label": label,
                "tag": tag,
                "status": "success" if code == 0 else "failed",
                "return_code": code,
                **candidate_params,
                **metrics,
            }
            results = [old for old in results if old.get("trial") != trial]
            results.append(row)
            results.sort(key=lambda item: int(item["trial"]))
            write_csv(results_path, results)

    if not args.dry_run:
        write_validation_summary(work / "validation_summary.csv", results)
        print(f"[done] results: {results_path}")
        print(f"[done] ranking: {work / 'validation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
