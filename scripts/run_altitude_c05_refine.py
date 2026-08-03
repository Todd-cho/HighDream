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
    eligible_candidates,
    load_yaml,
    read_csv,
    run_experiment,
    summarize,
    write_csv,
    write_validation_summary,
    write_yaml,
)


C05 = {
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
}


def variant(label: str, **changes: float) -> tuple[str, dict[str, float]]:
    params = dict(C05)
    params.update(changes)
    return label, params


# One-factor-at-a-time candidates around C05, followed by one conservative combo.
CANDIDATES = dict(
    [
        variant("B00_c05"),
        variant("N01_nose_alt3800", nose_down_altitude_m=3800.0),
        variant("N02_nose_pen140", nose_down_penalty=-1.40),
        variant("B01_bonus_high060", altitude_bonus_high=0.60),
        variant("B02_bonus_mid030", altitude_bonus_mid=0.30),
        variant(
            "F01_soft3400",
            altitude_soft_floor_m=3400.0,
            altitude_bonus_high_min_m=3400.0,
        ),
        variant("P01_low_pen120", low_altitude_penalty=1.20),
        variant(
            "X01_proactive_combo",
            altitude_bonus_high=0.60,
            altitude_bonus_mid=0.25,
            nose_down_altitude_m=3800.0,
            nose_down_penalty=-1.40,
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine altitude reward around C05.")
    parser.add_argument("--base", default="experiments/altitude_reward_search_base.yaml")
    parser.add_argument("--campaign", default="altitude_c05_refine_v1")
    parser.add_argument("--screen-iterations", type=int, default=100)
    parser.add_argument("--validation-iterations", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_experiment(
    base: dict[str, Any], tag: str, iterations: int, params: dict[str, float], replay: bool
) -> dict[str, Any]:
    experiment = build_experiment(base, tag, iterations, params, validation=False)
    experiment.setdefault("env", {})["target_mode"] = "loiter"
    experiment.setdefault("env_config", {})["initial_scenario"] = copy.deepcopy(
        CONFIRM_SCENARIO_POOL
    )
    engagement = experiment.setdefault("engagement_log", {})
    if replay:
        engagement.update(
            {"enabled": True, "interval": 30, "steps": 1200, "episodes": 2, "print": False}
        )
    else:
        engagement["enabled"] = False
    return experiment


def run_jobs(
    *,
    base: dict[str, Any],
    work: Path,
    phase: str,
    iterations: int,
    jobs: list[tuple[str, str, int, dict[str, float]]],
    resume: bool,
    replay: bool,
) -> list[dict[str, Any]]:
    results_path = work / f"{phase}_results.csv"
    existing = read_csv(results_path) if resume else []
    completed = {row["trial"] for row in existing if row.get("status") == "success"}
    results: list[dict[str, Any]] = list(existing)
    for index, (candidate, label, repeat, params) in enumerate(jobs, 1):
        trial = f"{index:03d}"
        tag = f"{work.name}_{phase}_{candidate}_r{repeat}_{iterations}iter"
        yaml_path = work / "generated" / phase / f"{trial}_{candidate}_r{repeat}.yaml"
        experiment = make_experiment(base, tag, iterations, params, replay)
        write_yaml(yaml_path, experiment)
        if trial in completed:
            print(f"[resume] skip {phase} {trial} {candidate} r{repeat}")
            continue
        code = run_experiment(yaml_path, work / "logs" / phase / f"{trial}.log", False)
        row = {
            "trial": trial,
            "candidate": candidate,
            "repeat": repeat,
            "label": label,
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
    return results


def main() -> int:
    args = parse_args()
    if min(args.screen_iterations, args.validation_iterations, args.top_k, args.repeats) < 1:
        raise ValueError("iteration, top-k, and repeat values must be positive")
    scenarios = validate_scenario_pool(CONFIRM_SCENARIO_POOL)
    if not all(item.get("target_mode") == "loiter" for item in scenarios):
        raise RuntimeError("Confirmation scenarios must use the level loiter target")

    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign

    screen_jobs = [
        (f"R{index:02d}", label, 1, params)
        for index, (label, params) in enumerate(CANDIDATES.items(), 1)
    ]
    run_jobs(
        base=base,
        work=work,
        phase="screen",
        iterations=args.screen_iterations,
        jobs=screen_jobs,
        resume=args.resume,
        replay=False,
    )

    selected = eligible_candidates(work / "screen_results.csv", args.top_k)
    if not selected:
        raise RuntimeError("No valid refined candidates were produced")
    validation_jobs: list[tuple[str, str, int, dict[str, float]]] = []
    for row in selected:
        params = {key: float(row[key]) for key in PARAMETER_KEYS}
        for repeat in range(1, args.repeats + 1):
            validation_jobs.append((row["candidate"], row["label"], repeat, params))
    validation = run_jobs(
        base=base,
        work=work,
        phase="validation",
        iterations=args.validation_iterations,
        jobs=validation_jobs,
        resume=args.resume,
        replay=True,
    )
    write_validation_summary(work / "validation_summary.csv", validation)
    print(f"[done] final ranking: {work / 'validation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
