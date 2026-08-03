from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from dogfight.envs.initial_scenario import validate_scenario_pool
from student.reward_search_curriculum import CONFIRM_SCENARIO_POOL
from scripts.run_altitude_c05_refine import make_experiment, run_jobs, variant
from scripts.run_altitude_reward_search import (
    PARAMETER_KEYS,
    eligible_candidates,
    load_yaml,
    write_validation_summary,
)


R08 = {
    "altitude_soft_floor_m": 3000.0,
    "altitude_hard_floor_m": 1000.0,
    "low_altitude_penalty": 0.90,
    "very_low_altitude_penalty": 2.88,
    "altitude_bonus_high_min_m": 3000.0,
    "altitude_bonus_high_max_m": 9000.0,
    "altitude_bonus_mid_min_m": 1000.0,
    "altitude_bonus_high": 0.60,
    "altitude_bonus_mid": 0.25,
    "nose_down_altitude_m": 3800.0,
    "nose_down_pitch_deg": -8.0,
    "nose_down_penalty": -1.40,
    "crash_penalty": -280.0,
}


def r08_variant(label: str, **changes: float) -> tuple[str, dict[str, float]]:
    params = dict(R08)
    params.update(changes)
    return label, params


CANDIDATES = dict(
    [
        r08_variant("Q00_r08_reference"),
        r08_variant("Q01_nose_alt3500", nose_down_altitude_m=3500.0),
        r08_variant("Q02_nose_alt4100", nose_down_altitude_m=4100.0),
        r08_variant("Q03_nose_pen120", nose_down_penalty=-1.20),
        r08_variant("Q04_nose_pen160", nose_down_penalty=-1.60),
        r08_variant("Q05_bonus_high055", altitude_bonus_high=0.55),
        r08_variant("Q06_bonus_high065", altitude_bonus_high=0.65),
        r08_variant("Q07_bonus_mid030", altitude_bonus_mid=0.30),
        r08_variant(
            "Q08_stability_combo",
            altitude_bonus_high=0.65,
            altitude_bonus_mid=0.30,
            nose_down_altitude_m=4000.0,
            nose_down_penalty=-1.50,
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine search around R08 altitude reward.")
    parser.add_argument("--base", default="experiments/altitude_reward_search_base.yaml")
    parser.add_argument("--campaign", default="altitude_r08_refine_v1")
    parser.add_argument("--screen-iterations", type=int, default=150)
    parser.add_argument("--validation-iterations", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.screen_iterations, args.validation_iterations, args.top_k, args.repeats) < 1:
        raise ValueError("iteration, top-k, and repeat values must be positive")
    scenarios = validate_scenario_pool(CONFIRM_SCENARIO_POOL)
    if not all(item.get("target_mode") == "loiter" for item in scenarios):
        raise RuntimeError("R08 refinement requires the level loiter target")

    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign

    screen_jobs = [
        (f"Q{index:02d}", label, 1, params)
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
        raise RuntimeError("No valid R08 refinement candidates were produced")
    validation_jobs = []
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
