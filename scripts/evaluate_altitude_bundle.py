from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_reward_hook
from dogfight.config import merge_env_config
from student.altitude_safety_pipeline_curriculum import SCENARIOS, scenario_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen RL bundle on altitude scenarios."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes-per-scenario", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260900)
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="Enable the curriculum's small position/attitude randomization.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes_per_scenario < 1:
        raise ValueError("episodes-per-scenario must be positive")
    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_fn, reward_config = load_reward_hook("student.my_reward")
    provider = RLActionProvider(
        bundle_dir=bundle,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
    )
    env_config = merge_env_config({
        "observation_mode": "tactical16",
        "ownship_control_mode": "rl",
        "target_mode": "loiter",
        "max_engage_time": 60.0,
        "episode_step_limit": 3600,
        "reward": reward_config,
        "episode_summary_path": str(output_dir / "episode_summary.csv"),
        "initial_scenario": scenario_pool("level_5000", args.jitter),
    })
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=provider,
        reward_fn=reward_fn,
    )
    try:
        episode_number = 0
        for scenario_name in SCENARIOS:
            options = {
                "initial_scenario": scenario_pool(
                    scenario_name,
                    args.jitter,
                ),
                "target_mode": "loiter",
            }
            for _ in range(args.episodes_per_scenario):
                episode_number += 1
                env.reset(seed=args.seed + episode_number, options=options)
                terminated = truncated = False
                while not (terminated or truncated):
                    _, _, terminated, truncated, _ = env.step(
                        np.zeros(4, dtype=np.float32)
                    )
                print(
                    f"[frozen-eval] {scenario_name} "
                    f"episode {episode_number}/{len(SCENARIOS) * args.episodes_per_scenario}"
                )
    finally:
        env.close()

    paths = sorted(output_dir.glob("episode_summary*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    report_path = output_dir / "frozen_evaluation_summary.csv"
    fields = [
        "scenario", "episodes", "crashes", "crash_rate",
        "minimum_altitude_mean_m", "minimum_altitude_worst_m",
    ]
    reports = []
    for scenario_name in SCENARIOS:
        group = [row for row in rows if row.get("scenario_name") == scenario_name]
        altitudes = [float(row["minimum_altitude_m"]) for row in group]
        crashes = sum(row.get("outcome") == "crash" for row in group)
        reports.append({
            "scenario": scenario_name,
            "episodes": len(group),
            "crashes": crashes,
            "crash_rate": crashes / len(group) if group else "",
            "minimum_altitude_mean_m": (
                sum(altitudes) / len(altitudes) if altitudes else ""
            ),
            "minimum_altitude_worst_m": min(altitudes) if altitudes else "",
        })
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reports)
    print(f"[done] frozen evaluation: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
