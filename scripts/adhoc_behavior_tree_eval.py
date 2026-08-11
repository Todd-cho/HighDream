"""Ad-hoc eval of stage6e_delta_down_400iter (+ safety override) against an
active behavior_tree opponent, using the DEFAULT_ENV_CONFIG scenario (plain
ownship/target start vectors, no scenario_pool) instead of close_quarters_750.

Background: overriding just target_mode="behavior_tree" while keeping the
close_quarters_750 scenario_pool caused the TARGET to crash itself in 20/20
episodes within ~75-105s ("target altitude below min"), regardless of what
the ownship did (min_distance never below ~686m, wez_steps always 0, draw
outcome every time) -- that scenario's geometry/target_loiter config was
tuned for a passive loiter target and is evidently incompatible with the
active AIP_BASE_target.dll behavior tree. src/dogfight/ai/curriculum.py's
final "full_dogfight" stage uses target_mode="behavior_tree" with the plain
DEFAULT_ENV_CONFIG scenario (initial_scenario.mode="default", no
scenario_pool) -- that combination is known-used elsewhere in this project,
so this script uses it to get a valid read on how the policy handles a
genuinely active/evasive opponent, separate from the close_quarters_750
scenario mismatch.

Usage: python scripts/adhoc_behavior_tree_eval.py [--episodes 20] [--max-engage-time 200]
"""
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
from scripts.run_stage6g_frozen_eval import load_training_env_config

TAG = "altitude_attack_followup_v1_stage6e_delta_down_400iter_C10"
MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
OUT_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1" / "frozen_eval" / f"{TAG}__targetbehavior_tree_default_scenario__safety"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=269000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reward_module, env_config = load_training_env_config(TAG)
    reward_fn, _ = load_reward_hook(reward_module)

    env_config["max_engage_time"] = args.max_engage_time
    env_config["target_mode"] = "behavior_tree"
    env_config["initial_scenario"] = {"mode": "default"}  # plain DEFAULT_ENV_CONFIG ownship/target vectors
    env_config["safety_override_enabled"] = True
    env_config["safety_override_altitude_m"] = 1500.0
    env_config["safety_override_time_horizon_s"] = 25.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_config["episode_summary_path"] = str(OUT_DIR / "episode_summary.csv")

    provider = RLActionProvider(
        bundle_dir=MODEL_ROOT / TAG,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )
    env = DogFightWrapper(env_config=env_config, ownship_action_provider=provider, reward_fn=reward_fn)
    try:
        for episode_number in range(1, args.episodes + 1):
            env.reset(seed=args.seed + episode_number)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            print(f"[bt-eval] episode {episode_number}/{args.episodes}")
    finally:
        env.close()

    paths = sorted(OUT_DIR.glob("episode_summary*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    def num(row: dict[str, str], key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    crashes = sum(row.get("outcome") == "crash" for row in rows)
    wins = sum(row.get("outcome") == "win" for row in rows)
    losses = sum(row.get("outcome") == "loss" for row in rows)
    draws = sum(row.get("outcome") == "draw" for row in rows)
    altitudes = [v for row in rows if (v := num(row, "minimum_altitude_m")) is not None]
    distances = [v for row in rows if (v := num(row, "mean_distance_m")) is not None]
    wez_steps = [v for row in rows if (v := num(row, "wez_steps")) is not None]

    print(f"\n[done] episodes={len(rows)} crash={crashes} win={wins} loss={losses} draw={draws}")
    print(f"  minimum_altitude_mean_m={sum(altitudes)/len(altitudes) if altitudes else None}")
    print(f"  minimum_altitude_worst_m={min(altitudes) if altitudes else None}")
    print(f"  mean_distance_m={sum(distances)/len(distances) if distances else None}")
    print(f"  wez_episode_rate={sum(v > 0.0 for v in wez_steps)/len(wez_steps) if wez_steps else None}")
    for i, row in enumerate(rows, 1):
        print(i, row.get("outcome"), row.get("end_condition"), row.get("steps"),
              row.get("minimum_altitude_m"), row.get("min_distance_m"),
              row.get("wez_steps"), row.get("target_health"), row.get("ownship_health"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
