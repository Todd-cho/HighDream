"""Frozen (deterministic, no exploration noise) evaluation of
stage6j_scripted_pursuit_opponent_50iter -- the first checkpoint ever
trained (not just evaluated) against ScriptedPursuitActionProvider, a
genuinely active/closing opponent -- against that same active opponent.

Background: this run's own training-exploration numbers (printed by
scripts/run_stage6j_scripted_pursuit_pilot.py) showed episode_crash_rate=0.8
(4/5) and wez_episode_rate=0.0 over its 50-iteration window. Per rule 15
(training-exploration episodes have repeatedly given a false read this
project, in both directions), that number is not trustworthy on its own --
small n (5), taken entirely within the post-restore replay-buffer-refill
window (rule 14), and with SAC's exploration noise added to actions, all of
which can make True performance look worse (or better) than it is. This
freezes the policy (RLActionProvider explore=False) and reruns it on the
exact scenario/reward config it trained under, against the same
ScriptedPursuitActionProvider (cruise_altitude_m read from the tag's own
training config, not a mismatched hardcoded default), with the hard safety
override enabled (matches how the current official-best candidate,
stage6e_delta_down_400iter+safety override, is actually being evaluated/
deployed), at the real regulation max_engage_time=200s.

Usage: python scripts/adhoc_scripted_pursuit_eval_stage6j.py [--episodes 20] [--max-engage-time 200]
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
from dogfight.ai.scripted_pursuit_provider import ScriptedPursuitActionProvider
from dogfight.ai.student_hooks import load_reward_hook
from scripts.run_stage6g_frozen_eval import load_training_env_config

DEFAULT_TAG = "altitude_attack_followup_v1_stage6j_scripted_pursuit_opponent_50iter_C10"
MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
FROZEN_EVAL_ROOT = ROOT / "artifacts" / "altitude_attack_followup_v1" / "frozen_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", default=DEFAULT_TAG,
        help="Output tag of the checkpoint to evaluate (2026-08-12: generalized "
             "from a hardcoded stage6j-only TAG so this script is reusable "
             "across future scripted_pursuit-trained variants, e.g. stage6k).",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=269000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag
    out_dir = FROZEN_EVAL_ROOT / f"{tag}__scripted_pursuit__safety"
    reward_module, env_config = load_training_env_config(tag)
    reward_fn, _ = load_reward_hook(reward_module)

    env_config["max_engage_time"] = args.max_engage_time
    env_config["safety_override_enabled"] = True
    env_config["safety_override_altitude_m"] = 1500.0
    env_config["safety_override_time_horizon_s"] = 25.0

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    env_config["episode_summary_path"] = str(out_dir / "episode_summary.csv")

    ownship_provider = RLActionProvider(
        bundle_dir=MODEL_ROOT / tag,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )
    target_provider = ScriptedPursuitActionProvider(**pursuit_cfg)
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=ownship_provider,
        target_action_provider=target_provider,
        reward_fn=reward_fn,
    )
    try:
        for episode_number in range(1, args.episodes + 1):
            env.reset(seed=args.seed + episode_number)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            print(f"[pursuit-eval] episode {episode_number}/{args.episodes}")
    finally:
        env.close()

    paths = sorted(out_dir.glob("episode_summary*.csv"))
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
    min_distances = [v for row in rows if (v := num(row, "min_distance_m")) is not None]
    wez_steps = [v for row in rows if (v := num(row, "wez_steps")) is not None]
    target_healths = [v for row in rows if (v := num(row, "target_health")) is not None]
    override_steps = [v for row in rows if (v := num(row, "safety_override_steps")) is not None]

    print(f"\n[done] episodes={len(rows)} crash={crashes} win={wins} loss={losses} draw={draws}")
    print(f"  minimum_altitude_mean_m={sum(altitudes)/len(altitudes) if altitudes else None:.1f}")
    print(f"  minimum_altitude_worst_m={min(altitudes) if altitudes else None}")
    print(f"  mean_distance_m={sum(distances)/len(distances) if distances else None:.1f}")
    print(f"  min_distance_mean_m={sum(min_distances)/len(min_distances) if min_distances else None:.1f}")
    print(f"  wez_episode_rate={sum(v > 0.0 for v in wez_steps)/len(wez_steps) if wez_steps else None}")
    print(f"  mean_target_damage={1 - sum(target_healths)/len(target_healths) if target_healths else None}")
    print(f"  safety_override_episode_rate={sum(v > 0.0 for v in override_steps)/len(override_steps) if override_steps else None}")
    print()
    for i, row in enumerate(rows, 1):
        print(i, row.get("outcome"), row.get("end_condition"), row.get("steps"),
              row.get("minimum_altitude_m"), row.get("min_distance_m"), row.get("mean_distance_m"),
              row.get("wez_steps"), row.get("target_health"), row.get("ownship_health"),
              row.get("safety_override_steps"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
