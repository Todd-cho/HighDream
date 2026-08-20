"""Frozen (deterministic) evaluation of a checkpoint wrapped by
TacticalWrapperActionProvider (src/dogfight/ai/tactical_wrapper.py) --
the no-retrain supervisory layer from the 2026-08-20 design doc
(주최조건_1대1_계층형전술_v3_v4_설계전략.txt, section 4.2): blends the
existing SAC policy with a rule-based lead-pursuit controller based on a
Neutral/Offensive/Overshoot hysteresis state machine, so high-ATA starts
(the 91deg competition condition) get handled by rules instead of by a
policy that never trained on them.

Same scenario/reward/opponent config as adhoc_scripted_pursuit_eval_stage6j.py
(loaded from the tag's own training YAML), same safety override -- only
difference is the ownship provider is wrapped. Pass --plain to run the same
tag WITHOUT the wrapper for a same-script A/B baseline.

Usage: python scripts/adhoc_tactical_wrapper_eval.py --tag <tag> [--episodes 30] [--plain]
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
from dogfight.ai.tactical_wrapper import TacticalWrapperActionProvider, TacticalWrapperConfig
from scripts.run_stage6g_frozen_eval import load_training_env_config

MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
FROZEN_EVAL_ROOT = ROOT / "artifacts" / "altitude_attack_followup_v1" / "frozen_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=269000)
    parser.add_argument("--plain", action="store_true", help="Skip the tactical wrapper (A/B baseline).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag
    suffix = "plain" if args.plain else "tactical_wrapper"
    out_dir = FROZEN_EVAL_ROOT / f"{tag}__scripted_pursuit__{suffix}"
    reward_module, env_config = load_training_env_config(tag)
    reward_fn, _ = load_reward_hook(reward_module)

    env_config["max_engage_time"] = args.max_engage_time
    env_config["safety_override_enabled"] = True
    env_config["safety_override_altitude_m"] = env_config.get("safety_override_altitude_m", 1500.0)
    env_config["safety_override_time_horizon_s"] = env_config.get("safety_override_time_horizon_s", 25.0)

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    env_config["episode_summary_path"] = str(out_dir / "episode_summary.csv")

    rl_provider = RLActionProvider(
        bundle_dir=MODEL_ROOT / tag,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )
    ownship_provider = rl_provider if args.plain else TacticalWrapperActionProvider(
        rl_provider, TacticalWrapperConfig()
    )
    target_provider = ScriptedPursuitActionProvider(**pursuit_cfg)
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=ownship_provider,
        target_action_provider=target_provider,
        reward_fn=reward_fn,
    )
    state_counts: dict[str, int] = {}
    try:
        for episode_number in range(1, args.episodes + 1):
            env.reset(seed=args.seed + episode_number)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            print(f"[tactical-wrapper-eval] episode {episode_number}/{args.episodes}")
            if isinstance(ownship_provider, TacticalWrapperActionProvider):
                for s in ownship_provider.state_log:
                    state_counts[s] = state_counts.get(s, 0) + 1
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

    print(f"\n[done] mode={suffix} episodes={len(rows)} crash={crashes} win={wins} loss={losses} draw={draws}")
    print(f"  minimum_altitude_mean_m={sum(altitudes)/len(altitudes) if altitudes else None:.1f}")
    print(f"  minimum_altitude_worst_m={min(altitudes) if altitudes else None}")
    print(f"  mean_distance_m={sum(distances)/len(distances) if distances else None:.1f}")
    print(f"  min_distance_mean_m={sum(min_distances)/len(min_distances) if min_distances else None:.1f}")
    print(f"  wez_episode_rate={sum(v > 0.0 for v in wez_steps)/len(wez_steps) if wez_steps else None}")
    print(f"  mean_target_damage={1 - sum(target_healths)/len(target_healths) if target_healths else None}")
    if state_counts:
        total = sum(state_counts.values())
        print(f"  tactical state distribution: " + ", ".join(f"{k}={v/total*100:.1f}%" for k, v in state_counts.items()))
    print()
    for i, row in enumerate(rows, 1):
        print(i, row.get("scenario_name"), row.get("outcome"), row.get("end_condition"),
              row.get("minimum_altitude_m"), row.get("min_distance_m"), row.get("mean_distance_m"),
              row.get("wez_steps"), row.get("target_health"), row.get("ownship_health"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
