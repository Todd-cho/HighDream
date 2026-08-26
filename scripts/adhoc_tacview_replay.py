"""Replay one frozen-eval episode against ScriptedPursuitActionProvider and
save a Tacview-importable CSV log (ownship + target lat/lon/alt/roll/pitch/
yaw/health per step), so the merge moment can be watched visually instead of
read as numbers.

Background (2026-08-20): rr_phasewez_100iter / ss_safetytuned_100iter /
tt_atadelta_boost_100iter / uu_fastepisode_100iter all produced byte-identical
wez_episode_rate (0.05) and mean_target_damage (0.0001874499999999779) in
their frozen eval -- traced to the exact same single episode (#18, the 8deg
scenario, seed=269018) scoring exactly one Phase-1 WEZ step in all four,
despite four different reward changes meant to affect exactly this. This
script re-plays that episode (or any other tag/episode) so it can be opened
in Tacview and watched, instead of inferred from CSV columns.

Usage:
  python scripts/adhoc_tacview_replay.py --tag altitude_attack_followup_v1_stage6uu_fastepisode_100iter_C10 --episode-number 18
  python scripts/adhoc_tacview_replay.py --tag <tag> --episode-number 3 --no-safety-override
"""
from __future__ import annotations

import argparse
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

MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
BASE_SEED = 269000  # matches adhoc_scripted_pursuit_eval_stage6j.py's default --seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Output tag of the checkpoint to replay.")
    parser.add_argument(
        "--episode-number", type=int, default=18,
        help="Episode number matching the frozen eval CSV (seed = 269000 + episode_number). "
             "Default 18 is the WEZ-hit episode shared by rr/ss/tt/uu.",
    )
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument(
        "--no-safety-override", action="store_true",
        help="Disable the hard safety override to see the raw policy tendency "
             "instead of the deployed/gated behavior.",
    )
    parser.add_argument("--safety-override-altitude-m", type=float, default=None)
    parser.add_argument("--safety-override-time-horizon-s", type=float, default=None)
    parser.add_argument(
        "--artifacts-dir", default=None,
        help="Override where the Tacview CSV pair is written (default: artifacts/logs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag
    reward_module, env_config = load_training_env_config(tag)
    reward_fn, _ = load_reward_hook(reward_module)

    env_config["max_engage_time"] = args.max_engage_time
    if args.artifacts_dir:
        env_config["artifacts_dir"] = args.artifacts_dir

    if args.no_safety_override:
        env_config["safety_override_enabled"] = False
    else:
        env_config["safety_override_enabled"] = True
        env_config["safety_override_altitude_m"] = (
            args.safety_override_altitude_m
            if args.safety_override_altitude_m is not None
            else env_config.get("safety_override_altitude_m", 1500.0)
        )
        env_config["safety_override_time_horizon_s"] = (
            args.safety_override_time_horizon_s
            if args.safety_override_time_horizon_s is not None
            else env_config.get("safety_override_time_horizon_s", 25.0)
        )

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}
    env_config["episode_summary_path"] = ""  # no CSV needed for a single replay

    seed = BASE_SEED + args.episode_number
    print(f"[replay] tag={tag}")
    print(f"[replay] episode_number={args.episode_number} seed={seed}")
    print(f"[replay] safety_override_enabled={env_config.get('safety_override_enabled')}")

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
        _, reset_info = env.reset(seed=seed)
        print(f"[replay] scenario: {reset_info.get('initial_scenario_name')}")
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        print(f"[replay] outcome={info.get('outcome')} end_condition={info.get('end_condition')}")
        print(f"[replay] target_health={info.get('target_health')} ownship_health={info.get('ownship_health')}")
        env.make_tacviewLog()
        artifacts_dir = env_config.get("artifacts_dir", "artifacts/logs")
        print(f"[done] Tacview-importable CSV pair (+ summary.json) written under: {artifacts_dir}")
        print("       Import both the *_ownship_*.csv and *_target_*.csv into Tacview "
              "(File > Import > CSV, map Longitude/Latitude/Altitude/Roll/Pitch/Yaw) to watch the replay.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
