"""Frozen (deterministic) JSBSim evaluation of RLRuleHandoffActionProvider
(src/dogfight/ai/rl_rule_handoff.py) -- the RL<->rule-controller handoff that
student/my_submission_rl_w112_handoff.py uses live, but had never been run
through the fast local JSBSim harness before (only live-tested).

Rationale (2026-08-25): tactical_wrapper.py's own from-scratch rule engine
was live-verified to never converge ATA from a 91deg start (see
rl_rule_handoff.py's module docstring). The handoff instead falls back to
a rule controller from the W-series (run_unreal_inference.py --mode wNNN),
which HAS been live-flown repeatedly, and only lets the RL policy fly once
ATA is inside its trained comfort zone. This script lets that combination
be A/B'd against the plain tactical_wrapper (and the raw RL policy) offline,
at the real competition max_engage_time (200s), before spending another live
session on it.

Same scenario/reward/opponent config as adhoc_tactical_wrapper_eval.py
(loaded from the tag's own training YAML), same safety override.

Usage:
  python scripts/adhoc_rl_rule_handoff_eval.py --tag <rl_tag> [--rule-mode w112] [--episodes 30]
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
from dogfight.ai.rl_rule_handoff import RLRuleHandoffActionProvider
from dogfight.ai.scripted_pursuit_provider import ScriptedPursuitActionProvider
from dogfight.ai.student_hooks import load_reward_hook
from scripts.run_stage6g_frozen_eval import load_training_env_config

import run_unreal_inference as rui

MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
FROZEN_EVAL_ROOT = ROOT / "artifacts" / "altitude_attack_followup_v1" / "frozen_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="RL bundle tag, e.g. altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10")
    parser.add_argument("--rule-mode", default="w112", help="run_unreal_inference.py --mode value to use as the rule-controller fallback.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-engage-time", type=float, default=200.0, help="200.0 = real competition value.")
    parser.add_argument("--seed", type=int, default=269000)
    parser.add_argument("--enter-rl-ata-deg", type=float, default=20.0)
    parser.add_argument("--exit-rl-ata-deg", type=float, default=28.0)
    return parser.parse_args()


def build_rule_provider(rule_mode: str):
    saved_argv = sys.argv
    try:
        sys.argv = [
            "run_unreal_inference.py",
            "--mode", rule_mode,
            "--team-name", "eval",
            "--server-ip", "127.0.0.1",
        ]
        rui_args = rui.parse_args()
    finally:
        sys.argv = saved_argv
    return rui.build_action_provider(rui_args)


def main() -> int:
    args = parse_args()
    tag = args.tag
    suffix = f"handoff_{args.rule_mode}"
    out_dir = FROZEN_EVAL_ROOT / f"{tag}__scripted_pursuit__{suffix}"
    reward_module, env_config = load_training_env_config(tag)
    reward_fn, _ = load_reward_hook(reward_module)

    env_config["max_engage_time"] = args.max_engage_time
    env_config["safety_override_enabled"] = True
    env_config["safety_override_altitude_m"] = env_config.get("safety_override_altitude_m", 1500.0)
    env_config["safety_override_time_horizon_s"] = env_config.get("safety_override_time_horizon_s", 25.0)

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("episode_summary*.csv"):
        stale.unlink()
    env_config["episode_summary_path"] = str(out_dir / "episode_summary.csv")

    rl_provider = RLActionProvider(
        bundle_dir=MODEL_ROOT / tag,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )
    print(f"[handoff-eval] building rule provider --mode {args.rule_mode}")
    rule_provider = build_rule_provider(args.rule_mode)
    ownship_provider = RLRuleHandoffActionProvider(
        rl_provider, rule_provider,
        enter_rl_ata_deg=args.enter_rl_ata_deg,
        exit_rl_ata_deg=args.exit_rl_ata_deg,
    )
    target_provider = ScriptedPursuitActionProvider(**pursuit_cfg)
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=ownship_provider,
        target_action_provider=target_provider,
        reward_fn=reward_fn,
    )
    mode_counts: dict[str, int] = {}
    try:
        for episode_number in range(1, args.episodes + 1):
            env.reset(seed=args.seed + episode_number)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
                mode_counts[ownship_provider._using_rl] = mode_counts.get(ownship_provider._using_rl, 0) + 1
            print(f"[handoff-eval] episode {episode_number}/{args.episodes}")
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
    total_steps = sum(mode_counts.values())
    if total_steps:
        rl_frac = mode_counts.get(True, 0) / total_steps
        print(f"  rl_step_fraction={rl_frac:.3f} rule_step_fraction={1 - rl_frac:.3f}")
    print()
    for i, row in enumerate(rows, 1):
        print(i, row.get("scenario_name"), row.get("outcome"), row.get("end_condition"),
              row.get("minimum_altitude_m"), row.get("min_distance_m"), row.get("mean_distance_m"),
              row.get("wez_steps"), row.get("target_health"), row.get("ownship_health"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
