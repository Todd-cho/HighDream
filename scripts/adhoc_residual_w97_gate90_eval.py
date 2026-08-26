"""Frozen (deterministic) evaluation of residual_w97_gate90_120iter_seed02
against the real opponent (scripted_pursuit, opp055 tuning) at real
regulation settings (200s engage time, safety override on) -- same protocol
as adhoc_scripted_pursuit_eval_stage6j.py, so the outcome numbers are
directly comparable to stage6obs19_v8's 90.1% (n=101) win rate.

The training run itself (experiments/residual_w97_gate90_120iter_seed02.yaml)
used opponent_pool + 5 varied acquisition-geometry scenarios, not the plain
0/91deg pair -- this eval deliberately narrows back down to the actual
competition-relevant opponent/geometry to answer "how does it do against
what we'll actually face", not "how does it do on its own training mix".

Usage: python scripts/adhoc_residual_w97_gate90_eval.py [--bundle bundle_000090] [--episodes 20]
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
from dogfight.ai.rule_profiles import build_w97_controller
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.scripted_pursuit_provider import ScriptedPursuitActionProvider
from dogfight.ai.student_hooks import load_reward_hook
from dogfight.ai.w56_residual_action_provider import W56ResidualActionProvider

TAG = "residual_w97_gate90_120iter_seed02"
MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream" / TAG
FROZEN_EVAL_ROOT = ROOT / "artifacts" / "altitude_attack_followup_v1" / "frozen_eval"

# Same 0/91deg-equivalent pair + opp055 opponent tuning v8 was evaluated
# against, for a direct comparison.
SCENARIOS = [
    {
        "name": "scale1.00_00.0deg_750",
        "ownship": [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0],
        "target": [1749.5331880577405, 0.0, -4500.0, 0.0, 0.0, 90.0, 250.0],
    },
    {
        "name": "scale1.00_90.0deg_750",
        "ownship": [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0],
        "target": [1000.0, 749.5331880577404, -4500.0, 0.0, 0.0, 90.0, 250.0],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="bundle_000090")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=269100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = MODEL_ROOT / args.bundle
    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle not found: {bundle_dir}")

    out_dir = FROZEN_EVAL_ROOT / f"{TAG}__{args.bundle}__scripted_pursuit__safety"
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_fn, _ = load_reward_hook("student.my_reward_residual_w97_v2")

    env_config = {
        "observation_mode": "tactical19",
        "target_mode": "scripted_pursuit",
        "target_behavior_dll": "AIP_BASE_target.dll",
        "ownship_control_mode": "rl",
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": 3600,
        "step_ratio": 6,
        "safety_override_enabled": True,
        "safety_override_altitude_m": 1500.0,
        "safety_override_time_horizon_s": 25.0,
        "target_scripted_pursuit": {
            "cruise_altitude_m": 4500.0,
            "heading_to_bank_gain": 0.55,
            "max_bank_deg": 32.0,
        },
        "initial_scenario": {
            "mode": "scenario_pool",
            "scenarios": [{**s, "weight": 1.0} for s in SCENARIOS],
            "ownship_randomization": {
                "enabled": True, "radius": 40.0, "r_roll": 2.0,
                "r_pitch": 1.0, "r_heading": 5.0, "speed_mps": 3.0,
            },
            "target_randomization": {
                "enabled": True, "radius": 40.0, "r_heading": 5.0, "speed_mps": 2.0,
            },
        },
        "episode_summary_path": str(out_dir / "episode_summary.csv"),
    }

    ownship_provider = W56ResidualActionProvider(
        bundle_dir=bundle_dir,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        roll_scale=0.20,
        pitch_scale=0.20,
        throttle_scale=0.10,
        gate_ata_deg=90.0,
        gate_range_m=3500.0,
        gate_min_threat_ata_deg=10.0,
        force_zero_residual=False,
        rule_provider=build_w97_controller(),
    )
    target_provider = ScriptedPursuitActionProvider(**env_config["target_scripted_pursuit"])
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
            print(f"[w97-residual-eval] episode {episode_number}/{args.episodes}")
    finally:
        env.close()

    paths = sorted(out_dir.glob("episode_summary*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    crashes = sum(row.get("outcome") == "crash" for row in rows)
    wins = sum(row.get("outcome") == "win" for row in rows)
    losses = sum(row.get("outcome") == "loss" for row in rows)
    draws = sum(row.get("outcome") == "draw" for row in rows)
    n = len(rows) or 1
    print(f"\n[done] episodes={len(rows)} crash={crashes} win={wins} loss={losses} draw={draws}")
    print(f"  win_rate={wins/n:.3f} crash_rate={crashes/n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
