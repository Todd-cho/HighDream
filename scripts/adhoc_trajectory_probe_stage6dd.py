"""Step-level trajectory probe for stage6dd_multiangle4_mix_80iter, to
diagnose WHY extending stage6cc's 4-angle-mix recipe from 50iter to 80iter
eroded safety margin so badly: frozen 20-episode eval minimum_altitude_mean
dropped from 3657.6m (stage6cc, 50iter) to 1254.0m, with the single worst
episode (episode #2, seed 269002, scenario multiangle_15.0deg_750) hitting
minimum_altitude_m=338.229m -- BELOW safety_override_hard_floor_m (400m) --
while wez_episode_rate rose from 5% to 20% and mean_distance blew up from
3866.9m to 17669.3m. safety_override_steps=3534 for that episode (heavy,
sustained intervention, not a brief one-off).

This runs WITHOUT the safety override (so the raw policy tendency is
visible, not overwritten by forced pull-ups) against
ScriptedPursuitActionProvider using stage6dd's own eased opponent config
and 4-angle scenario_pool (env.reset(seed=...) samples the pool, so which
angle a given episode gets is deterministic per seed -- episode #2/seed
269002 is confirmed to draw the 15deg scenario from stage6dd's own frozen
eval CSV), logging distance/altitude/roll/pitch/ata/action every step
across the FULL episode to see directly: is this the same "aggressive bank
-> slow altitude erosion over tens of seconds" mechanism found in stage6k's
lead_pursuit trajectory probe (2026-08-12), or a different failure mode
specific to the multi-angle/higher-iteration combination?

Usage: python scripts/adhoc_trajectory_probe_stage6dd.py [--episodes 3]
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
from dogfight.sim.state_schema import StateIndex
from scripts.run_stage6g_frozen_eval import load_training_env_config

TAG = "altitude_attack_followup_v1_stage6dd_multiangle4_mix_80iter_C10"
MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
BASE_SEED = 269000  # matches adhoc_scripted_pursuit_eval_stage6j.py's default --seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--sample-every-s", type=float, default=10.0)
    return parser.parse_args()


def run_episode(env, seed: int) -> list[dict]:
    _, reset_info = env.reset(seed=seed)
    print(f"  scenario: {reset_info.get('initial_scenario_name')}")
    traj: list[dict] = []
    terminated = truncated = False
    t = 0
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        traj.append({
            "t_s": t / 10.0,
            "distance_m": info.get("final_distance_m"),
            "altitude_m": info.get("final_altitude_m"),
            "roll_deg": info.get("final_roll_deg"),
            "pitch_deg": info.get("final_pitch_deg"),
            "ata_deg": info.get("final_ata_deg"),
            "aa_deg": info.get("final_aa_deg"),
            "reward": reward,
            "outcome": info.get("outcome"),
            "end_condition": info.get("end_condition"),
        })
        t += 1
    return traj


def summarize(traj: list[dict], sample_every_s: float) -> None:
    first = traj[0]
    print(f"  t=0: dist={first['distance_m']:.1f}m alt={first['altitude_m']:.0f}m "
          f"ata={first['ata_deg']:.1f} aa={first['aa_deg']:.1f} roll={first['roll_deg']:.1f}")

    min_idx = min(range(len(traj)), key=lambda i: traj[i]["distance_m"])
    min_row = traj[min_idx]
    print(f"  closest approach at t={min_row['t_s']:.1f}s: "
          f"dist={min_row['distance_m']:.1f}m alt={min_row['altitude_m']:.0f}m "
          f"roll={min_row['roll_deg']:.1f} pitch={min_row['pitch_deg']:.1f} ata={min_row['ata_deg']:.1f}")

    step = max(1, int(sample_every_s * 10))
    print(f"  --- full-episode trajectory (every {sample_every_s:.0f}s) ---")
    for row in traj[::step]:
        print(f"    t={row['t_s']:6.1f}s dist={row['distance_m']:8.1f}m alt={row['altitude_m']:7.0f}m "
              f"pitch={row['pitch_deg']:6.1f} roll={row['roll_deg']:7.1f} ata={row['ata_deg']:6.1f} "
              f"aa={row['aa_deg']:6.1f} reward={row['reward']:7.3f}")
    last = traj[-1]
    print(f"  episode end at t={last['t_s']:.1f}s outcome={last['outcome']} "
          f"end_condition={last['end_condition']} alt={last['altitude_m']:.0f}m "
          f"dist={last['distance_m']:.1f}m")


def main() -> int:
    args = parse_args()
    reward_module, env_config = load_training_env_config(TAG)
    reward_fn, _ = load_reward_hook(reward_module)
    env_config["max_engage_time"] = args.max_engage_time
    env_config["episode_summary_path"] = ""  # no CSV needed, we log ourselves
    # Deliberately NOT enabling safety_override here: we want to see the raw
    # policy tendency, not a trajectory dominated by forced pull-ups.

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}
    print(f"[probe] target_scripted_pursuit config: {pursuit_cfg}")

    bundle = MODEL_ROOT / TAG
    ownship_provider = RLActionProvider(
        bundle_dir=bundle,
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
        for ep_num in range(1, args.episodes + 1):
            seed = BASE_SEED + ep_num
            print(f"\n=== episode #{ep_num} (seed={seed}) ===")
            traj = run_episode(env, seed)
            summarize(traj, args.sample_every_s)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
