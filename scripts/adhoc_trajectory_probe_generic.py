"""Generic step-level trajectory probe, parameterized by --tag and
--episode-numbers, for the stage6gg/6hh/6ii/6jj stability-only single-angle
runs (2026-08-18). Generalizes adhoc_trajectory_probe_stage6dd.py so the
same tool can target specific crash episodes across multiple tags without a
new script per tag.

Background: stage6hh(8deg)/6ii(15deg)/6jj(22deg) -- pure stability reward
(all approach/engagement terms zeroed) -- still crashed in frozen eval
(25%/10%/15% respectively) even with the hard-coded safety override
enabled. Per-episode CSV inspection (csv.DictReader, respecting the
quoted `suggested_params` field that breaks naive comma-splitting) shows
every crash happens LATE in the 200s episode (88-193s), with
safety_override_steps in the 2000-3900 range -- i.e. the override was
fighting continuously for most of the episode before eventually losing,
not a one-off reaction at the initial merge. This probe runs WITHOUT the
safety override (raw policy tendency) on the exact same seeds that
crashed in the safety-override-on frozen eval, to see directly what the
policy is actually doing throughout the whole 200s, especially in the
run-up to the eventual dive.

Usage: python scripts/adhoc_trajectory_probe_generic.py --tag <output_tag> --episode-numbers 1,11,15
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
    parser.add_argument("--tag", required=True)
    parser.add_argument("--episode-numbers", required=True,
                         help="Comma-separated episode numbers matching the frozen eval CSV "
                              "(seed = 269000 + episode_number).")
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

    # Also zoom in on the final 30s, since safety_override intensity in the
    # source CSV suggests the real story is in the run-up to the crash, not
    # the initial merge.
    tail_start = max(0.0, last["t_s"] - 30.0)
    print(f"  --- final 30s zoom (every 2s), t>={tail_start:.1f}s ---")
    tail_step = max(1, int(2.0 * 10))
    for row in traj[::tail_step]:
        if row["t_s"] >= tail_start:
            print(f"    t={row['t_s']:6.1f}s dist={row['distance_m']:8.1f}m alt={row['altitude_m']:7.0f}m "
                  f"pitch={row['pitch_deg']:6.1f} roll={row['roll_deg']:7.1f} ata={row['ata_deg']:6.1f} "
                  f"aa={row['aa_deg']:6.1f} reward={row['reward']:7.3f}")


def main() -> int:
    args = parse_args()
    tag = args.tag
    reward_module, env_config = load_training_env_config(tag)
    reward_fn, _ = load_reward_hook(reward_module)
    env_config["max_engage_time"] = args.max_engage_time
    env_config["episode_summary_path"] = ""  # no CSV needed, we log ourselves
    # Deliberately NOT enabling safety_override here: we want to see the raw
    # policy tendency, not a trajectory dominated by forced pull-ups.

    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}
    print(f"[probe] tag={tag}")
    print(f"[probe] target_scripted_pursuit config: {pursuit_cfg}")

    bundle = MODEL_ROOT / tag
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
    episode_numbers = [int(x) for x in args.episode_numbers.split(",")]
    try:
        for ep_num in episode_numbers:
            seed = BASE_SEED + ep_num
            print(f"\n=== episode #{ep_num} (seed={seed}) ===")
            traj = run_episode(env, seed)
            summarize(traj, args.sample_every_s)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
