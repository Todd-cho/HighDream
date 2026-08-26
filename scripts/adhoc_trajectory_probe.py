"""Ad-hoc step-level trajectory probe for stage6e_delta_down_400iter, to check
whether crash episodes and "close pass then never returns" episodes share a
common root cause: an attempted hard turn-back at low altitude/low energy
right after the closest approach.

episode_summary.csv only has per-episode aggregates (final/min/mean), not the
time series, so this replays specific seeds (matching known episode numbers
from a prior 20-episode frozen eval at --max-engage-time 200) and logs
distance/altitude/roll/pitch/ata every step, then prints the trajectory
around each episode's point of closest approach.

Usage: python scripts/adhoc_trajectory_probe.py
"""
from __future__ import annotations

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
BASE_SEED = 269000  # matches run_stage6g_frozen_eval.py's default --seed

# episode numbers from the 20-episode --max-engage-time 200 frozen eval,
# grouped by the pattern seen in episode_summary.csv.
GROUPS = {
    "crash (Type A: successful merge -> low-alt trap)": [3, 5, 8, 17],
    "survived but flew away (never returns)": [1, 15, 20],
    "survived, sustained tracking to the end": [4, 12, 13, 18],
}


def run_episode(env, seed: int) -> list[dict]:
    env.reset(seed=seed)
    traj: list[dict] = []
    terminated = truncated = False
    t = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        traj.append({
            "t_s": t / 10.0,  # 10Hz control rate
            "distance_m": info.get("final_distance_m"),
            "altitude_m": info.get("final_altitude_m"),
            "roll_deg": info.get("final_roll_deg"),
            "pitch_deg": info.get("final_pitch_deg"),
            "ata_deg": info.get("final_ata_deg"),
            "outcome": info.get("outcome"),
            "end_condition": info.get("end_condition"),
        })
        t += 1
    return traj


def summarize(traj: list[dict]) -> None:
    min_idx = min(range(len(traj)), key=lambda i: traj[i]["distance_m"])
    min_row = traj[min_idx]
    print(f"  closest approach at t={min_row['t_s']:.1f}s: "
          f"dist={min_row['distance_m']:.1f}m alt={min_row['altitude_m']:.0f}m "
          f"roll={min_row['roll_deg']:.1f} pitch={min_row['pitch_deg']:.1f} ata={min_row['ata_deg']:.1f}")
    # trajectory for 20s after closest approach (or to episode end), every 2s
    window = [row for row in traj if min_row["t_s"] <= row["t_s"] <= min_row["t_s"] + 20.0]
    for row in window[::20]:  # every ~2s at 10Hz
        print(f"    t={row['t_s']:6.1f}s dist={row['distance_m']:8.1f}m alt={row['altitude_m']:7.0f}m "
              f"roll={row['roll_deg']:7.1f} pitch={row['pitch_deg']:6.1f} ata={row['ata_deg']:6.1f}")
    last = traj[-1]
    print(f"  episode end at t={last['t_s']:.1f}s outcome={last['outcome']} "
          f"end_condition={last['end_condition']} alt={last['altitude_m']:.0f}m "
          f"dist={last['distance_m']:.1f}m roll={last['roll_deg']:.1f} ata={last['ata_deg']:.1f}")


def main() -> int:
    reward_module, env_config = load_training_env_config(TAG)
    reward_fn, _ = load_reward_hook(reward_module)
    env_config["max_engage_time"] = 200.0
    env_config["episode_summary_path"] = ""  # no CSV needed, we log ourselves

    bundle = MODEL_ROOT / TAG
    provider = RLActionProvider(
        bundle_dir=bundle,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )
    env = DogFightWrapper(env_config=env_config, ownship_action_provider=provider, reward_fn=reward_fn)
    try:
        for group_name, episode_numbers in GROUPS.items():
            print(f"\n=== {group_name} ===")
            for ep_num in episode_numbers:
                seed = BASE_SEED + ep_num
                print(f" episode #{ep_num} (seed={seed}):")
                traj = run_episode(env, seed)
                summarize(traj)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
