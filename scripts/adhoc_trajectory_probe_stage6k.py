"""Step-level trajectory probe for stage6k_lead_pursuit_scale020_50iter, to
diagnose WHY its frozen eval showed minimum_altitude collapsing to
397-618m (worst 397.2m, right at the 400m hard-floor backstop) in every one
of 20 episodes, with the safety override firing far more (1031-2997
sim-ticks/episode) than stage6j's opponent-swap-only checkpoint
(102-1420/episode) -- despite lead_pursuit_scale being altitude-gated
(inactive below altitude_soft_floor_m=1200m) exactly like every other
delta/proximity term in this reward lineage.

Hypothesis under test: target_velocity_step is estimated from a single-step
position difference (target_position(t) - target_position(t-1)) and then
extrapolated lead_pursuit_horizon_steps (20) ahead. ScriptedPursuitActionProvider
banks aggressively to track the ownship (heading_to_bank_gain=1.5, max
+-70deg) and has a bank-induced pitch compensation term
(bank_pitch_compensation_gain), so its OWN vertical rate can genuinely
oscillate step to step as it out-turns the ownship near a close pass -- and
critically, angular sensitivity to any extrapolation error in the predicted
point scales with 1/distance, so the same absolute vertical noise that is
harmless at 2000m can demand large, fast-oscillating pitch commands from the
ownship right at the close-range merge, which is exactly where this project
has repeatedly found the crash/recovery fork happens (see the 2026-08-11
step-level trajectory analysis in project memory).

This runs WITHOUT the safety override (so the raw policy tendency is visible,
not overwritten by forced pull-ups) against ScriptedPursuitActionProvider,
logging every step's distance/altitude/pitch/roll/ata AND the reward
function's own lead_pursuit component (from info["reward_components"]) plus
the target's per-step vertical displacement, to see directly whether sharp
pitch-down moments line up with (a) close range and (b) target vertical
noise/lead_pursuit component swings.

Usage: python scripts/adhoc_trajectory_probe_stage6k.py [--episodes 3]
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

TAG = "altitude_attack_followup_v1_stage6k_lead_pursuit_scale020_50iter_C10"
MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
BASE_SEED = 269000  # matches adhoc_scripted_pursuit_eval_stage6j.py's default --seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    return parser.parse_args()


def run_episode(env, seed: int) -> list[dict]:
    env.reset(seed=seed)
    traj: list[dict] = []
    terminated = truncated = False
    t = 0
    prev_target_d = None
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        target_d = float(env._target_state[StateIndex.D])
        target_vd_step = target_d - prev_target_d if prev_target_d is not None else 0.0
        prev_target_d = target_d
        components = info.get("reward_components", {})
        traj.append({
            "t_s": t / 10.0,
            "distance_m": info.get("final_distance_m"),
            "altitude_m": info.get("final_altitude_m"),
            "roll_deg": info.get("final_roll_deg"),
            "pitch_deg": info.get("final_pitch_deg"),
            "ata_deg": info.get("final_ata_deg"),
            "lead_pursuit_reward": components.get("lead_pursuit", 0.0),
            "target_vd_step_m": target_vd_step,  # target vertical rate this step (NED D, +down)
            "reward": reward,
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

    min_alt_idx = min(range(len(traj)), key=lambda i: traj[i]["altitude_m"])
    min_alt_row = traj[min_alt_idx]
    print(f"  lowest altitude at t={min_alt_row['t_s']:.1f}s: "
          f"alt={min_alt_row['altitude_m']:.0f}m dist={min_alt_row['distance_m']:.1f}m "
          f"pitch={min_alt_row['pitch_deg']:.1f}")

    # Dense window around closest approach: every step for 15s before/after.
    window = [row for row in traj if min_row["t_s"] - 5.0 <= row["t_s"] <= min_row["t_s"] + 15.0]
    print(f"  --- dense window around merge (every 0.5s) ---")
    for row in window[::5]:
        print(f"    t={row['t_s']:6.1f}s dist={row['distance_m']:7.1f}m alt={row['altitude_m']:7.0f}m "
              f"pitch={row['pitch_deg']:6.1f} roll={row['roll_deg']:7.1f} ata={row['ata_deg']:6.1f} "
              f"lead_pursuit_r={row['lead_pursuit_reward']:7.4f} tgt_vD/step={row['target_vd_step_m']:6.2f} "
              f"reward={row['reward']:7.3f}")
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
            summarize(traj)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
