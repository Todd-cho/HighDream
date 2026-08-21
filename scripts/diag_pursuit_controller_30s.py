"""Step 2 of the P0-P3 ablation protocol (2026-08-21, user's plan after
pursuit_controller.py's JSBSim eval showed mean_distance=16.1km / wez_rate=
16.7% -- averages alone don't say WHY it never closes). Runs a single
deterministic episode against V8 (altitude_attack_followup_v1_
stage6obs19_v8_angle090_opp055_50iter_C10) with a given
PursuitControllerConfig (P0 baseline or a P1/P2/P3 ablation) and dumps
PursuitControllerActionProvider.info_log's first N seconds of sim_time to
CSV + a compact printed table: ata, aa, distance, los_az/el, target_bank,
current_bank, bank_error, alignment, pitch_raw, own_speed, own_alt.

Usage: python scripts/diag_pursuit_controller_30s.py [--seconds 30]
       [--pc-disable-alignment-gate] [--pc-disable-pitch-envelope]
       [--pc-flip-bank-sign] [--out artifacts/logs/pc_diag_p0.csv]
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
from dogfight.ai.pursuit_controller import PursuitControllerActionProvider, PursuitControllerConfig
from scripts.run_stage6g_frozen_eval import load_training_env_config

MODEL_ROOT = ROOT / "artifacts" / "models" / "highdream"
TAG = "altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10"

COLUMNS = [
    "sim_time", "state", "ata", "aa", "distance", "closure_rate",
    "los_az", "los_el", "target_bank", "current_bank", "bank_error",
    "alignment", "pitch_raw", "turn_pull_scale", "own_speed", "own_alt", "own_pitch",
    "roll_cmd", "pitch_cmd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=269001)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--pc-disable-alignment-gate", action="store_true")
    parser.add_argument("--pc-disable-pitch-envelope", action="store_true")
    parser.add_argument("--pc-flip-bank-sign", action="store_true")
    parser.add_argument("--pc-world-frame-pitch-gate", action="store_true")
    parser.add_argument("--pc-turn-pull-decomposition", action="store_true")
    parser.add_argument("--pc-vertical-damping-gain", type=float, default=None)
    parser.add_argument("--pc-turn-pull-priority-gate", action="store_true")
    parser.add_argument("--pc-turn-sign-flip-deg", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reward_module, env_config = load_training_env_config(TAG)
    reward_fn, _ = load_reward_hook(reward_module)

    env_config["max_engage_time"] = 200.0
    env_config["safety_override_enabled"] = True
    env_config["safety_override_altitude_m"] = env_config.get("safety_override_altitude_m", 1500.0)
    env_config["safety_override_time_horizon_s"] = env_config.get("safety_override_time_horizon_s", 25.0)
    pursuit_cfg = env_config.get("target_scripted_pursuit", {}) or {}
    env_config.pop("episode_summary_path", None)

    rl_provider = RLActionProvider(
        bundle_dir=MODEL_ROOT / TAG,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy",
        explore=False,
    )
    pc_kwargs = dict(
        disable_alignment_gate=args.pc_disable_alignment_gate,
        disable_pitch_envelope=args.pc_disable_pitch_envelope,
        flip_bank_sign=args.pc_flip_bank_sign,
        world_frame_pitch_gate=args.pc_world_frame_pitch_gate,
        turn_pull_decomposition=args.pc_turn_pull_decomposition,
    )
    if args.pc_vertical_damping_gain is not None:
        pc_kwargs["vertical_damping_gain"] = args.pc_vertical_damping_gain
    pc_kwargs["turn_pull_priority_gate"] = args.pc_turn_pull_priority_gate
    if args.pc_turn_sign_flip_deg is not None:
        pc_kwargs["turn_sign_flip_deg"] = args.pc_turn_sign_flip_deg
    pc_config = PursuitControllerConfig(**pc_kwargs)
    ownship_provider = PursuitControllerActionProvider(rl_provider, pc_config)
    target_provider = ScriptedPursuitActionProvider(**pursuit_cfg)
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=ownship_provider,
        target_action_provider=target_provider,
        reward_fn=reward_fn,
    )
    try:
        env.reset(seed=args.seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            if ownship_provider.info_log and ownship_provider.info_log[-1]["sim_time"] >= args.seconds:
                break
    finally:
        env.close()

    rows = ownship_provider.info_log
    ablation_label = "p0"
    if args.pc_disable_alignment_gate:
        ablation_label = "p1_noalign"
    elif args.pc_disable_pitch_envelope:
        ablation_label = "p2_noenv"
    elif args.pc_flip_bank_sign:
        ablation_label = "p3_flipbank"
    elif args.pc_world_frame_pitch_gate:
        ablation_label = "p4_worldpitch"
    elif args.pc_turn_pull_decomposition:
        if args.pc_turn_pull_priority_gate:
            ablation_label = "p6_prioritygate"
        elif args.pc_vertical_damping_gain is not None:
            ablation_label = "p5b_turnpull"
        else:
            ablation_label = "p5_turnpull"
    if args.pc_turn_sign_flip_deg is not None:
        ablation_label += "_p7flip"
    out_path = Path(args.out) if args.out else ROOT / "artifacts" / "logs" / f"pc_diag_{ablation_label}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in COLUMNS})

    print(f"[diag] wrote {len(rows)} rows to {out_path}")
    print(f"{'t':>5} {'state':<13} {'ata':>6} {'aa':>6} {'dist':>7} {'az':>7} {'el':>6} "
          f"{'tgt_bank':>8} {'cur_bank':>8} {'bank_err':>8} {'align':>6} {'pitch_raw':>9} "
          f"{'spd':>6} {'alt':>6}")
    for row in rows:
        print(f"{row['sim_time']:5.1f} {row['state']:<13} {row['ata']:6.1f} {row['aa']:6.1f} "
              f"{row['distance']:7.0f} {row['los_az']:7.1f} {row['los_el']:6.1f} "
              f"{row['target_bank']:8.1f} {row['current_bank']:8.1f} {row['bank_error']:8.1f} "
              f"{row['alignment']:6.2f} {row['pitch_raw']:9.3f} {row['own_speed']:6.0f} {row['own_alt']:6.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
