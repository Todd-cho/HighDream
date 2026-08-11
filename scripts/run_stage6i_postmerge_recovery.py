"""Continue stage6e_delta_down_400iter (the model being developed further,
2026-08-11) with a narrowly-targeted post-merge climb-recovery bonus, based
on step-level trajectory analysis of exactly what distinguishes crash from
survived episodes.

Background: scripts/adhoc_trajectory_probe.py replayed 10 specific episodes
(4 crash, 3 "flew away", 3 "sustained tracking") from a 200s frozen eval and
found that EVERY episode has a near-collision merge (~5-15m) at ~4400m
altitude, followed by a chaotic multi-second maneuver. The crash/survive
fork happens in the next 10-20s: crash episodes keep a mild nose-down pitch
and bleed altitude continuously down to the ~300m crash floor; survivors
pitch up (positive pitch, climbing) within seconds and arrest the descent.
The prior fix attempt (my_reward_delta_v1_floorstep.py's critical_altitude_penalty,
a blanket penalty active for ANY altitude below altitude_hard_floor_m)
collapsed the policy to 100% crash in 50iter -- likely because most
successful episodes never go below altitude_soft_floor_m at all, so the
blanket penalty perturbed the reward landscape in state space visited
constantly, not just the narrow post-merge danger window, and 50iter (~4
completed episodes given the empty post-restore replay buffer) was nowhere
near enough data for the critic to safely re-adapt to that broad a change.

student/my_reward_delta_v1_postmerge_recovery.py instead adds a *pure bonus*
(never a penalty) that only activates for post_merge_recovery_window_steps
(150 steps = 15s) after distance drops below post_merge_trigger_m (150m, a
tight merge -- well inside too_close_m=460m so normal WEZ-range flying is
unaffected), rewarding this step's altitude gain. Everything else is
stage6e_delta_down_400iter's exact reward config, unchanged. Small increment
(50iter, rule 13), both restore fixes active (--replay-warmup-steps=2000,
auto-inferred --initial-alpha).

Usage: python scripts/run_stage6i_postmerge_recovery.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.run_altitude_attack_followup import RUN_EXPERIMENT, attack_window_metrics, execute
from scripts.run_altitude_native_curriculum import (
    checkpoint_final,
    model_bundle,
    save_rows,
)
from scripts.run_altitude_reward_search import load_yaml, write_yaml
from scripts.run_stage6f_turnback_pilot import overshoot_recovery_metrics, passed_gate

BASE_YAML = (
    ROOT / "artifacts" / "altitude_attack_followup_v1" / "generated"
    / "stage6e_delta_down_400iter.yaml"
)
SOURCE_TAG = "altitude_attack_followup_v1_stage6e_delta_down_400iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6i_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261816  # next unused seed in the 2618xx family (261815 used by stage6h critical_floor_penalty, abandoned).

VARIANT = {
    "suffix": "stage6i_postmerge_recovery_50iter",
    "reward_module": "student.my_reward_delta_v1_postmerge_recovery",
    "reward_overrides": {
        # Exact stage6e_delta_down_400iter reward config carried forward
        # unchanged, only the four post_merge_* keys are new/nonzero.
        "step_penalty": -0.003,
        "survival_bonus": 0.0,
        "too_close_m": 460.0,
        "ideal_range_min_m": 450.0,
        "ideal_range_max_m": 1100.0,
        "range_scale": 2.4,
        "overshoot_penalty": 4.0,
        "overshoot_quadratic_scale": 1.5,
        "inside_min_range_penalty": -3.0,
        "ata_scale": 0.15,
        "aa_scale": 0.04,
        "wez_bonus": 1.0,
        "damage_scale": 20.0,
        "altitude_soft_floor_m": 1200.0,
        "altitude_hard_floor_m": 600.0,
        "low_altitude_penalty": 1.15,
        "very_low_altitude_penalty": 3.45,
        "altitude_bonus_high_min_m": 3400.0,
        "altitude_bonus_high_max_m": 9000.0,
        "altitude_bonus_mid_min_m": 1800.0,
        "altitude_bonus_high": 0.5,
        "altitude_bonus_mid": 0.31,
        "nose_down_altitude_m": 4000.0,
        "nose_down_pitch_deg": -10.0,
        "nose_down_penalty": -1.13,
        "roll_limit_deg": 80.0,
        "roll_limit_penalty": 0.15,
        "pitch_down_limit_deg": -12.0,
        "pitch_down_penalty": 1.2,
        "pitch_up_limit_deg": 35.0,
        "pitch_up_penalty": 0.25,
        "attack_range_min_m": 250.0,
        "attack_range_max_m": 2200.0,
        "attack_range_bonus": 0.9,
        "far_range_penalty_start_m": 2500.0,
        "far_range_penalty": 1.05,
        "win_reward": 100.0,
        "loss_reward": -100.0,
        "draw_reward": -30.0,
        "crash_penalty": -280.0,
        "range_delta_scale": 0.03,
        "ata_delta_scale": 0.15,
        # New: post-merge climb-recovery bonus (pure bonus, narrow trigger).
        "post_merge_trigger_m": 150.0,
        "post_merge_recovery_window_steps": 150,
        "post_merge_climb_scale": 0.4,
        "post_merge_climb_clip_m": 5.0,
    },
    "notes": (
        "50iter continuation from stage6e_delta_down_400iter (original, NOT "
        "the failed stage6h critical_floor_penalty checkpoint) with a "
        "narrowly-targeted post-merge climb-recovery bonus, based on "
        "step-level trajectory analysis showing every episode merges to "
        "~5-15m at ~4400m and the crash/survive fork is whether pitch goes "
        "positive (climbing) within ~15s afterward. Pure bonus, only active "
        "within 150 steps of distance<150m, everything else unchanged from "
        "stage6e_delta_down_400iter's exact reward config."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_variant_experiment(base: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_{VARIANT['suffix']}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = VARIANT["reward_module"]
    experiment["env_config"]["reward"] = dict(VARIANT["reward_overrides"])
    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    # initial_alpha deliberately left unset: train_rllib.py auto-infers it
    # from SOURCE_TAG's own training_log.csv (--auto-restore-alpha, default on).
    experiment["notes"] = VARIANT["notes"]
    return tag, experiment


def main() -> int:
    args = parse_args()
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    if not checkpoint_final(SOURCE_TAG).exists():
        raise FileNotFoundError(f"Source checkpoint missing: {checkpoint_final(SOURCE_TAG)}")
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = (
        list(csv.DictReader(RESULT_PATH.open("r", encoding="utf-8-sig", newline="")))
        if RESULT_PATH.exists() else []
    )

    tag, experiment = build_variant_experiment(base)
    yaml_path = CAMPAIGN_DIR / "generated" / f"{VARIANT['suffix']}.yaml"
    write_yaml(yaml_path, experiment)
    code = execute(
        [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"[dry-run] would evaluate {tag}")
        return 0

    gate_metrics = attack_window_metrics(tag, args.evaluation_window)
    recovery_metrics = overshoot_recovery_metrics(tag, args.evaluation_window)
    row = {
        "variant": VARIANT["suffix"],
        "tag": tag,
        "reward_module": VARIANT["reward_module"],
        "status": "success" if code == 0 else "failed",
        "replay_warmup_steps": WARMUP_STEPS,
        **gate_metrics,
        **recovery_metrics,
        "gate_passed": passed_gate(gate_metrics) if code == 0 else False,
        "checkpoint_final": str(checkpoint_final(tag)),
        "model_bundle": str(model_bundle(tag)),
    }
    results.append(row)
    save_rows(RESULT_PATH, results)
    print(f"[stage6i] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print(f"[done] Stage6i postmerge_recovery result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
