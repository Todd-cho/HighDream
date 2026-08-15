"""Continue stage6e_delta_down_400iter (current official-best candidate)
with the training-time opponent swapped from the passive "loiter" target to
ScriptedPursuitActionProvider -- a genuinely active, closing (non-learned)
opponent -- while leaving the reward config completely unchanged.

Background: every training run and every reward-shaping experiment this
project has run so far used target_mode="loiter" (the target barely moves).
2026-08-12 active-opponent verification (scripts/adhoc_scripted_pursuit_eval.py)
found that stage6e_delta_down_400iter has wez_episode_rate=0% and
target_damage=0% against a genuinely pursuing opponent, despite 65-80%
against loiter: step-level trajectory replay showed the ownship tracks well
at range (t=5.1s, 1866m, ata=16.8deg) but the angle degrades as it closes
(t=7.5s, 452m, ata=42.4deg), passing almost perpendicular at closest approach
(t=8.1s, 269.5m, ata=94.5deg) -- pure pursuit (chasing the opponent's current
position) with no lead. Every reward term used so far (range_delta, ata_delta,
pursuit_heading, wez_proximity, ...) is a *reactive* signal (is distance/angle
improving right now), never explicitly requiring the opponent's own motion to
be anticipated, and no training run has ever experienced a moving opponent to
begin with.

This is the cheapest, most fundamental thing to test before touching reward
design at all: does merely training against a moving opponent (with the
existing range/ata-based reward, which does already push toward closing
distance and aligning angle) teach some amount of lead-pursuit habit for
free? Everything else (reward_module, reward config, scenario,
close_quarters_750 initial positions, algo hyperparameters) is carried over
byte-for-byte from stage6e_delta_down_400iter.yaml -- only env.target_mode
changes (loiter -> scripted_pursuit) plus the new target_scripted_pursuit
cruise_altitude_m, set to 4500m to match this scenario's target starting
altitude (D=-4500 in close_quarters_750) rather than the eval script's
default 7000m, which was only validated against the unrelated "default"
scenario. Small increment (50iter, rule 13: this project's repeated pattern
of "big change + long run = seed-dependent collapse"), fresh from the
original stage6e_delta_down_400iter checkpoint (not any of the abandoned
stage6h/6i lines), both restore-stability fixes active
(--replay-warmup-steps=2000, auto-inferred --initial-alpha).

IMPORTANT (rule 15): the in-training gate_metrics/recovery_metrics printed
by this script are computed from *training* (exploration-noise) episodes,
which this project has repeatedly found to give a false read (e.g.
stage6e_delta_down_400iter's own "4/4 gate" claim from 2026-08-07 turned out
to be exploration-only and reversed under frozen eval on 2026-08-11). Treat
this run's own printed numbers as a rough pilot signal only -- a frozen,
deterministic re-evaluation against ScriptedPursuitActionProvider (following
the scripts/adhoc_scripted_pursuit_eval.py pattern, pointed at this run's
checkpoint) is required before drawing any conclusion.

Usage: python scripts/run_stage6j_scripted_pursuit_pilot.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6j_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261825  # next unused seed in the 2618xx family (261824 used by the postmerge_recovery scale015 reproduce run).

VARIANT = {
    "suffix": "stage6j_scripted_pursuit_opponent_50iter",
    "reward_module": "student.my_reward_delta_v1",
    # Exact stage6e_delta_down_400iter reward config, byte-for-byte unchanged.
    "reward_overrides": {
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
    },
    "notes": (
        "50iter continuation from stage6e_delta_down_400iter (original) with "
        "the training-time opponent swapped loiter -> scripted_pursuit "
        "(ScriptedPursuitActionProvider, cruise_altitude_m=4500 matching this "
        "scenario's target start altitude). Reward module/config completely "
        "unchanged from stage6e_delta_down_400iter -- single-variable test of "
        "whether experiencing a moving opponent alone teaches any lead-pursuit "
        "habit, before trying a purpose-built predictive/lead reward term."
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
    experiment["env"]["target_mode"] = "scripted_pursuit"
    experiment["env_config"]["reward"] = dict(VARIANT["reward_overrides"])
    experiment["env_config"]["target_scripted_pursuit"] = {"cruise_altitude_m": 4500.0}
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
    print(f"[stage6j] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6j] NOTE: these are training-exploration episodes against "
          "scripted_pursuit, not a frozen eval -- rule 15 applies, re-verify "
          "with a frozen adhoc_scripted_pursuit_eval.py-style run before "
          "concluding anything.")
    print(f"[done] Stage6j scripted_pursuit-opponent pilot result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
