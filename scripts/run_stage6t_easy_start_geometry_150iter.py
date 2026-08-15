"""Stage6t: stabilize the 0deg easy_start_geometry line with a longer SINGLE
fresh run (150iter, no mid-run restore) instead of another restore-based
continuation.

Background: stage6o (fresh, ata approx 0deg, 50iter, seed=261831) got
frozen wez_episode_rate=0.7/crash=0%/mean_distance=3450m. stage6s (byte-
for-byte reproduction, only seed changed to 261835) got wez=0.6 (similar --
the WEZ-approach capability itself reproduces) but crash=15%/mean_distance
=13484m (4x worse) -- so the safety/consistency side of stage6o's result
was seed-lucky, not yet a stable baseline.

Separately, stage6r (2026-08-14, see project-aip-altitude-status memory)
proved that CONTINUING training from a checkpoint (restore) degrades
mean_distance/safety-override intervention on its own, independent of any
angle change -- continuing stage6o for 50 more iterations at the SAME 0deg
angle reproduced the same mean_distance blowout (28302m) and heavy
safety-override use (686-3037 steps/episode) seen in the angle-changing
continuations (stage6p/6q). So "stabilize 0deg by continuing training from
stage6o's checkpoint" would very likely just re-trigger that same
restore-vulnerability confound (rule 14: replay buffer is empty after any
restore(), ~100 iterations needed to refill at this env's throughput) --
it would not distinguish real stabilization from restore noise.

This script avoids that confound entirely: a single uninterrupted 150iter
run, fresh-restored ONCE from stage6e_delta_down_400iter (not from stage6o
or stage6s), at the exact same easy_start_750 geometry (ata approx 0deg)
and opponent config as stage6o/6s. One restore at the very start, then
150 iterations of continuous training with no further restore in between
-- this gives the replay buffer (capacity 10000, ~100 iterations to fill
at this env's throughput) room to reach steady state and stay there for a
further ~50 iterations, unlike every prior scripted_pursuit attempt, which
either stayed inside the post-restore refill window for its entire run
(all the 50iter pilots) or added a SECOND restore on top of an
already-vulnerable checkpoint (stage6n's resume-to-150iter, stage6p/6q/6r).
This mirrors exactly how stage6e_delta_down_400iter itself was produced
(single continuous 200-iter extension from stage6d, no intermediate
restore) -- the one reliably-reproduced success pattern in this project.

Single-variable relative to stage6o/6s: only the iteration budget changes
(50 -> 150) and this is a fresh run (not a continuation of either prior
checkpoint) with the next unused seed (261836; 261835 used by stage6s).
Reward module/config, opponent config (heading_to_bank_gain=0.4,
max_bank_deg=25), and easy_start_750 scenario are carried over byte-for-
byte from stage6o/6s.

If frozen crash_rate comes down toward stage6o's 0% (and stays there,
unlike stage6s's 15%) while wez_episode_rate holds in the 60-70% range
seen in both prior runs, that's a genuinely stabilized 0deg baseline to
build the angle curriculum on top of (via fresh restarts per angle step,
never continuations, per the stage6r finding). If crash/mean_distance are
still noisy at 150iter, that argues iteration budget isn't the lever and
the instability is more fundamental to this reward/scenario combination.

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6t_easy_start_geometry_150iter.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
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
RESULT_PATH = CAMPAIGN_DIR / "stage6t_ab_results.csv"

ITERATIONS = 150
WARMUP_STEPS = 2000
SEED = 261836  # next unused seed in the 2618xx family (261835 used by stage6s).

# Identical to stage6o/6s: same distance as the original close_quarters_750
# (530^2+530^2 -> 749.53m), target placed dead ahead of the ownship's nose
# (heading=0) instead of 45deg off, so ata approx 0deg at reset.
_ORIGINAL_DISTANCE_M = math.hypot(530.0, 530.0)  # 749.53...

OWNSHIP_START = [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0]
TARGET_START_EASY = [
    1000.0 + _ORIGINAL_DISTANCE_M,  # N: dead ahead of ownship (heading=0 -> +N)
    0.0,                             # E: unchanged from ownship's E -> ata≈0
    -4500.0,
    0.0,
    0.0,
    90.0,   # target heading unchanged from original (crossing geometry preserved)
    250.0,
]

EASED_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 0.4,  # stage6m/6o/6s's level, unchanged here.
    "max_bank_deg": 25.0,
}

VARIANT = {
    "suffix": "stage6t_easy_start_geometry_150iter",
    "reward_module": "student.my_reward_delta_v1",
    # Exact stage6e_delta_down_400iter / stage6o / stage6s reward config, byte-for-byte unchanged.
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
        "150iter SINGLE fresh run (one restore at the start only, no "
        "mid-run continuation) from stage6e_delta_down_400iter against "
        "scripted_pursuit (opponent held at stage6m/6o/6s's fixed "
        "difficulty, heading_to_bank_gain=0.4/max_bank_deg=25), "
        "easy_start_750 geometry (target dead ahead, ata approx 0deg at "
        "reset). Attempts to stabilize the 0deg baseline (stage6o crash=0%, "
        "stage6s crash=15% -- seed-sensitive at 50iter) via a longer "
        "uninterrupted run instead of a restore-based continuation, since "
        "stage6r proved continuation itself degrades mean_distance/"
        "safety-override independent of angle."
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
    experiment["env_config"]["target_scripted_pursuit"] = dict(EASED_PURSUIT_CFG)

    scenario = experiment["env_config"]["initial_scenario"]["scenarios"][0]
    scenario["name"] = "easy_start_750"
    scenario["ownship"] = list(OWNSHIP_START)
    scenario["target"] = list(TARGET_START_EASY)

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
    print(f"[stage6t] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6t] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6t 150iter stabilization result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
