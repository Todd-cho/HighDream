"""Stage6s: pure seed-reproduction check of stage6o (easy_start_geometry).

Background: stage6o (fresh from stage6e_delta_down_400iter, target placed
dead ahead so ata approx 0deg at reset, opponent fixed at stage6m's level,
50iter, seed=261831) broke this project's scripted_pursuit losing streak
with frozen wez_episode_rate=0.7 (crash 0%, mean_distance 3450m, real
target damage in several episodes) -- the first non-zero WEZ result across
six consecutive scripted_pursuit attempts (stage6j/6k/6l/6m/6n before it).

However every follow-up continuation FROM stage6o's own checkpoint
(stage6p: angle raised to 22deg, stage6q: angle raised to 8deg, stage6r:
angle left unchanged as a pure restore/continuation control) degraded
badly on mean_distance and safety-override intervention -- and stage6r
(2026-08-14 frozen re-run, see project-aip-altitude-status memory) proved
that specific degradation is caused by the restore/continuation process
itself (rule 14), not by the angle changes, since it reproduced identically
even with the angle held at stage6o's own 0deg. That leaves stage6o's own
headline number (wez_episode_rate=0.7) still resting on a single fresh run,
single seed, never reproduced -- exactly the kind of single-seed result
this project's rules (rule 7: fair A/B needs matched seed but still IS a
single sample) has been burned by before (e.g. stage6f_turnback's "crash
0%" that did not hold up under frozen re-eval, or climb_scale's complete
seed-to-seed reversal in the 2026-08-11 postmerge_recovery reproduction
attempts).

This script is a byte-for-byte reproduction of stage6o -- same source
checkpoint (fresh restart from stage6e_delta_down_400iter, NOT continuing
from stage6o itself, to keep this a clean independent sample rather than
another restore-vulnerable continuation), same easy_start_750 scenario
(target dead ahead, ata approx 0deg, 749.5m distance), same opponent
config (heading_to_bank_gain=0.4, max_bank_deg=25), same reward module/
config (student.my_reward_delta_v1, stage6e's exact overrides), same
50iter budget, same replay-warmup-steps=2000 -- ONLY the seed differs
(261835, next unused in the 2618xx family; 261834 used by stage6r).

If wez_episode_rate again lands well above 0% (even if not exactly 0.7),
that supports stage6o's result being a real, seed-robust capability tied
to the easy starting geometry. If it collapses back toward 0% like every
other scripted_pursuit variant, stage6o's 0.7 was itself a seed-lucky
outlier and the starting-geometry hypothesis needs to be revisited (or
retested across several seeds before trusting any single one).

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6s_easy_start_geometry_reproduce.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6s_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261835  # next unused seed in the 2618xx family (261834 used by stage6r).

# Identical to stage6o: same distance as the original close_quarters_750
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
    "heading_to_bank_gain": 0.4,  # stage6m/6o's level, unchanged here.
    "max_bank_deg": 25.0,
}

VARIANT = {
    "suffix": "stage6s_easy_start_geometry_reproduce_50iter",
    "reward_module": "student.my_reward_delta_v1",
    # Exact stage6e_delta_down_400iter / stage6o reward config, byte-for-byte unchanged.
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
        "50iter fresh start from stage6e_delta_down_400iter against "
        "scripted_pursuit (opponent held at stage6m's fixed difficulty, "
        "heading_to_bank_gain=0.4/max_bank_deg=25), easy_start_750 geometry "
        "(target dead ahead, ata approx 0deg at reset) -- byte-for-byte "
        "reproduction of stage6o with only the seed changed (261835 vs "
        "stage6o's 261831), to check whether stage6o's wez_episode_rate=0.7 "
        "frozen result was a real seed-robust capability or a single-seed "
        "outlier."
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
    print(f"[stage6s] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6s] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6s reproduction pilot result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
