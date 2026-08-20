"""Stage6u: retry the starting-geometry angle curriculum (0deg -> nonzero),
but this time as a FRESH restart from stage6e_delta_down_400iter directly,
never as a continuation of a 0deg-specialized checkpoint.

Background: stage6p (0->22deg) and stage6q (0->8deg) both continued FROM
stage6o's checkpoint (a checkpoint already specialized on ata approx 0deg)
and both collapsed (wez_episode_rate 0.7->0.0/0.05, mean_distance
3450m->25590m/26522m). Separately, stage6r proved that CONTINUING training
from ANY checkpoint in this line degrades mean_distance/safety-override
independent of angle (same-angle continuation reproduced the same blowout).
So stage6p/6q's collapse is confounded: it's impossible to tell whether the
angle change itself is what broke things, or whether it was really just
another instance of the restore-instability pattern rule14/rule "iteration
increases hurt" keeps reproducing (most recently: this session's stage6t
150iter single run, which regressed vs. its own iter125 snapshot: min
altitude mean 2879m->1215m, gate FAILED on altitude, mean_distance
2696m->7091m -- see stage6t entries in project-aip-altitude-status memory,
2026-08-15).

This script removes that confound by not touching stage6o/6s/6t's
checkpoint at all. It restores fresh from stage6e_delta_down_400iter (the
same, untouched, stable base every prior scripted_pursuit line -- 6j/6k/6l/
6m/6n/6o -- started from) and sets the starting geometry DIRECTLY to
ata approx 8deg (stage6q's target angle, reusing its exact geometry
formula/values) in that single fresh run. The policy never specializes on
0deg first, so there is nothing to "de-specialize" or destabilize when the
angle isn't 0.

Single-variable relative to stage6o (the last successful fresh run from
stage6e_delta_down_400iter): only the starting geometry changes (ata
approx 0deg -> approx 8deg, same 749.5m distance, same target heading=
90deg). Opponent config (heading_to_bank_gain=0.4, max_bank_deg=25, i.e.
the eased difficulty used since stage6o) and reward module/config are
carried over byte-for-byte from stage6o/6t. Small increment (50iter,
rule 13) -- deliberately NOT stage6t's 150iter approach, since that just
demonstrated more iterations regress this line rather than stabilize it.

If frozen wez_episode_rate lands anywhere near stage6o's 0deg result
(60-70%) at 8deg, that's strong evidence stage6p/6q's failure was a
restore-instability confound, not genuine angle-specific memorization --
and the angle curriculum can proceed via repeated fresh restarts (never
continuations) at increasing angles. If it's still near 0%, that argues
0deg truly is a narrow local optimum this reward/curriculum can't
generalize away from, and a different lever (e.g. a genuine multi-angle
training distribution instead of single-angle curriculum steps, or a
turn-energy-management reward term) is needed instead.

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6u_fresh_angle8_50iter.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6u_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261837  # next unused seed in the 2618xx family (261836 used by stage6t).

_ORIGINAL_DISTANCE_M = math.hypot(530.0, 530.0)  # 749.53..., same as close_quarters_750
_ANGLE_DEG = 8.0  # same target angle as stage6q, but reached via fresh restart this time

OWNSHIP_START = [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0]
TARGET_START_ANGLE8 = [
    1000.0 + _ORIGINAL_DISTANCE_M * math.cos(math.radians(_ANGLE_DEG)),
    0.0 + _ORIGINAL_DISTANCE_M * math.sin(math.radians(_ANGLE_DEG)),
    -4500.0,
    0.0,
    0.0,
    90.0,
    250.0,
]

EASED_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 0.4,  # stage6m/6o/6s/6t's level, unchanged here.
    "max_bank_deg": 25.0,
}

VARIANT = {
    "suffix": "stage6u_fresh_angle8_50iter",
    "reward_module": "student.my_reward_delta_v1",
    # Exact stage6e_delta_down_400iter / stage6o / stage6t reward config, byte-for-byte unchanged.
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
        "50iter SINGLE fresh run (restore from stage6e_delta_down_400iter "
        "directly, NOT from stage6o/6s/6t) at ata approx 8deg starting "
        "geometry (stage6q's target angle, reused geometry formula), "
        "against scripted_pursuit at stage6o/6t's eased difficulty "
        "(heading_to_bank_gain=0.4/max_bank_deg=25). Tests whether "
        "stage6p/6q's collapse at nonzero angles was genuine angle-"
        "specific memorization or a restore-instability confound from "
        "continuing an already-0deg-specialized checkpoint (stage6r "
        "showed continuation alone degrades this line, and stage6t's "
        "150iter single run also regressed -- so a fresh single run at "
        "the target angle, never touching a 0deg-adapted checkpoint, is "
        "the cleanest way to isolate the angle effect)."
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
    scenario["name"] = "fresh_angle8_750"
    scenario["ownship"] = list(OWNSHIP_START)
    scenario["target"] = list(TARGET_START_ANGLE8)

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
    print(f"[stage6u] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6u] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6u fresh-angle8 result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
