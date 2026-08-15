"""Stage6n: curriculum step 2 -- combine (a) an even easier scripted_pursuit
opponent than stage6m and (b) a much larger iteration budget (150 vs 50),
per explicit user request to try both at once rather than one at a time.

Background: stage6m (curriculum step 1: heading_to_bank_gain 1.5->0.4,
max_bank_deg 70->25, 50iter, reward untouched) still landed on
wez_episode_rate=0.0% in frozen eval (crash 0%, min_distance_mean 508m --
closer than any prior scripted_pursuit attempt, but still never converting
to an actual WEZ hit). That makes FOUR consecutive scripted_pursuit attempts
(stage6j/6k/6l/6m) at exactly 0.0% wez_episode_rate.

This run deviates from rule 13 (single-variable discipline) at the user's
explicit direction: it changes both the opponent-difficulty axis AND the
iteration-budget axis in one shot. If the result is surprising, the two
effects are not separable from this run alone -- flagged here per this
project's established pattern (see stage6l's docstring for the precedent).

Two changes:
1. Opponent eased further than stage6m (fresh from stage6e_delta_down_400iter,
   NOT continuing from stage6m's checkpoint, to avoid stacking a second
   restore-vulnerability window (rule 14) on top of an already-50iter run):
     - heading_to_bank_gain: 0.4 -> 0.15 (stage6m's value, halved again)
     - max_bank_deg: 25 -> 15 (stage6m's value, further reduced)
   cruise_altitude_m unchanged at 4500 (matches this scenario's target start
   altitude).
2. Iteration budget: 50 -> 150 (3x stage6m's, within the 100-200 range
   discussed with the user), giving the replay buffer (capacity 10000,
   ~100 iterations to fill at this env's throughput) room to actually reach
   steady state before evaluation, unlike every prior scripted_pursuit
   attempt which stayed inside the post-restore refill window (rule 14)
   for its entire run.

Reward module/config is byte-for-byte unchanged from stage6e/6j/6m
(student.my_reward_delta_v1) -- only the two opponent-gain params and the
iteration count change relative to stage6m.

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6n_easier_pursuit_150iter.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6n_ab_results.csv"

ITERATIONS = 150
WARMUP_STEPS = 2000
SEED = 261829  # next unused seed in the 2618xx family (261828 used by stage6m).

EASIER_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 0.15,  # stage6m used 0.4 (default 1.5) -- halved again
    "max_bank_deg": 15.0,  # stage6m used 25 (default 70) -- further reduced
}

VARIANT = {
    "suffix": "stage6n_easier_pursuit_150iter",
    "reward_module": "student.my_reward_delta_v1",
    # Exact stage6e_delta_down_400iter / stage6j / stage6m reward config, byte-for-byte unchanged.
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
        "150iter fresh start from stage6e_delta_down_400iter (original) "
        "against scripted_pursuit with the opponent eased further than "
        "stage6m (heading_to_bank_gain 0.4->0.15, max_bank_deg 25->15) AND "
        "a 3x larger iteration budget (50->150), combined at user's "
        "explicit request. Deviates from rule 13 single-variable discipline "
        "-- if results are surprising, the two effects are not separable "
        "from this run alone."
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
    experiment["env_config"]["target_scripted_pursuit"] = dict(EASIER_PURSUIT_CFG)
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
    print(f"[stage6n] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6n] NOTE: these are training-exploration episodes against "
          "an eased scripted_pursuit opponent, not a frozen eval -- rule 15 "
          "applies, re-verify with adhoc_scripted_pursuit_eval_stage6j.py "
          "--tag before concluding anything.")
    print(f"[done] Stage6n easier-pursuit+150iter curriculum pilot result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
