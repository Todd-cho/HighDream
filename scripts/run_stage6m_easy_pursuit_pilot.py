"""Stage6m: curriculum step 1 (easy opponent) -- continue
stage6e_delta_down_400iter (current official-best candidate) against
ScriptedPursuitActionProvider with its aggressiveness turned WAY down,
reward module/config completely unchanged from stage6e/stage6j.

Background: stage6j (opponent swapped loiter->scripted_pursuit at full
strength, reward untouched), stage6k (+ raw lead_pursuit reward), and
stage6l (+ differential lead_pursuit reward, engage_time=200) all landed on
wez_episode_rate=0.0% / target_damage=0.0%, unmoved across three different
reward designs. scripts/adhoc_trajectory_probe_stage6k.py's step-level
replay explained why: close_quarters_750 was tuned around a passive loiter
target, so against a target that starts banking toward ownship immediately
(default gains: max_bank_deg=70, heading_to_bank_gain=1.5), the engagement
starts with already-bad geometry (ata 48-51deg at t=0) and the ownship
policy reacts by holding an 85-90deg knife-edge bank for 10+ seconds, which
bleeds altitude it never learned to manage because every prior stage
(4/5/6a-i) only ever saw a near-stationary target. Session-end diagnosis
(2026-08-13, project-aip-altitude-status memory) concluded this is a missing
*capability* (turnfight energy/altitude management under an active
opponent), not a reward-shaping gap -- three different reward redesigns
against the same full-strength opponent all left the two headline metrics
completely unmoved.

This script is curriculum step 1: instead of a bigger reward change or a
bigger iteration budget, it eases the *opponent* itself so the ownship's
first exposure to "target that moves" isn't simultaneously "target that
immediately forces a near-vertical bank". Eased ScriptedPursuitActionProvider
params (both far below default, everything else -- cruise_altitude_m=4500 to
match this scenario's target start altitude, cruise_speed_kcas, all pitch/
throttle gains -- left at class defaults):
  - heading_to_bank_gain: 1.5 -> 0.4  (reacts to bearing error far more
    gently, so it doesn't snap toward a large commanded bank on every step)
  - max_bank_deg: 70 -> 25  (hard ceiling that keeps it well short of the
    knife-edge regime that was the actual altitude-bleeding mechanism)
This keeps the opponent genuinely active (still turns toward ownship, still
requires *some* lead to hit consistently) while removing the specific
"near-vertical bank forced from t=0" failure trigger identified by trajectory
replay -- a middle rung between full loiter (stage6e, no turning at all) and
full-strength scripted_pursuit (stage6j/6k/6l).

Single-variable test: everything else -- reward_module
(student.my_reward_delta_v1), reward config, scenario, algo hyperparameters,
source checkpoint, restore-stability fixes (--replay-warmup-steps=2000,
auto-inferred --initial-alpha) -- is carried over byte-for-byte from
stage6j. Only target_scripted_pursuit's two gain params change. Small
increment (50iter, rule 13).

If this clears wez_episode_rate above 0% (even weakly), it's evidence the
capability gap is opponent-difficulty-shaped and the curriculum should
continue by stepping the gains back up toward full strength over further
stages. If wez_episode_rate is still exactly 0%, that argues against the
opponent-difficulty hypothesis specifically at this easing level, and the
next thing to check is whether the eased opponent still forces the t=0
bad-geometry condition (re-run adhoc_trajectory_probe_stage6k.py-style replay
against this checkpoint before concluding the whole curriculum idea failed).

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>, which reads target_scripted_pursuit straight out of this
run's own training YAML so it automatically replays against the SAME eased
opponent) is required before drawing any conclusion.

Usage: python scripts/run_stage6m_easy_pursuit_pilot.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6m_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261828  # next unused seed in the 2618xx family (261827 used by stage6l).

EASY_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 0.4,  # default 1.5 -- much gentler reaction to bearing error
    "max_bank_deg": 25.0,  # default 70.0 -- stays well short of the knife-edge regime
}

VARIANT = {
    "suffix": "stage6m_easy_pursuit_50iter",
    "reward_module": "student.my_reward_delta_v1",
    # Exact stage6e_delta_down_400iter / stage6j reward config, byte-for-byte unchanged.
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
        "50iter continuation from stage6e_delta_down_400iter (original) "
        "against scripted_pursuit with the opponent's own aggressiveness "
        "eased (heading_to_bank_gain 1.5->0.4, max_bank_deg 70->25) instead "
        "of touching reward design. Curriculum step 1 of the "
        "opponent-difficulty-ramp direction chosen after stage6j/6k/6l all "
        "left wez_episode_rate/target_damage at exactly 0.0% across three "
        "different reward designs against the full-strength opponent."
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
    experiment["env_config"]["target_scripted_pursuit"] = dict(EASY_PURSUIT_CFG)
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
    print(f"[stage6m] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6m] NOTE: these are training-exploration episodes against "
          "an eased scripted_pursuit opponent, not a frozen eval -- rule 15 "
          "applies, re-verify with adhoc_scripted_pursuit_eval_stage6j.py "
          "--tag before concluding anything.")
    print(f"[done] Stage6m easy-pursuit curriculum pilot result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
