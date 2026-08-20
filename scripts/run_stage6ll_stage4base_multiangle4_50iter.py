"""Stage6ll: retry stage6cc's 4-angle mixed pool (0/8/15/22deg, 50iter),
but sourced from the loiter curriculum's Stage4 (banked-descent-recovery)
checkpoint instead of stage6e_delta_down_400iter, with the stability-only
reward (all approach/engagement terms zeroed, per the 2026-08-18 user
direction "공격성은 일단 두고 여러 각도에서 기체 안정성을 가질수있게 학습해보자").

Background: stage6kk (single angle=8deg, same stability-only reward,
sourced from Stage4 instead of stage6e) crashed 0% (0/20) in frozen eval
-- vs stage6hh's 25% (5/20) crash rate for the exact same recipe sourced
from stage6e. Trajectory probe confirmed the mechanism: stage6hh's raw
policy got stuck in near-inverted flight (roll pinned 173-179deg) or dove
near-vertically; stage6kk's raw policy instead degrades via a slow, mild
90s glide-descent while flying away (pitch mostly -1 to -25deg) that the
predictive safety override can catch with plenty of margin. This
confirmed the "residual habit" hypothesis: stage6e's weights carry
hundreds of iterations of approach-reward-driven bias that 100iter of
zeroed-reward fine-tuning couldn't undo, while Stage4 -- verified
crash=0% on banked-descent-recovery, entirely BEFORE any approach reward
was ever introduced -- has genuine recovery competence intact.

User's key strategic point (2026-08-19): a single-angle-specialized model
is the wrong target regardless of which base checkpoint is used, since
the actual competition's starting angle is unspecified ("정확한 수치...는
차후 공개" per slide 15) -- what actually needs validating is whether ONE
model, trained across a distribution of angles, survives whichever angle
shows up. This directly revisits stage6cc's 4-angle pool (which got gate
3/4 from stage6e, safe margins but wez capped at ~5%) with the one change
that just proved decisive for the single-angle case: swap the source
checkpoint to Stage4.

Single-variable relative to stage6cc: ONLY the source checkpoint changes
(stage6e_delta_down_400iter -> Stage4 C10) AND the reward is the
stability-only variant (matching stage6gg-6kk) instead of stage6cc's
original full engagement reward -- both changes were already validated
independently (Stage4-base: stage6kk; stability-only reward: stage6gg-jj)
so combining them here is a natural next step, not a fresh unvalidated
combination. Same 4-angle pool (0/8/15/22deg), same eased scripted_pursuit
opponent config, same 50iter, same close_quarters_750 geometry formula.
New seed 261854 (next unused in the 2618xx family after stage6kk's
261853).

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion. Note the
frozen-eval "gate" wez_episode_rate threshold doesn't strictly apply here
since wez reward is intentionally zeroed -- crash rate / minimum_altitude
is what matters for this line.

Usage: python scripts/run_stage6ll_stage4base_multiangle4_50iter.py [--dry-run]
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
SOURCE_TAG = "altitude_native_curriculum_v1_stage4_banked_descent_recovery_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6ll_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261854  # next unused seed in the 2618xx family (261853 used by stage6kk).

_ORIGINAL_DISTANCE_M = math.hypot(530.0, 530.0)  # 749.53..., same as close_quarters_750
_ANGLES_DEG = [0.0, 8.0, 15.0, 22.0]

OWNSHIP_START = [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0]


def _target_start(angle_deg: float) -> list[float]:
    return [
        1000.0 + _ORIGINAL_DISTANCE_M * math.cos(math.radians(angle_deg)),
        0.0 + _ORIGINAL_DISTANCE_M * math.sin(math.radians(angle_deg)),
        -4500.0,
        0.0,
        0.0,
        90.0,
        250.0,
    ]


EASED_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 0.4,  # stage6m/6o/6t/6u/6v/6w/6x/6y/6z's level, unchanged here.
    "max_bank_deg": 25.0,
}

VARIANT = {
    "suffix": "stage6ll_stage4base_multiangle4_50iter",
    "reward_module": "student.my_reward_delta_v1",
    # Stability-only reward (matches stage6gg-6kk): every approach/engagement
    # term zeroed, only altitude/attitude safety terms active.
    "reward_overrides": {
        "step_penalty": -0.003,
        "survival_bonus": 0.0,
        "too_close_m": 460.0,
        "ideal_range_min_m": 450.0,
        "ideal_range_max_m": 1100.0,
        "range_scale": 0.0,
        "overshoot_penalty": 0.0,
        "overshoot_quadratic_scale": 0.0,
        "inside_min_range_penalty": 0.0,
        "ata_scale": 0.0,
        "aa_scale": 0.0,
        "wez_bonus": 0.0,
        "damage_scale": 0.0,
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
        "attack_range_bonus": 0.0,
        "far_range_penalty_start_m": 2500.0,
        "far_range_penalty": 0.0,
        "win_reward": 100.0,
        "loss_reward": -100.0,
        "draw_reward": -30.0,
        "crash_penalty": -280.0,
        "range_delta_scale": 0.0,
        "ata_delta_scale": 0.0,
    },
    "notes": (
        "50iter SINGLE fresh run (restore from Stage4 "
        "banked_descent_recovery_C10, NOT stage6e_delta_down_400iter), "
        "4-angle scenario_pool (0/8/15/22deg, same as stage6cc), "
        "stability-only reward (all approach/engagement terms zeroed). "
        "Combines two independently-validated changes: Stage4 as source "
        "checkpoint (stage6kk: crash 25%->0% at angle=8deg vs "
        "stage6e-sourced stage6hh) and the stability-only reward "
        "(stage6gg-6kk line). Tests whether stage6kk's single-angle "
        "improvement survives being spread across a 4-angle pool, since "
        "the actual competition starting angle is unspecified (slide 15: "
        "'정확한 수치...는 차후 공개') -- a single-angle-specialized model "
        "answers the wrong question regardless of which base checkpoint "
        "is used."
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

    scenarios = [
        {
            "name": f"multiangle_{angle_deg:04.1f}deg_750",
            "weight": 1.0,
            "ownship": list(OWNSHIP_START),
            "target": _target_start(angle_deg),
        }
        for angle_deg in _ANGLES_DEG
    ]
    experiment["env_config"]["initial_scenario"]["mode"] = "scenario_pool"
    experiment["env_config"]["initial_scenario"]["scenarios"] = scenarios

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
    print(f"[stage6ll] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6ll] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6ll stage4base-multiangle4 result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
