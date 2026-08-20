"""Stage6gg: per-angle stability-only training (angle=0deg) -- one of four
separate, dedicated single-angle runs replacing stage6ff's single pooled
800iter run, per explicit user direction (2026-08-18: "한 학습을 3시간 말고
여러 각도의 학습을 진행시켜줘" -- not one 3h run, run multiple angles
instead).

Background: stage6dd (4-angle POOLED training, 80iter) raised
wez_episode_rate to 20% but its raw-policy trajectory probe (safety
override OFF) showed all 3 replayed episodes crashing via unstable,
oscillating roll/pitch (roll -20deg to 163deg, pitch +40deg to -71deg)
rather than one clean sustained maneuver -- the policy has not learned
smooth, recoverable flight control against the ACTIVE scripted_pursuit
opponent. Per user direction, this sets aggression aside entirely (same
reward-zeroing as the now-stopped stage6ff: range/ata/aa/wez/damage/
attack_range/far_range/range_delta/ata_delta/overshoot/inside_min_range
all 0.0, only altitude/attitude safety terms active) and, instead of one
long pooled run, trains FOUR separate single-angle runs -- 0/8/15/22deg,
matching stage6cc/6dd's pool -- so each angle gets its own fully dedicated
iteration budget rather than sharing one pool (avoiding the dilution
problem stage6aa/6bb already demonstrated: pooling angles thins
per-angle experience).

This is the 0deg run. Same fresh restart from stage6e_delta_down_400iter,
same eased scripted_pursuit opponent config (heading_to_bank_gain=0.4,
max_bank_deg=25) as every prior scripted_pursuit experiment. 100
iterations (~22min at this setup's empirically observed ~13s/iter,
calibrated from stage6cc/6dd's own training_log.csv) per angle -- a more
conservative choice than the original 200iter plan, since 100iter is the
point where several PRIOR (aggression-reward-active) experiments on this
line first started to regress; testing that boundary here doubles as a
check on whether the stability-only reward is actually free of the same
iteration-collapse pattern. Four such runs use only ~88min of the ~3h
budget, leaving time to extend or re-seed whichever angle(s) look
promising. Seed 261849 (next unused in the 2618xx family after
stage6ff's 261848).

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion. Note the
frozen-eval "gate" thresholds (wez_episode_rate>=10% etc.) do not apply
meaningfully here since wez reward is intentionally zeroed -- the metric
that matters for this line is crash rate / minimum_altitude / absence of
wild roll-pitch oscillation, not engagement rate.

Usage: python scripts/run_stage6jj_stability_angle22_100iter.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6jj_ab_results.csv"

ITERATIONS = 100
WARMUP_STEPS = 2000
SEED = 261852  # next unused seed in the 2618xx family (261851 used by stage6ii).

_ORIGINAL_DISTANCE_M = math.hypot(530.0, 530.0)  # 749.53..., same as close_quarters_750
_ANGLE_DEG = 22.0

OWNSHIP_START = [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0]
TARGET_START = [
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
    "heading_to_bank_gain": 0.4,
    "max_bank_deg": 25.0,
}

VARIANT = {
    "suffix": "stage6jj_stability_angle22_100iter",
    "reward_module": "student.my_reward_delta_v1",
    # Same safety/control terms as stage6cc-6ee, but every approach/
    # engagement-driving term zeroed (see stage6ff for the full rationale).
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
        "100iter SINGLE fresh run (restore from stage6e_delta_down_400iter "
        "directly) at ata approx 0deg starting geometry, stability-only "
        "reward (all approach/engagement terms zeroed, only altitude/"
        "attitude safety terms active), against scripted_pursuit at the "
        "eased difficulty (heading_to_bank_gain=0.4/max_bank_deg=25). One "
        "of 4 dedicated single-angle stability runs (0/8/15/22deg) "
        "replacing stage6ff's single pooled 800iter run, per user request "
        "to split the ~3h budget across angles instead of one long run."
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
    scenario["name"] = "stability_angle22_750"
    scenario["ownship"] = list(OWNSHIP_START)
    scenario["target"] = list(TARGET_START)

    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
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
    print(f"[stage6jj] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6jj] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6gg stability-angle22 result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
