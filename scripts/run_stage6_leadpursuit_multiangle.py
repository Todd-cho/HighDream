"""Reintroduce the lead_pursuit (differential form) reward on top of the
Stage4-based, 4-angle-pool, scale=100% baseline -- per user direction
(2026-08-19): "lead pursuit reward 다시 얹어서 시도해줘".

Background: the overnight reward-scale sweep (0-200% of stage6e's
engagement values, 50-150iter, all Stage4-sourced, 4-angle pool) produced
crash=0% across all 11 configurations, but wez_episode_rate never rose
above 5-10% regardless of scale or iteration -- CSV/trajectory analysis
(2026-08-19, triggered by user question "min_range는 어땠어?") confirmed
this is NOT a proximity problem (min_distance_mean was 643-649m in every
single config, comfortably inside the WEZ range bands 152-1219m) but an
ANGLE problem at the moment of closest approach: all 20 episodes of
scale100_100iter ended with final_ata_deg clustered at 160-172deg
(nearly facing directly away). Step-level trajectory replay of episode 13
(524.5m closest approach) showed ata=2.3deg at t=0 degrading to ata=58deg
by t=1.4s -- the merge itself, not any later drift -- classic pure-pursuit
overshoot: chasing the target's CURRENT position rather than leading its
predicted future position. This exactly matches the 2026-08-12 diagnosis
from the stage6j/6k/6l line (against the same scripted_pursuit opponent,
but sourced from stage6e back then).

Reused module: student/my_reward_delta_v1_lead_pursuit_diff.py (the
DIFFERENTIAL form, not the raw stage6k version). The raw form
(my_reward_delta_v1_lead_pursuit.py) scored the lead-projected ata
directly, which turned out to almost always move in the same
direction/magnitude as the existing static `ata` term -- effectively
doubling that term's weight rather than teaching anything new, and drove
stage6k into a sustained ~90deg knife-edge turn that slowly bled altitude
(diagnosed via adhoc_trajectory_probe_stage6k.py, 2026-08-12). The
differential form only rewards when the lead-projected position is
BETTER-aligned than the CURRENT bearing (`lead_pursuit_reward =
lead_pursuit_scale * min(diff_deg, lead_pursuit_diff_clip_deg) / 90.0`
where `diff_deg = max(0, current_ata - lead_ata)`), so it's silent
whenever pure pursuit already looks fine and only fires when leading the
target would genuinely help -- this halved stage6l's safety-override
intervention intensity relative to the raw form back in 2026-08-12, but
that test never got to prove out on engagement rate because it was run
on stage6e's WEZ-detection-limited setup before the 3-phase WEZ fix, and
before this session's finding that stage6e's own residual chase-habit
was independently destabilizing.

This run combines: Stage4 source checkpoint (validated crash-free base),
4-angle scenario_pool (0/8/15/22deg), the scale=100% engagement reward
(this sweep's most-validated, 2-seed-reproduced configuration), PLUS
lead_pursuit_scale=0.3 (stage6l's precedent value) on top -- single new
variable relative to the validated scale100_100iter baseline. Fresh
restart from Stage4, 100 iterations (this line's empirically-best
iteration count -- 50 undertrained, 150 started re-eroding margin).

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6_leadpursuit_multiangle.py --lead-scale 0.3 --seed 261865 --tag-suffix stage6nn_leadpursuit030 [--iterations 100] [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6_leadpursuit_sweep_results.csv"
REWARD_MODULE = "student.my_reward_delta_v1_lead_pursuit_diff"

WARMUP_STEPS = 2000

_ORIGINAL_DISTANCE_M = math.hypot(530.0, 530.0)  # 749.53..., same as close_quarters_750
_ANGLES_DEG = [0.0, 8.0, 15.0, 22.0]

OWNSHIP_START = [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0]

# stage6e_delta_down_400iter's original, validated engagement-reward values.
# Every other field (safety/control) stays fixed regardless of scale.
FULL_ENGAGEMENT_VALUES = {
    "range_scale": 2.4,
    "overshoot_penalty": 4.0,
    "overshoot_quadratic_scale": 1.5,
    "inside_min_range_penalty": -3.0,
    "ata_scale": 0.15,
    "aa_scale": 0.04,
    "wez_bonus": 1.0,
    "damage_scale": 20.0,
    "attack_range_bonus": 0.9,
    "far_range_penalty": 1.05,
    "range_delta_scale": 0.03,
    "ata_delta_scale": 0.15,
}

FIXED_SAFETY_VALUES = {
    "step_penalty": -0.003,
    "survival_bonus": 0.0,
    "too_close_m": 460.0,
    "ideal_range_min_m": 450.0,
    "ideal_range_max_m": 1100.0,
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
    "far_range_penalty_start_m": 2500.0,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "crash_penalty": -280.0,
}

EASED_PURSUIT_CFG = {
    "cruise_altitude_m": 4500.0,
    "heading_to_bank_gain": 0.4,
    "max_bank_deg": 25.0,
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-scale", type=float, required=True,
                         help="lead_pursuit_scale value (stage6l used 0.3).")
    parser.add_argument("--lead-horizon-steps", type=int, default=20,
                         help="lead_pursuit_horizon_steps (steps at 10Hz; default 20 == 2s). "
                              "2026-08-19 trajectory analysis showed the actual merge collapses "
                              "in ~1.4s and the opponent continuously banks/turns, so a shorter "
                              "horizon (5-10 steps == 0.5-1s) may track the target's curved path "
                              "more accurately than the default's straight-line 2s extrapolation.")
    parser.add_argument("--engagement-scale", type=float, default=1.0,
                         help="Fraction of stage6e's engagement reward values (default 1.0 == the "
                              "validated scale100_100iter baseline).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag-suffix", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=None)
    return parser.parse_args()


def build_reward_overrides(engagement_scale: float, lead_scale: float, horizon_steps: int) -> dict[str, float]:
    overrides = dict(FIXED_SAFETY_VALUES)
    for key, full_value in FULL_ENGAGEMENT_VALUES.items():
        overrides[key] = full_value * engagement_scale
    overrides["lead_pursuit_scale"] = lead_scale
    overrides["lead_pursuit_horizon_steps"] = horizon_steps
    overrides["lead_pursuit_velocity_clip_m"] = 40.0
    overrides["lead_pursuit_diff_clip_deg"] = 45.0
    return overrides


def build_variant_experiment(base: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_{args.tag_suffix}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = REWARD_MODULE
    experiment["env"]["target_mode"] = "scripted_pursuit"
    experiment["env_config"]["reward"] = build_reward_overrides(
        args.engagement_scale, args.lead_scale, args.lead_horizon_steps
    )
    experiment["env_config"]["target_scripted_pursuit"] = dict(EASED_PURSUIT_CFG)

    scenarios = [
        {
            "name": f"lead{args.lead_scale:.2f}_{angle_deg:04.1f}deg_750",
            "weight": 1.0,
            "ownship": list(OWNSHIP_START),
            "target": _target_start(angle_deg),
        }
        for angle_deg in _ANGLES_DEG
    ]
    experiment["env_config"]["initial_scenario"]["mode"] = "scenario_pool"
    experiment["env_config"]["initial_scenario"]["scenarios"] = scenarios

    experiment["runtime"]["iterations"] = args.iterations
    experiment["runtime"]["seed"] = args.seed
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    experiment["notes"] = (
        f"lead_pursuit sweep: lead_scale={args.lead_scale}, "
        f"lead_horizon_steps={args.lead_horizon_steps}, "
        f"engagement_scale={args.engagement_scale}, seed={args.seed}, "
        f"iterations={args.iterations}, fresh restart from Stage4 "
        f"({SOURCE_TAG}), 4-angle pool (0/8/15/22deg), reward_module="
        f"{REWARD_MODULE} (differential lead-pursuit form)."
    )
    return tag, experiment


def main() -> int:
    args = parse_args()
    if args.evaluation_window is None:
        args.evaluation_window = args.iterations
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    if not checkpoint_final(SOURCE_TAG).exists():
        raise FileNotFoundError(f"Source checkpoint missing: {checkpoint_final(SOURCE_TAG)}")
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = (
        list(csv.DictReader(RESULT_PATH.open("r", encoding="utf-8-sig", newline="")))
        if RESULT_PATH.exists() else []
    )

    tag, experiment = build_variant_experiment(base, args)
    yaml_path = CAMPAIGN_DIR / "generated" / f"{args.tag_suffix}.yaml"
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
        "variant": args.tag_suffix,
        "lead_scale": args.lead_scale,
        "lead_horizon_steps": args.lead_horizon_steps,
        "engagement_scale": args.engagement_scale,
        "seed": args.seed,
        "iterations": args.iterations,
        "tag": tag,
        "reward_module": REWARD_MODULE,
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
    print(f"[lead-pursuit] {tag}: lead_scale={args.lead_scale} gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[lead-pursuit] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] lead-pursuit sweep result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
