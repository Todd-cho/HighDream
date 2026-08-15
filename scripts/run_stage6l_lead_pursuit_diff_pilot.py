"""Stage6l pilot: fresh from stage6e_delta_down_400iter, training against
target_mode="scripted_pursuit" with the DIFFERENTIAL lead-pursuit reward
(student.my_reward_delta_v1_lead_pursuit_diff) instead of stage6k's raw-score
version, AND with max_engage_time raised to the actual competition
regulation value (200s) for training itself, not just post-hoc eval.

Background -- reward redesign: stage6k's raw lead_pursuit score (same shape
as the existing static "ata" component, just evaluated against a predicted
future point) turned out to be redundant with that ata component almost all
of the time (step-level trajectory replay in
scripts/adhoc_trajectory_probe_stage6k.py showed it paying out near its max
at the exact moments current ata was already near 0), roughly doubling the
angle-correction incentive rather than teaching anything lead-specific. That
made the policy commit to a much longer/harder knife-edge bank (~85-90deg
sustained 13+ seconds) than any prior checkpoint, which bled altitude
continuously (no vertical lift component available near 90deg bank) until
crashing around t=67-70s in every replayed episode -- a new, slow failure
mode, with zero WEZ/damage improvement to show for it.
student/my_reward_delta_v1_lead_pursuit_diff.py fixes this by only paying
when the predicted point is a strictly BETTER aim point than the live one
(diff_deg = max(0, ata - lead_ata)), which is 0.0 in the common case where
lead_ata tracks ata closely and only positive when the target's actual
motion means leading genuinely helps -- see that module's docstring for the
full mechanism and the trajectory-replay numbers motivating it.
lead_pursuit_scale=0.3 (up from stage6k's 0.2) because this term is
zero-valued most of the time by construction, so a comparable *typical*
nonzero payout needs a larger coefficient.

Background -- engagement time: project memory
(project-aip-competition-rules, slides 11/13/14 of the rules deck) confirms
200s is the actual regulation engagement time, not a "tiebreak extension" --
every training run so far (including stage6e/6j/6k) used max_engage_time=120s
during training and only checked 200s post-hoc at frozen-eval time. This run
trains directly at 200s so the policy's own experience matches the real
competition duration, per user request to align conditions with the rules
deck wherever practical. The close_quarters_750 initial engagement distance
(~750m, computed from the scenario's own N/E offsets) is left unchanged: it
already falls inside the regulation's 610-914m (2000-3000ft) prelim/round1-3
starting-distance band, so no change was needed there. The crash-altitude
threshold (env's min_altitude=300m default, vs the rules deck's 1000ft =
304.8m) is likewise already ~compliant and untouched -- only
altitude_hard_floor_m (a reward-shaping penalty threshold, 600m here) is
different from the true crash floor, which is an intentional safety margin,
not a rules mismatch, and is also left unchanged.

Two changes at once (reward redesign + engagement time) deviates from this
project's usual single-variable discipline (rule 13) -- flagged explicitly
per user's direct request to combine them. If results are surprising, the
two effects are not separable from this run alone and would need a follow-up
to isolate.

Small increment (50iter), fresh single-variable start from
stage6e_delta_down_400iter (not stage6j or stage6k, both abandoned lines),
both restore stability fixes active (--replay-warmup-steps=2000,
auto-inferred --initial-alpha).

Usage: python scripts/run_stage6l_lead_pursuit_diff_pilot.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6l_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261827  # next unused seed in the 2618xx family (261826 used by stage6k lead_pursuit raw-score pilot).
MAX_ENGAGE_TIME = 200.0  # real competition regulation value (rules deck slide 11), not the 120s used by every prior training run.

VARIANT = {
    "suffix": "stage6l_lead_pursuit_diff_scale030_engage200_50iter",
    "reward_module": "student.my_reward_delta_v1_lead_pursuit_diff",
    "reward_overrides": {
        # Exact stage6e_delta_down_400iter reward config, unchanged.
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
        # New: differential predictive lead-pursuit shaping (only new nonzero keys).
        "lead_pursuit_scale": 0.3,
        "lead_pursuit_horizon_steps": 20,
        "lead_pursuit_velocity_clip_m": 40.0,
        "lead_pursuit_diff_clip_deg": 45.0,
    },
    "notes": (
        "50iter fresh start from stage6e_delta_down_400iter with the "
        "DIFFERENTIAL lead-pursuit reward (lead_pursuit_scale=0.3, only pays "
        "when the extrapolated future target position beats the current one "
        "as an aim point -- fixes stage6k's raw-score redundancy with the "
        "static ata term, diagnosed via trajectory replay) AND "
        "max_engage_time raised to the real regulation value (200s, was "
        "120s in every prior training run). Training opponent kept as "
        "scripted_pursuit. Two changes combined at user's explicit request "
        "to align training conditions with the rules deck; not a strict "
        "single-variable test."
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
    experiment["env"]["max_engage_time"] = MAX_ENGAGE_TIME
    experiment["env_config"]["reward"] = dict(VARIANT["reward_overrides"])
    experiment["env_config"]["target_scripted_pursuit"] = {"cruise_altitude_m": 4500.0}
    experiment["env_config"]["max_engage_time"] = MAX_ENGAGE_TIME
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
    print(f"[stage6l] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6l] NOTE: these are training-exploration episodes against "
          "scripted_pursuit, not a frozen eval -- rule 15 applies, re-verify "
          "with a frozen scripted_pursuit eval before concluding anything.")
    print(f"[done] Stage6l lead_pursuit_diff pilot result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
