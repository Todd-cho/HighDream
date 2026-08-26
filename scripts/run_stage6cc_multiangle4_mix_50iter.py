"""Stage6cc: shrink the mixed-angle pool from 12 points to 4 -- more
repetitions per angle within the same 50iter budget.

Background: both stage6aa (12-angle pool, 100iter) and stage6bb (same
12-angle pool, 50iter) produced wez_episode_rate=0.0% in frozen eval --
WORSE than every single-fixed-angle experiment in this line (stage6u/6v/
6w/6x/6y/6z all landed between 5% and 25%). stage6bb (50iter) at least
avoided stage6aa's total-flight collapse (mean_distance 8261.9m vs
30730.1m, both roughly at/above the 7500m gate), but zero engagements
across 12 angles x 50iter suggests each angle gets too few repetitions
to learn anything beyond generic caution: 50iter spread over 12 pool
entries is roughly 4 iterations of experience per angle, far short of
the 50-iteration budget every single-angle stage6u/6v/6w run got
dedicated to just one angle.

This script keeps stage6bb's recipe (fresh restart from
stage6e_delta_down_400iter, 50iter, same opponent/reward config) but
shrinks the pool from 12 angles (0-22deg, 2deg steps) down to 4 discrete
points -- 0/8/15/22deg, exactly the four angles this line has already
validated individually (stage6u/6w/6v, plus 0deg's stage6o/6s/6t
lineage) -- so each angle gets roughly 12-13 iterations of experience
within the same 50iter budget, a partial step back toward per-angle
specialization while still testing whether ANY cross-angle
generalization survives. New seed 261845 (next unused after stage6bb's
261844).

Background (angle-curriculum context, unchanged from stage6aa): stage6u/
6v/6w (single seed, fresh restart, ata approx 8/15/22deg respectively)
were each retried with a second seed (stage6z/6x/6y). None of the three
angles passed Stage6 gates reliably across both seeds:
  8deg:  seed1(stage6u) 4/4 clean pass   -> seed2(stage6z) 1/4 (only crash;
         min_altitude_mean 1632m->959m, mean_distance 3270m->12414m,
         wez_episode_rate 25%->5%) -- a complete reversal.
  15deg: seed1(stage6w) 3/4 (wez=5%, fail) -> seed2(stage6x) 4/4 (wez=10%,
         right at the boundary).
  22deg: seed1(stage6v) 2/4 (altitude/distance fail, wez=10% boundary
         pass)  -> seed2(stage6y) 3/4 (wez=5%, fail).
See project-aip-altitude-status memory, 2026-08-17~18 entry, for the full
6-row table. Conclusion: no single angle is seed-stable with this recipe
(50iter, fresh restart, single fixed starting geometry per run) -- the
"find the working angle" framing itself was the wrong question. Also, the
official competition rules (slide 15, re-checked 2026-08-17) only fix the
starting DISTANCE (2000-3000ft / 610-914m) for prelim/round1-3; starting
angle is never specified as fixed, and the exact starting numbers are
explicitly "to be announced later" -- so a policy specialized on any single
angle is answering a question the competition may not even ask that way.

This script switches the lever entirely: instead of continuing the
single-angle-curriculum approach (more seeds, more iterations on one
angle), it trains against a *distribution* of starting angles, uniformly
resampled every episode reset via the engine's existing `scenario_pool`
mechanism (src/dogfight/envs/initial_scenario.py, already used by
stage6u/6v/6w/6x/6y/6z for their single-scenario pools; validated,
reused code path -- no new engine logic needed). 12 discrete angles, 0deg
to 22deg in 2deg steps (the range spanned by stage6u/6v/6w, now unified
into one pool with equal weight per angle so each episode's starting
geometry is close to uniform over [0,22]deg), same 749.5m starting
distance, same target heading=90deg, all reusing stage6u/6v/6w's exact
geometry formula.

Single-variable relative to stage6u/6v/6w's line: only the starting
geometry changes from ONE fixed angle to a uniform 12-point DISTRIBUTION
over [0,22]deg. Opponent config (heading_to_bank_gain=0.4, max_bank_deg=25,
the eased difficulty used since stage6o) and reward module/config are
carried over byte-for-byte, unchanged. Iteration count raised from 50 to
100 (the harder, more varied training signal was expected by the user to
need more iterations to converge) -- this deviates from rule13's "small
increment" default deliberately, per explicit user request, since the
previous single-angle 50iter results were noise-dominated regardless of
iteration count and a genuinely different lever (distribution vs point) is
being tested, not an iteration-count escalation of the same recipe.

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion. Because the
eval scenario config is read back from this tag's own training YAML
(load_training_env_config), the frozen eval will ALSO sample from the same
12-angle pool per episode (env.reset(seed=...) drives the pool's RNG) --
so a single frozen run already gives a multi-angle read, not just one
angle's worth of episodes.

Usage: python scripts/run_stage6cc_multiangle4_mix_50iter.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6cc_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261845  # next unused seed in the 2618xx family (261844 used by stage6bb).

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
    "suffix": "stage6cc_multiangle4_mix_50iter",
    "reward_module": "student.my_reward_delta_v1",
    # Exact stage6e_delta_down_400iter / stage6u-6z reward config, byte-for-byte unchanged.
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
        "100iter SINGLE fresh run (restore from stage6e_delta_down_400iter "
        "directly), starting geometry is a uniform 12-point pool over "
        "ata in [0,22]deg (2deg steps) resampled every episode via the "
        "engine's scenario_pool mechanism, instead of stage6u/6v/6w's "
        "single fixed angle per run. Opponent (heading_to_bank_gain=0.4/"
        "max_bank_deg=25) and reward module/config carried over "
        "byte-for-byte. Follows the discovery that no single angle "
        "(8/15/22deg) was seed-stable (stage6u/6v/6w vs their seed2 "
        "retries stage6z/6x/6y all flipped pass/fail) -- switches the "
        "lever from 'find the right angle' to 'train across the whole "
        "range at once', per explicit user direction after reviewing the "
        "seed-reproducibility results."
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
    print(f"[stage6cc] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6cc] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6cc multiangle4-mix result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
