"""Continue stage6e_delta_down_400iter (the model the user picked to develop
further, 2026-08-11) with an escalating safety penalty near the true crash
floor, targeting the low_altitude/roll_instability crashes seen under the
real 200s engagement-time frozen eval.

Background: stage6e_delta_down_400iter is by far the most effective engager
in this project (200s frozen eval: wez_steps up to 101/episode, average
target_health ~0.65 i.e. ~35% average damage dealt, wez_episode_rate 70%) but
crashes 25% of the time (5/20), all low_altitude_crash or
altitude_roll_instability_crash, clustered at 296-300m altitude -- right at
the actual crash floor (config.py DEFAULT_ENV_CONFIG min_altitude=300.0,
matching the real competition's 1000ft/304.8m rule). The active reward module
(student.my_reward_delta_v1) only has a *flat* very_low_altitude_penalty for
anything below altitude_hard_floor_m (600m) -- the 600m->300m stretch where
these crashes actually happen carries no extra disincentive for getting even
closer to the ground while aggressively pursuing the target.

student/my_reward_delta_v1_floorstep.py already implements exactly this: an
escalating critical_altitude_penalty term that ramps 0 (at altitude_hard_floor_m)
-> critical_altitude_penalty (at critical_altitude_floor_m), on top of the
existing flat penalty. It was built 2026-08-07 for a different problem (the
delta-reward dive exploit) and lost that A/B to delta-scale-down, but its
mechanism is a direct fit for *this* problem -- it was never actually tried
with a nonzero critical_altitude_penalty in combination with the winning
delta_down config. This run carries forward stage6e_delta_down_400iter's
exact reward config unchanged, adds only critical_altitude_floor_m=320.0
(just above the real 300m crash floor) and critical_altitude_penalty=4.0
(comparable in magnitude to the existing very_low_altitude_penalty=3.45, so
the last 280m before a crash gets steadily more expensive instead of a flat
per-step cost). Single-variable change, small increment (50iter, rule 13),
both restore fixes active (--replay-warmup-steps=2000, auto-inferred
--initial-alpha).

Usage: python scripts/run_stage6h_critical_floor_penalty.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6h_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261815  # next unused seed in the 2618xx family (261814 used by pursuit_heading scale010).

VARIANT = {
    "suffix": "stage6h_critical_floor_penalty_50iter",
    "reward_module": "student.my_reward_delta_v1_floorstep",
    "reward_overrides": {
        # Exact stage6e_delta_down_400iter reward config carried forward
        # unchanged (copied from artifacts/altitude_attack_followup_v1/generated/
        # stage6e_delta_down_400iter.yaml), only the two critical_altitude_*
        # keys are new/nonzero.
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
        "critical_altitude_floor_m": 320.0,
        "critical_altitude_penalty": 4.0,
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
        "50iter continuation from stage6e_delta_down_400iter (the model chosen "
        "to develop further, 2026-08-11), switched to my_reward_delta_v1_floorstep "
        "with stage6e's exact reward config unchanged plus critical_altitude_floor_m=320, "
        "critical_altitude_penalty=4.0 (new escalating penalty between "
        "altitude_hard_floor_m=600m and the true ~300m crash floor). Targets the "
        "200s frozen eval's 25% crash rate (all low_altitude/roll_instability, "
        "clustered at 296-300m) without touching the pursuit/damage terms that "
        "make this line the strongest engager (wez_episode_rate 70%, ~35% avg "
        "damage dealt at 200s)."
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
    experiment["env_config"]["reward"] = dict(VARIANT["reward_overrides"])
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
    print(f"[stage6h] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print(f"[done] Stage6h critical_floor_penalty result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
