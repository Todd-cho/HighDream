"""Test the new pursuit_heading directional shaping term (added 2026-08-11 to
my_reward_delta_v1_wez_proximity.py) against the "approach then flee" pattern,
isolated as a single-variable change.

Background: range_delta already rewards the *outcome* of closing distance
step-over-step, but gives no *directional* signal for how to get there.
Raising max_engage_time toward the real competition value (200s) on top of
turnback_wez_hold_50iter did NOT help -- it made things worse (mean_distance
6951m -> 12556m, n=3), ruling out "not enough time" as the primary cause and
pointing instead at "the policy doesn't know which way to turn to close
distance again once far away." pursuit_heading rewards this step's actual
flight-path direction (previous position -> current position) for pointing
at the target's live position, independent of whether range has started
shrinking yet -- see my_reward_delta_v1_wez_proximity.py's module docstring
for the full design and a synthetic-scenario sanity check (toward target ->
+scale, away -> -scale, perpendicular -> ~0) run in this session's
transcript.

This restarts from turnback_wez_hold_50iter with max_engage_time reverted to
120.0 (the engage200 test regressed distance, so isolate pursuit_heading
against the last known-OK engage-time setting) plus pursuit_heading_scale=0.2
(new, modest -- same order of magnitude as ata_scale/wez_proximity_scale)
added on top of the unchanged wez_hold config. Both restore fixes apply
automatically (--replay-warmup-steps=2000, auto-inferred --initial-alpha).

Usage: python scripts/run_stage6g_turnback_pursuit_heading.py [--dry-run]
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6g_turnback_wez_hold_50iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261812  # next unused seed (261811 used by engage200 run).
MAX_ENGAGE_TIME_S = 120.0  # reverted from the regressed engage200=200.0 trial

VARIANT = {
    "suffix": "turnback_pursuit_heading_50iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.5,
        "wez_proximity_use_aa": True,
        "wez_hold_scale": 0.4,
        "wez_hold_break_penalty_scale": 0.3,
        "pursuit_heading_scale": 0.2,
    },
    "notes": (
        "First test of the new pursuit_heading directional shaping term "
        "(rewards this step's actual flight-path direction for pointing at "
        "the target's live position, +-scale bounded, independent of range_delta). "
        "Restarted from turnback_wez_hold_50iter with max_engage_time reverted "
        "to 120.0 (the engage200=200.0 trial regressed mean_distance and is not "
        "carried forward), all wez_hold/wez_proximity params unchanged, only "
        "pursuit_heading_scale=0.2 added. Targets the diagnosed lack of "
        "directional guidance for the return-to-target maneuver after a merge."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_variant_experiment(base: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_stage6g_{VARIANT['suffix']}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = VARIANT["reward_module"]
    experiment["env_config"]["reward"].update(VARIANT["reward_overrides"])
    experiment["env_config"]["max_engage_time"] = MAX_ENGAGE_TIME_S
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
    yaml_path = CAMPAIGN_DIR / "generated" / f"stage6g_{VARIANT['suffix']}.yaml"
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
    print(f"[stage6g] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print(f"[done] Stage6g pursuit_heading result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
