"""Retry the pursuit_heading directional-shaping term with a smaller scale
(0.2 -> 0.1), isolated as a single-variable change against the original pilot.

Background: the original pilot (run_stage6g_turnback_pursuit_heading.py,
pursuit_heading_scale=0.2, 50iter from turnback_wez_hold_50iter) landed
closest to the Stage6 gate this session (frozen eval: crash 0%, mean_distance
7512.6m vs 7500 target, wez_episode_rate 0.05 vs 0.10 target) and produced
this line's first real WEZ damage hit. A 50iter continuation of that same
run (run_stage6g_turnback_pursuit_heading_continue.py, cumulative 100iter,
unchanged scale=0.2) then closed distance much further (mean_distance
2985.6m, clearly under gate) but wez_episode_rate regressed to 0.0 (from
0.05) and crash_rate rose 0% -> 15% (3/20, worst min-altitude 286m) --
per-episode inspection showed min_distance collapsing to 166-330m (well
inside the official 610-914m engagement band) while final_ata_deg stayed
bad (72-176 deg), i.e. the policy is diving straight through the target
without matching the angle condition, consistent with pursuit_heading's
directional pull being strong enough to cause overshoot/dive risk before
angle catches up.

Hypothesis (user-directed): pursuit_heading_scale=0.2 is pulling too hard --
a weaker pull (0.1, half strength) might still provide enough directional
signal to close distance (rule 16 diagnosis: direction, not magnitude, was
the missing ingredient) without overpowering the angle-alignment and
altitude-safety terms. This restarts from the exact same source checkpoint
and iteration count as the original pilot (turnback_wez_hold_50iter, 50iter,
same reward module/overrides otherwise) with only pursuit_heading_scale
changed 0.2 -> 0.1, for a clean A/B against the original pilot's frozen eval.

Usage: python scripts/run_stage6g_turnback_pursuit_heading_scale_down.py [--dry-run]
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
SEED = 261814  # next unused seed (261813 used by the pursuit_heading 50->100iter continuation).
MAX_ENGAGE_TIME_S = 120.0  # same as the original pursuit_heading pilot.

VARIANT = {
    "suffix": "turnback_pursuit_heading_scale010_50iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.5,
        "wez_proximity_use_aa": True,
        "wez_hold_scale": 0.4,
        "wez_hold_break_penalty_scale": 0.3,
        "pursuit_heading_scale": 0.1,
    },
    "notes": (
        "Single-variable retry of the original pursuit_heading pilot with "
        "pursuit_heading_scale halved (0.2 -> 0.1). Same source checkpoint "
        "(turnback_wez_hold_50iter), same iteration count (50), same seed "
        "family, everything else unchanged. The 0.2 line's 100iter "
        "continuation overshot badly (min_distance collapsing to 166-330m, "
        "well inside the 610-914m engagement band, final_ata_deg still bad) "
        "and lost its only WEZ hit -- testing whether a gentler directional "
        "pull still closes distance without causing overshoot/dive risk."
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
    print(f"[done] Stage6g pursuit_heading scale-down result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
