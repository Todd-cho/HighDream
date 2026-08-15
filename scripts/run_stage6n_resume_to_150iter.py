"""Resume stage6n (run_stage6n_easier_pursuit_150iter.py), which was killed
externally at iteration 145/150 with no error (actor_loss was stable
throughout, no divergence) -- only a periodic native checkpoint at
iteration 125 was saved, no checkpoint_final/model bundle, so
adhoc_scripted_pursuit_eval_stage6j.py can't evaluate it yet.

This restores from that iteration-125 checkpoint and runs 25 more
iterations (same env/reward config, same easier-pursuit opponent settings)
to reach the original 150iter target and produce a proper checkpoint_final
+ lightweight bundle. seed+1 vs the original run, matching this project's
established restore-continuation convention (e.g. stage6e 200->400iter,
stage6f 50->200iter).

Usage: python scripts/run_stage6n_resume_to_150iter.py [--dry-run]
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

CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
SOURCE_YAML = CAMPAIGN_DIR / "generated" / "stage6n_easier_pursuit_150iter.yaml"
TAG = "altitude_attack_followup_v1_stage6n_easier_pursuit_150iter_C10"
RESULT_PATH = CAMPAIGN_DIR / "stage6n_ab_results.csv"

REMAINING_ITERATIONS = 25  # 125 (last saved checkpoint) -> 150 (original target)
WARMUP_STEPS = 2000
SEED = 261830  # stage6n used 261829; +1 for this resume continuation.
RESUME_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints" / "highdream" / TAG / "checkpoint_000125"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=150)
    return parser.parse_args()


def build_experiment(base: dict[str, Any]) -> dict[str, Any]:
    experiment = copy.deepcopy(base)
    experiment["runtime"]["iterations"] = REMAINING_ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(RESUME_CHECKPOINT)
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    experiment["notes"] = (
        "Resume of stage6n_easier_pursuit_150iter, killed externally at "
        "iter 145/150 with no error. Restores from its own checkpoint_000125 "
        "and runs 25 more iterations to reach the original 150iter target "
        "and produce checkpoint_final + model bundle."
    )
    return experiment


def main() -> int:
    args = parse_args()
    if not SOURCE_YAML.exists():
        raise FileNotFoundError(f"Source experiment YAML missing: {SOURCE_YAML}")
    if not RESUME_CHECKPOINT.exists():
        raise FileNotFoundError(f"Resume checkpoint missing: {RESUME_CHECKPOINT}")
    base = load_yaml(SOURCE_YAML)

    results: list[dict[str, Any]] = (
        list(csv.DictReader(RESULT_PATH.open("r", encoding="utf-8-sig", newline="")))
        if RESULT_PATH.exists() else []
    )

    experiment = build_experiment(base)
    yaml_path = CAMPAIGN_DIR / "generated" / "stage6n_resume_to_150iter.yaml"
    write_yaml(yaml_path, experiment)
    code = execute(
        [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"[dry-run] would evaluate {TAG}")
        return 0

    gate_metrics = attack_window_metrics(TAG, args.evaluation_window)
    recovery_metrics = overshoot_recovery_metrics(TAG, args.evaluation_window)
    row = {
        "variant": "stage6n_easier_pursuit_150iter_resumed",
        "tag": TAG,
        "reward_module": experiment["env"]["reward_module"],
        "status": "success" if code == 0 else "failed",
        "replay_warmup_steps": WARMUP_STEPS,
        **gate_metrics,
        **recovery_metrics,
        "gate_passed": passed_gate(gate_metrics) if code == 0 else False,
        "checkpoint_final": str(checkpoint_final(TAG)),
        "model_bundle": str(model_bundle(TAG)),
    }
    results.append(row)
    save_rows(RESULT_PATH, results)
    print(f"[stage6n-resume] {TAG}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print(f"[done] Stage6n resume-to-150iter result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
