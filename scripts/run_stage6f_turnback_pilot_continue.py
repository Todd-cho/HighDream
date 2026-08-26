"""Continue the Stage6f control/turnback pilot from the 50iter checkpoints for
150 more iterations each (cumulative 200iter), to get enough episodes for a
reliable read.

The 50iter pilot (run_stage6f_turnback_pilot.py) only produced 4-5 episodes
per variant -- far below the project's "last 10-20 episodes" evaluation
standard -- though a same-seed paired comparison across episodes 1-4 showed
turnback avoiding both nose-down crashes control hit on identical scenario
starts. This continuation restores each variant from its own 50iter
checkpoint and runs 150 more iterations (seed+1, matching the
stage6e_delta_down_200iter -> 400iter continuation convention) to reach a
~15-20 episode window.

Usage: python scripts/run_stage6f_turnback_pilot_continue.py [--dry-run]
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
from scripts.run_stage6f_turnback_pilot import (
    GATE,
    overshoot_recovery_metrics,
    passed_gate,
)

BASE_YAML = (
    ROOT / "artifacts" / "altitude_attack_followup_v1" / "generated"
    / "stage6e_delta_down_400iter.yaml"
)
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6f_ab_results.csv"

ITERATIONS = 150
SEED = 261801  # 50iter pilot used 261800; +1 for the continuation, matching
# the stage6e_delta_down_200iter -> 400iter continuation convention.

VARIANTS = [
    {
        "suffix": "control_200iter",
        "source_tag": "altitude_attack_followup_v1_stage6f_control_50iter_C10",
        "reward_module": "student.my_reward_delta_v1",
        "reward_overrides": {},
        "notes": (
            "Stage6f control - 150iter continuation from stage6f_control_50iter "
            "(cumulative 200iter), same my_reward_delta_v1 module/params, "
            "no ata_recovery term."
        ),
    },
    {
        "suffix": "turnback_200iter",
        "source_tag": "altitude_attack_followup_v1_stage6f_turnback_50iter_C10",
        "reward_module": "student.my_reward_delta_v1_turnback",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
        },
        "notes": (
            "Stage6f treatment - 150iter continuation from "
            "stage6f_turnback_50iter (cumulative 200iter), ata_recovery_scale=0.15."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_variant_experiment(base: dict[str, Any], variant: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_stage6f_{variant['suffix']}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = variant["reward_module"]
    experiment["env_config"]["reward"].update(variant["reward_overrides"])
    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(variant["source_tag"]))
    experiment["notes"] = variant["notes"]
    return tag, experiment


def main() -> int:
    args = parse_args()
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = (
        list(csv.DictReader(RESULT_PATH.open("r", encoding="utf-8-sig", newline="")))
        if RESULT_PATH.exists() else []
    )

    for variant in VARIANTS:
        source_checkpoint = checkpoint_final(variant["source_tag"])
        if not source_checkpoint.exists():
            raise FileNotFoundError(f"Source checkpoint missing: {source_checkpoint}")
        tag, experiment = build_variant_experiment(base, variant)
        yaml_path = CAMPAIGN_DIR / "generated" / f"stage6f_{variant['suffix']}.yaml"
        write_yaml(yaml_path, experiment)
        code = execute(
            [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)],
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"[dry-run] would evaluate {tag}")
            continue
        gate_metrics = attack_window_metrics(tag, args.evaluation_window)
        recovery_metrics = overshoot_recovery_metrics(tag, args.evaluation_window)
        row = {
            "variant": variant["suffix"],
            "tag": tag,
            "reward_module": variant["reward_module"],
            "status": "success" if code == 0 else "failed",
            **gate_metrics,
            **recovery_metrics,
            "gate_passed": passed_gate(gate_metrics) if code == 0 else False,
            "checkpoint_final": str(checkpoint_final(tag)),
            "model_bundle": str(model_bundle(tag)),
        }
        results.append(row)
        save_rows(RESULT_PATH, results)
        print(f"[stage6f] {tag}: gate_passed={row['gate_passed']} "
              f"overshoot_recovery_rate={row.get('overshoot_recovery_rate')} "
              f"bad_ata_rate={row.get('bad_ata_rate')}")

    if not args.dry_run:
        print(f"[done] Stage6f 200iter continuation results appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
