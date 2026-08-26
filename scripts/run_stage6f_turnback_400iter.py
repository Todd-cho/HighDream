"""Continue the Stage6f turnback variant from its 200iter checkpoint for 200
more iterations (cumulative 400iter), matching the stage6e_delta_down_200iter
-> 400iter continuation convention (200iter increment, seed+1).

Background: stage6f_turnback_200iter (my_reward_delta_v1_turnback,
ata_recovery_scale=0.15) already clears 3/4 Stage6 safe_wez_geometry gates
(crash 0%, altitude 2624m, distance 6245m) and only misses wez_episode_rate
(0.083 vs 0.10 required) over a 12-episode window. The delta-reward altitude
gate that caused stage6d_400iter's crash collapse is unconditionally inherited
by my_reward_delta_v1_turnback.py (ata_recovery_reward is gated by the same
safe_altitude check as range_delta/ata_delta), and the one prior gated
200->400 continuation (stage6e_delta_down) improved rather than regressed.
Control (my_reward_delta_v1_200iter, crash 60%) is not extended further --
it already failed the gate badly enough that more iterations would not be a
fair use of compute.

Usage: python scripts/run_stage6f_turnback_400iter.py [--dry-run]
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6f_turnback_200iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6f_ab_results.csv"

ITERATIONS = 200
SEED = 261802  # turnback_200iter continuation used 261801; +1 here, matching
# the stage6e_delta_down_200iter -> 400iter continuation convention.

VARIANT = {
    "suffix": "turnback_400iter",
    "reward_module": "student.my_reward_delta_v1_turnback",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
    },
    "notes": (
        "Stage6f treatment - 200iter continuation from stage6f_turnback_200iter "
        "(cumulative 400iter), same ata_recovery_scale=0.15, checking whether "
        "wez_episode_rate clears the 0.10 gate with more training."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_variant_experiment(base: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_stage6f_{VARIANT['suffix']}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = VARIANT["reward_module"]
    experiment["env_config"]["reward"].update(VARIANT["reward_overrides"])
    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
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
    yaml_path = CAMPAIGN_DIR / "generated" / f"stage6f_{VARIANT['suffix']}.yaml"
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
    print(f"[done] Stage6f 400iter continuation result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
