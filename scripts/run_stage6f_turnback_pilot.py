"""Short fair A/B pilot for the Stage6f ata-recovery ("turn-to-track") reward
term against a plain continuation control, both starting fresh from the
passing stage6e_delta_down_400iter checkpoint.

Background: stage6e_delta_down_400iter clears all 4 Stage6 safe_wez_geometry
gates, but post-hoc analysis of final_ata_deg found 6/17 evaluation episodes
ending 128-173 deg (behind/beside the target) -- mostly episodes where
min_distance_m dropped to a few hundred meters or less (overshoot) with no
recovery turn afterward. student/my_reward_delta_v1_turnback.py adds an
ata_recovery bonus that only fires while reducing ata from a bad (>90 deg)
starting angle, see that module's docstring for the full rationale.

Usage: python scripts/run_stage6f_turnback_pilot.py [--dry-run]
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


BASE_YAML = (
    ROOT / "artifacts" / "altitude_attack_followup_v1" / "generated"
    / "stage6e_delta_down_400iter.yaml"
)
SOURCE_TAG = "altitude_attack_followup_v1_stage6e_delta_down_400iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6f_ab_results.csv"

# Official Stage6 safe_wez_geometry gate (must not regress).
GATE = {
    "crash_max": 0.25,
    "altitude_min": 1500.0,
    "mean_distance_max": 7500.0,
    "wez_episode_rate_min": 0.10,
}

ITERATIONS = 50
SEED = 261800

VARIANTS = [
    {
        "suffix": "control_50iter",
        "reward_module": "student.my_reward_delta_v1",
        "reward_overrides": {},
        "notes": (
            "Stage6f control - plain 50iter continuation from "
            "stage6e_delta_down_400iter, same my_reward_delta_v1 module/params, "
            "no ata_recovery term. Isolates 'more training' from the new term."
        ),
    },
    {
        "suffix": "turnback_50iter",
        "reward_module": "student.my_reward_delta_v1_turnback",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
        },
        "notes": (
            "Stage6f treatment - 50iter continuation from "
            "stage6e_delta_down_400iter with my_reward_delta_v1_turnback, "
            "ata_recovery_scale=0.15 (same order of magnitude as the existing "
            "ata_delta_scale=0.15) so it only fires while ata > 90deg and "
            "actively closing."
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
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["notes"] = variant["notes"]
    return tag, experiment


def passed_gate(metrics: dict[str, Any]) -> bool:
    def num(key: str) -> float | None:
        value = metrics.get(key)
        return None if value is None else float(value)

    crash = num("episode_crash_rate")
    altitude = num("minimum_altitude_mean_m")
    distance = num("mean_distance_m")
    wez_rate = num("wez_episode_rate")
    return bool(
        metrics.get("evaluation_episodes", 0) >= 10
        and crash is not None and crash <= GATE["crash_max"]
        and altitude is not None and altitude >= GATE["altitude_min"]
        and distance is not None and distance <= GATE["mean_distance_max"]
        and wez_rate is not None and wez_rate >= GATE["wez_episode_rate_min"]
    )


def overshoot_recovery_metrics(tag: str, window: int) -> dict[str, Any]:
    """Overshoot (min_distance_m < too_close_m) recovery rate, not tracked by
    the official gate -- read directly from episode_summary*.csv."""
    too_close_m = 460.0
    run_dir = ROOT / "artifacts" / "logs" / "highdream" / tag
    rows: list[dict[str, str]] = []
    for path in sorted(run_dir.glob("episode_summary*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    evaluation = rows[-window:]

    def num(row: dict[str, str], key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    ata_values = [v for row in evaluation if (v := num(row, "final_ata_deg")) is not None]
    overshoot_rows = [
        row for row in evaluation
        if (v := num(row, "min_distance_m")) is not None and v < too_close_m
    ]
    recovered = [
        row for row in overshoot_rows
        if (v := num(row, "final_ata_deg")) is not None and v < 90.0
    ]
    bad_ata_rows = [v for v in ata_values if v >= 90.0]
    return {
        "episodes_in_window": len(evaluation),
        "mean_final_ata_deg": (sum(ata_values) / len(ata_values)) if ata_values else None,
        "bad_ata_rate": (len(bad_ata_rows) / len(ata_values)) if ata_values else None,
        "overshoot_episode_count": len(overshoot_rows),
        "overshoot_recovered_count": len(recovered),
        "overshoot_recovery_rate": (
            len(recovered) / len(overshoot_rows) if overshoot_rows else None
        ),
    }


def main() -> int:
    args = parse_args()
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base pilot YAML missing: {BASE_YAML}")
    if not checkpoint_final(SOURCE_TAG).exists():
        raise FileNotFoundError(
            f"Source checkpoint missing: {checkpoint_final(SOURCE_TAG)}"
        )
    base = load_yaml(BASE_YAML)

    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
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
        print(f"[done] Stage6f A/B results written to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
