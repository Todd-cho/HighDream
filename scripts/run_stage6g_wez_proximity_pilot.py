"""Stage6g A/B pilot: does a dense, multiplicative range-x-angle "WEZ
proximity" shaping term (student/my_reward_delta_v1_wez_proximity.py) get
wez_episode_rate off zero, versus a plain continuation control?

Background: stage6f_turnback_200iter clears 3/4 Stage6 safe_wez_geometry
gates but wez_episode_rate is 0.0 -- across every Stage6f variant (control
and turnback, 50/200/400iter alike), not one evaluation episode ever hit the
exact WEZ condition (min_range_m <= distance <= max_range_m AND
ata <= angle_deg/2) simultaneously. Per-episode final_distance_m/final_ata_deg
pairs show why: good-ata episodes are far away, close-approach episodes have
bad ata that only corrects afterward (by which point distance has re-opened).
The wez_proximity module adds a dense multiplicative proxy for the joint
condition so gradient exists before the sparse wez_bonus can ever fire -- see
that module's docstring for the full rationale.

Iteration count is deliberately smaller than the last Stage6f continuation
(100 vs 200): stage6f_turnback_400iter collapsed into crash 65% despite the
same delta-reward altitude gate being present, for reasons not yet fully
understood, so this pilot takes a more conservative step and the resulting
episode_summary.csv should be inspected in full timeline order (not just the
final-window aggregate) to catch any similar drift early -- see
[[feedback-aip-experiment-rules]] rule 11.

Both variants restore from the same stage6f_turnback_200iter checkpoint (the
best-so-far candidate: crash 0%, altitude 2624m, distance 6245m) with the
same seed, isolating "more training with the existing turnback reward" from
"more training with the new wez_proximity term added on top."

Usage: python scripts/run_stage6g_wez_proximity_pilot.py [--dry-run]
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
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 100
SEED = 261803  # stage6f_turnback_200iter used 261801, the (regressed)
# turnback_400iter continuation used 261802; fresh seed here shared by both
# Stage6g variants for a fair paired A/B.

VARIANTS = [
    {
        "suffix": "control_100iter",
        "reward_module": "student.my_reward_delta_v1_turnback",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
        },
        "notes": (
            "Stage6g control - 100iter continuation from stage6f_turnback_200iter, "
            "same my_reward_delta_v1_turnback module/params, no wez_proximity term. "
            "Isolates 'more training' from the new term."
        ),
    },
    {
        "suffix": "wez_proximity_100iter",
        "reward_module": "student.my_reward_delta_v1_wez_proximity",
        "reward_overrides": {
            "ata_recovery_scale": 0.15,
            "ata_recovery_threshold_deg": 90.0,
            "wez_proximity_scale": 0.5,
        },
        "notes": (
            "Stage6g treatment - 100iter continuation from stage6f_turnback_200iter "
            "with my_reward_delta_v1_wez_proximity, wez_proximity_scale=0.5 (same "
            "order of magnitude as wez_bonus/attack_range_bonus), dense range x "
            "angle joint-proximity shaping to get wez_episode_rate off zero."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_variant_experiment(base: dict[str, Any], variant: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = f"altitude_attack_followup_v1_stage6g_{variant['suffix']}_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    experiment["env"]["reward_module"] = variant["reward_module"]
    experiment["env_config"]["reward"].update(variant["reward_overrides"])
    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["notes"] = variant["notes"]
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

    for variant in VARIANTS:
        tag, experiment = build_variant_experiment(base, variant)
        yaml_path = CAMPAIGN_DIR / "generated" / f"stage6g_{variant['suffix']}.yaml"
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
        print(f"[stage6g] {tag}: gate_passed={row['gate_passed']} "
              f"wez_episode_rate={row.get('wez_episode_rate')} "
              f"episode_crash_rate={row.get('episode_crash_rate')}")

    if not args.dry_run:
        print(f"[done] Stage6g A/B results written to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
