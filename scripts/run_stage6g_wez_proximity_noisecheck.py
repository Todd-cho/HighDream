"""Stage6g noise-check: is the late-window crash uptick in
stage6g_wez_proximity_100iter (episodes 9-10 of 10, crash 40% overall) a real
degrading trend, or small-sample noise?

Background: stage6g_wez_proximity_100iter got wez_episode_rate off zero for
the first time in this whole Stage6f/6g series (0.10, matching the gate
threshold) and produced the series' first real WEZ engagement with actual
target damage (episode 7). But crash rate also rose versus the
stage6f_turnback_200iter source checkpoint (0% -> 40%), and episodes 9-10
crashed after a clean 3-8 stretch. With only 10 episodes total this could be
sampling variance (SAC's stochastic exploration/scenario draws), or it could
be the start of the same kind of collapse seen in stage6f_turnback_400iter
(crash 0% -> 65% over a 200iter continuation despite the same delta-reward
altitude gate being present, cause still unresolved).

Per [[feedback-aip-experiment-rules]] rule 13, this pilot extends by a small
increment (50iter, not 200) rather than assuming safety, so the run can be
stopped early / the outcome re-examined without burning a large compute
budget on a possible second collapse. After the run, inspect the FULL
episode_summary.csv timeline in order (not just the final-window aggregate)
-- if crash keeps climbing through the new episodes, that's signal; if it
stays scattered/low like episodes 3-8 did, the episode 9-10 crashes were
noise.

Single variant only (wez_proximity, no fresh control) -- this is a
within-run stability question about one trajectory, not a comparative A/B.

Usage: python scripts/run_stage6g_wez_proximity_noisecheck.py [--dry-run]
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6g_wez_proximity_100iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 50
SEED = 261804  # stage6g control/wez_proximity 100iter pair used 261803; +1 here.

VARIANT = {
    "suffix": "wez_proximity_150iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.5,
    },
    "notes": (
        "Stage6g noise-check - 50iter continuation from "
        "stage6g_wez_proximity_100iter (cumulative 150iter), same reward "
        "params, checking whether the episode 9-10 crash uptick was noise "
        "or a real degrading trend."
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
    print(f"[done] Stage6g noise-check result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
