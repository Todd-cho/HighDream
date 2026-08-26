"""Continue stage6g_turnback_wez_proximity_aa_100iter for 50 more iterations
(cumulative 150iter from stage6f_turnback_200iter) to get a bigger post-warmup
episode sample.

Background: the from-turnback pilot (run_stage6g_wez_proximity_from_turnback_ab.py)
restarted wez_proximity shaping from stage6f_turnback_200iter (0% crash, this
project's safest checkpoint) instead of the fragile stage6g_wez_proximity_100iter
line that collapsed three times in a row. Both restore fixes were active
(--replay-warmup-steps=2000, auto-inferred --initial-alpha). The first 50iter
pilot's post-warmup episodes (5-6) showed variant B (range x ata x aa,
wez_proximity_use_aa=True) going 2/2 safe with one clean WEZ engagement
(wez_steps=8, no crash). The first 50iter continuation (cumulative 100iter)
then went 4/4 safe (crash 0%), closing to ~700m on 2 of 4 episodes, but
wez_episode_rate dropped back to 0.0 (close approaches missed the exact WEZ
angle+range condition). actor_loss stayed bounded (-77 to 64) and alpha kept
decaying smoothly (0.170 -> 0.124) across that run -- no divergence. Sample
size is still tiny (4 post-warmup episodes), so this continuation extends the
same B line by another 50 iterations (rule 13: small increments) to build a
more reliable post-warmup episode window.

Usage: python scripts/run_stage6g_turnback_wez_proximity_aa_continue.py [--dry-run]
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
SOURCE_TAG = "altitude_attack_followup_v1_stage6g_turnback_wez_proximity_aa_100iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6g_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261808  # next unused seed (261807 used by the 50->100iter continuation).

VARIANT = {
    "suffix": "turnback_wez_proximity_aa_150iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.5,
        "wez_proximity_use_aa": True,
    },
    "notes": (
        "50iter continuation from stage6g_turnback_wez_proximity_aa_100iter "
        "(cumulative 150iter from stage6f_turnback_200iter), same reward "
        "params (range x ata x aa joint term). Building a bigger post-warmup "
        "episode sample after the 100iter run went 4/4 safe (crash 0%) with "
        "2 close approaches (~700m) but wez_episode_rate back to 0.0 -- "
        "checking whether more training lands the exact WEZ angle+range "
        "condition without regressing the crash-free run."
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
    print(f"[done] Stage6g turnback_wez_proximity_aa continuation appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
