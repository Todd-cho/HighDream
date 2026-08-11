"""Test max_engage_time=200s (the real competition regulation value) on top of
the wez_hold checkpoint, isolated as a single-variable change.

Background: 2026-08-10/11 analysis of the "approach then flee" pattern
(policy always closes to 350-800m but drifts back out to 14000-46000m by
episode end) found that at sim_hz=60/step_ratio=6 (0.1s/env-step), a
120-step-limited 120-second episode leaves only ~100 seconds after an early
merge for a turn-back re-engagement -- and simply flying straight for that
long at ~130 m/s (250-260kt scenario speed) already accounts for most of the
observed final-distance drift, no active fleeing required.

Opening the actual competition rules deck (`1일차 강의 자료\\2026 AI Pilot Top
Gun Challenge.pptx`, slide 11) confirms max_engage_time=120.0 (used
throughout Stage5/6 training since 2026-08-06's stage6b_realistic_pilot) is
itself already a partial fix -- the real regulation engagement time is
200 seconds ("교전 시간 200초동안 상대에게 더 많은 대미지를 입히거나 상대를
격추 시키면 승리"), with Phase1 (LOS<1 deg, 152.4-914.4m -- exactly matching
this project's wez_config) active as the highest-priority zone for the full
200s (slides 13/14: Phase1 0-100s, Phase2 100-150s, Phase3 150-200s, all
additive not replacing). 200s is the base competition round length, not a
fallback used only when no decision has been reached -- decisive
kills/crashes still end episodes early via this sim's existing termination
logic regardless of the time cap value, so raising the cap only changes
episodes that are still undecided, which is exactly the population this
pattern lives in.

This restarts from turnback_wez_hold_50iter (the current most-promising
checkpoint: crash 15% frozen/20ep, first run in this whole Stage6g line to
show materially better and more consistent final_ata_deg, 29-62deg vs the
100-170deg typical of earlier checkpoints) with the exact same reward config
(wez_hold_scale=0.4, wez_hold_break_penalty_scale=0.3, wez_proximity_scale=0.5,
wez_proximity_use_aa=True, ata_recovery_scale=0.15) -- only
env_config.max_engage_time changes, 120.0 -> 200.0. episode_step_limit
(3600) already comfortably covers 200s (=2000 steps at 0.1s/step), so it
does not need adjusting. Both restore fixes apply automatically
(--replay-warmup-steps=2000, auto-inferred --initial-alpha).

Usage: python scripts/run_stage6g_turnback_wez_hold_engage200.py [--dry-run]
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
SEED = 261811  # next unused seed (261810 used by turnback_wez_hold_50iter).
MAX_ENGAGE_TIME_S = 200.0  # real competition regulation value (slide 11), was 120.0

VARIANT = {
    "suffix": "turnback_wez_hold_engage200_50iter",
    "reward_module": "student.my_reward_delta_v1_wez_proximity",
    "reward_overrides": {
        "ata_recovery_scale": 0.15,
        "ata_recovery_threshold_deg": 90.0,
        "wez_proximity_scale": 0.5,
        "wez_proximity_use_aa": True,
        "wez_hold_scale": 0.4,
        "wez_hold_break_penalty_scale": 0.3,
    },
    "notes": (
        "50iter continuation from turnback_wez_hold_50iter, identical reward "
        "config, single-variable change: max_engage_time 120.0 -> 200.0 (the "
        "real competition regulation value confirmed from the original rules "
        "deck, slide 11). Tests whether the diagnosed 'approach then flee' "
        "pattern (0% wez_episode_rate despite reliably closing to 350-800m) "
        "is partly an artifact of the training episode being cut ~80 seconds "
        "shorter than the real engagement window, leaving too little time "
        "post-merge for a safe turn-back re-engagement."
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
    print(f"[done] Stage6g engage200 result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
