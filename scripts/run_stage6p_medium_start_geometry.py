"""Stage6p: curriculum step 2 on the STARTING GEOMETRY axis -- continue
stage6o (ata approx 0deg start) up to ata approx 22deg (halfway to the
original close_quarters_750's 45deg), same 749.5m distance, same fixed
opponent difficulty as stage6o/stage6m (heading_to_bank_gain=0.4,
max_bank_deg=25), reward unchanged.

Background: stage6o (fresh from stage6e_delta_down_400iter, ata approx
0deg start, 50iter) broke this project's 5-run scripted_pursuit losing
streak (stage6j/6k/6l/6m/6n all exactly 0.0% wez_episode_rate) with
wez_episode_rate=0.7 in frozen eval (crash 0%, mean_distance 3450m, real
target damage in several episodes). But this is a single seed with no
reproduction check, and 7/20 episodes still showed the "close then drift
away" pattern from stage6n's diagnosis (min_distance ~500m but
mean_distance blowing out to 6000-9800m) -- reduced, not eliminated. Per
user's explicit judgment call, treating a true curriculum continuation
(continuing FROM stage6o's own checkpoint, not fresh-restarting from
stage6e) as a combined next-difficulty-step-and-reproducibility check: if
the skill stage6o's frozen eval showed is real, it should degrade
gracefully rather than collapse outright when the starting angle is
increased partway back toward the original 45deg.

Single-variable relative to stage6o: only the starting geometry changes
(target placed at ata approx 22deg instead of approx 0deg, same 749.5m
distance, same target heading=90deg). Opponent config, reward
module/config, and all other scenario fields are carried over byte-for-byte
from stage6o. Small increment (50iter, rule 13), continuing (not
fresh-restarting) per the curriculum framing, with the usual restore
stability fixes (--replay-warmup-steps=2000, auto-inferred --initial-alpha).

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6p_medium_start_geometry.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
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
    / "stage6o_easy_start_geometry_50iter.yaml"
)
SOURCE_TAG = "altitude_attack_followup_v1_stage6o_easy_start_geometry_50iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6p_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261832  # next unused seed in the 2618xx family (261831 used by stage6o).

_ORIGINAL_DISTANCE_M = math.hypot(530.0, 530.0)  # 749.53..., same as close_quarters_750
_ANGLE_DEG = 22.0  # halfway to the original 45deg

OWNSHIP_START = [1000.0, 0.0, -4500.0, 0.0, 0.0, 0.0, 260.0]
TARGET_START_MEDIUM = [
    1000.0 + _ORIGINAL_DISTANCE_M * math.cos(math.radians(_ANGLE_DEG)),
    0.0 + _ORIGINAL_DISTANCE_M * math.sin(math.radians(_ANGLE_DEG)),
    -4500.0,
    0.0,
    0.0,
    90.0,
    250.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_experiment(base: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = "altitude_attack_followup_v1_stage6p_medium_start_geometry_50iter_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag

    scenario = experiment["env_config"]["initial_scenario"]["scenarios"][0]
    scenario["name"] = "medium_start_750"
    scenario["ownship"] = list(OWNSHIP_START)
    scenario["target"] = list(TARGET_START_MEDIUM)

    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    # initial_alpha deliberately left unset: train_rllib.py auto-infers it
    # from SOURCE_TAG's own training_log.csv (--auto-restore-alpha, default on).
    experiment["notes"] = (
        "50iter continuation from stage6o_easy_start_geometry_50iter "
        "(true curriculum step, not fresh restart): starting angle raised "
        "from approx 0deg to approx 22deg (halfway to close_quarters_750's "
        "original 45deg), same 749.5m distance, opponent/reward unchanged. "
        "Doubles as a reproducibility check on stage6o's single-seed "
        "wez_episode_rate=0.7 frozen-eval result."
    )
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

    tag, experiment = build_experiment(base)
    yaml_path = CAMPAIGN_DIR / "generated" / "stage6p_medium_start_geometry_50iter.yaml"
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
        "variant": "stage6p_medium_start_geometry_50iter",
        "tag": tag,
        "reward_module": experiment["env"]["reward_module"],
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
    print(f"[stage6p] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6p] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6p medium-start-geometry result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
