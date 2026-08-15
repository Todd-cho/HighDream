"""Stage6r: CONTROL experiment -- continue stage6o for 50 more iterations at
the SAME ata approx 0deg starting geometry (no change at all), to isolate
whether stage6p/stage6q's collapse (wez_episode_rate 0.7 -> 0.0-0.05) was
caused by the angle change itself or by the restore/continuation process
alone.

Background: stage6o (fresh from stage6e_delta_down_400iter, ata approx
0deg, 50iter) got wez_episode_rate=0.7 in frozen eval. Two follow-up
continuations from stage6o's own checkpoint both collapsed back to
near-zero: stage6p (angle raised to approx 22deg) -> wez_episode_rate=0.0,
stage6q (angle raised to only approx 8deg, a much smaller step) ->
wez_episode_rate=0.05. Both showed the same "flee to 15-37km, heavy
safety-override intervention" pattern as stage6n. Since BOTH attempts were
also 50iter *continuations* (restores) from stage6o, this project's
established replay-buffer-empties-on-restore instability (rule 14; see
project memory's 2026-08-10 root-cause writeup, and repeated real
collapses: stage6f_turnback 200->400iter, stage6g wez_proximity
100->150iter) is a confound that hasn't been ruled out -- it's possible
ANY continuation from stage6o's checkpoint destabilizes regardless of
whether the angle changes, and the two prior results say nothing
angle-specific at all.

This control changes NOTHING about the scenario (same exact ata approx
0deg placement as stage6o) -- only continues training 50 more iterations
from stage6o's own checkpoint, same opponent/reward, next seed in
sequence. If wez_episode_rate holds near stage6o's 0.7, that proves
stage6p/6q's collapse really was angle-sensitivity (stage6o's skill doesn't
generalize even slightly off 0deg). If it ALSO collapses here with the
angle completely unchanged, that proves the continuation/restore process
itself is the destabilizer, independent of geometry.

IMPORTANT (rule 15): this script's own printed gate_metrics are computed
from *training* (exploration-noise) episodes -- rough pilot signal only. A
frozen, deterministic re-evaluation (adhoc_scripted_pursuit_eval_stage6j.py
--tag <this tag>) is required before drawing any conclusion.

Usage: python scripts/run_stage6r_control_same_angle_continue.py [--dry-run]
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
    / "stage6o_easy_start_geometry_50iter.yaml"
)
SOURCE_TAG = "altitude_attack_followup_v1_stage6o_easy_start_geometry_50iter_C10"
CAMPAIGN_DIR = ROOT / "artifacts" / "altitude_attack_followup_v1"
RESULT_PATH = CAMPAIGN_DIR / "stage6r_ab_results.csv"

ITERATIONS = 50
WARMUP_STEPS = 2000
SEED = 261834  # next unused seed in the 2618xx family (261833 used by stage6q).


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-window", type=int, default=ITERATIONS)
    return parser.parse_args()


def build_experiment(base: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = "altitude_attack_followup_v1_stage6r_control_same_angle_50iter_C10"
    experiment = copy.deepcopy(base)
    experiment["output"]["tag"] = tag
    # Scenario deliberately left untouched -- identical ata≈0deg placement as stage6o.

    experiment["runtime"]["iterations"] = ITERATIONS
    experiment["runtime"]["seed"] = SEED
    experiment["runtime"]["restore_checkpoint"] = str(checkpoint_final(SOURCE_TAG))
    experiment["runtime"]["replay_warmup_steps"] = WARMUP_STEPS
    # initial_alpha deliberately left unset: train_rllib.py auto-infers it
    # from SOURCE_TAG's own training_log.csv (--auto-restore-alpha, default on).
    experiment["notes"] = (
        "CONTROL: 50iter continuation from stage6o_easy_start_geometry_50iter "
        "with the starting geometry completely UNCHANGED (same ata approx "
        "0deg placement) -- isolates whether stage6p/6q's collapse to near-zero "
        "wez_episode_rate was caused by the angle change or by the "
        "restore/continuation process itself."
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
    yaml_path = CAMPAIGN_DIR / "generated" / "stage6r_control_same_angle_50iter.yaml"
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
        "variant": "stage6r_control_same_angle_50iter",
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
    print(f"[stage6r] {tag}: gate_passed={row['gate_passed']} "
          f"wez_episode_rate={row.get('wez_episode_rate')} "
          f"episode_crash_rate={row.get('episode_crash_rate')}")
    print("[stage6r] NOTE: these are training-exploration episodes, not a "
          "frozen eval -- rule 15 applies, re-verify with "
          "adhoc_scripted_pursuit_eval_stage6j.py --tag before concluding "
          "anything.")
    print(f"[done] Stage6r control (same-angle continue) result appended to {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
