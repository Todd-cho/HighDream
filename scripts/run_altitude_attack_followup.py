from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.run_altitude_native_curriculum import (
    BELOW_NORMAL_PRIORITY_CLASS,
    checkpoint_final,
    episode_metrics,
    experiment_for,
    model_bundle,
    save_rows,
)
from scripts.run_altitude_reward_search import load_yaml, number, write_yaml
from student.altitude_safety_pipeline_curriculum import (
    C10,
    SCENARIOS,
)


RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
EVALUATOR = ROOT / "scripts" / "evaluate_altitude_bundle.py"
SOURCE_TAG = "altitude_native_curriculum_v1_stage4_banked_descent_recovery_C10"
SOURCE_BUNDLE = ROOT / "artifacts" / "models" / "highdream" / SOURCE_TAG
SOURCE_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints" / "highdream" / SOURCE_TAG
    / "checkpoint_final"
)


ATTACK_STAGES = [
    {
        "index": 5,
        "name": "safe_approach",
        "iterations": 180,
        "reward": {
            "attack_range_max_m": 2500.0,
            # 2026-08-04: bumped from 0.30/0.35 (weak placeholders that failed the
            # distance gate at 11977m) to the values already validated in the
            # ata_scale bisection (stage5_ata015_150iter etc.: crash 5%, dist 8913m
            # on this exact mixed scenario pool).
            "attack_range_bonus": 0.9,
            "far_range_penalty_start_m": 4000.0,
            "far_range_penalty": 1.05,
            "ata_scale": 0.15,
        },
        "crash_max": 0.20,
        "altitude_min": 1800.0,
        "mean_distance_max": 9000.0,
        "wez_episode_rate_min": 0.0,
    },
    {
        "index": 6,
        "name": "safe_wez_geometry",
        "iterations": 240,
        "reward": {
            "attack_range_max_m": 2200.0,
            "attack_range_bonus": 0.45,
            "far_range_penalty_start_m": 3500.0,
            "far_range_penalty": 0.45,
            "ata_scale": 0.15,
            "aa_scale": 0.04,
            "wez_bonus": 1.0,
        },
        "crash_max": 0.25,
        "altitude_min": 1500.0,
        "mean_distance_max": 7500.0,
        "wez_episode_rate_min": 0.10,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the final altitude model, then add safe approach/WEZ skills."
    )
    parser.add_argument(
        "--campaign",
        default="altitude_attack_followup_v1",
    )
    parser.add_argument(
        "--base",
        default="experiments/altitude_reward_search_base.yaml",
    )
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--evaluation-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=261500)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--checkpoint-frequency", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def mixed_scenario_pool() -> dict[str, Any]:
    scenarios = []
    for name, source in SCENARIOS.items():
        scenario = copy.deepcopy(source)
        scenario.update({
            "weight": 1.0,
            "target_mode": "loiter",
            "target_loiter": {"enabled": True, "bank": 0.0, "pitch": 0.0},
        })
        scenarios.append(scenario)
    return {
        "mode": "scenario_pool",
        "scenarios": scenarios,
        "ownship_randomization": {
            "enabled": True,
            "radius": 40.0,
            "r_roll": 2.0,
            "r_pitch": 1.0,
            "r_heading": 3.0,
            "speed_mps": 3.0,
        },
        "target_randomization": {
            "enabled": True,
            "radius": 40.0,
            "r_heading": 2.0,
            "speed_mps": 2.0,
        },
    }


def execute(command: list[str], dry_run: bool = False) -> int:
    print(f"[run] {' '.join(command)}")
    if dry_run and "--dry-run" not in command:
        command = [*command, "--dry-run"]
    creationflags = (
        BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" and not dry_run else 0
    )
    return subprocess.run(
        command,
        cwd=ROOT,
        creationflags=creationflags,
    ).returncode


def run_frozen_validation(args: argparse.Namespace, work: Path) -> bool:
    output = work / "01_frozen_mixed_validation"
    report = output / "frozen_evaluation_summary.csv"
    if not (args.resume and report.exists()):
        command = [
            sys.executable,
            str(EVALUATOR),
            "--bundle",
            str(SOURCE_BUNDLE),
            "--output-dir",
            str(output),
            "--episodes-per-scenario",
            str(args.validation_episodes),
            "--seed",
            str(args.seed),
            "--jitter",
        ]
        if args.dry_run:
            print(f"[dry-run frozen] {' '.join(command)}")
            return True
        if execute(command) != 0:
            return False
    rows = list(csv.DictReader(
        report.open("r", encoding="utf-8-sig", newline="")
    ))
    limits = {
        "level_5000": (0.20, 3000.0),
        "light_disturbance_4500": (0.20, 2500.0),
        "mild_dive_4000": (0.25, 2000.0),
        "low_dive_3500": (0.30, 1500.0),
        "banked_descent_3800": (0.30, 1500.0),
    }
    passed = True
    for row in rows:
        crash_max, altitude_min = limits[row["scenario"]]
        row_passed = (
            float(row["crash_rate"]) <= crash_max
            and float(row["minimum_altitude_mean_m"]) >= altitude_min
        )
        row["crash_gate"] = crash_max
        row["altitude_gate_m"] = altitude_min
        row["passed"] = row_passed
        passed = passed and row_passed
    save_rows(output / "frozen_safety_gate.csv", rows)
    return passed


def attack_window_metrics(tag: str, window: int) -> dict[str, Any]:
    base = episode_metrics(tag, window)
    run_dir = ROOT / "artifacts" / "logs" / "highdream" / tag
    rows: list[dict[str, str]] = []
    for path in sorted(run_dir.glob("episode_summary*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    evaluation = rows[-window:]
    distances = [
        value for row in evaluation
        if (value := number(row.get("mean_distance_m"))) is not None
    ]
    wez_steps = [
        value for row in evaluation
        if (value := number(row.get("wez_steps"))) is not None
    ]
    base.update({
        "mean_distance_m": mean(distances) if distances else None,
        "wez_steps_mean": mean(wez_steps) if wez_steps else None,
        "wez_episode_rate": (
            sum(value > 0.0 for value in wez_steps) / len(wez_steps)
            if wez_steps else None
        ),
    })
    return base


def passed_attack_gate(metrics: dict[str, Any], stage: dict[str, Any]) -> bool:
    return bool(
        metrics.get("evaluation_episodes", 0) >= 10
        and number(metrics.get("episode_crash_rate")) is not None
        and float(metrics["episode_crash_rate"]) <= stage["crash_max"]
        and number(metrics.get("minimum_altitude_mean_m")) is not None
        and float(metrics["minimum_altitude_mean_m"]) >= stage["altitude_min"]
        and number(metrics.get("mean_distance_m")) is not None
        and float(metrics["mean_distance_m"]) <= stage["mean_distance_max"]
        and number(metrics.get("wez_episode_rate")) is not None
        and float(metrics["wez_episode_rate"]) >= stage["wez_episode_rate_min"]
    )


def main() -> int:
    args = parse_args()
    if min(
        args.validation_episodes,
        args.evaluation_window,
        args.checkpoint_frequency,
    ) < 1:
        raise ValueError("episode, window, and checkpoint values must be positive")
    if not SOURCE_BUNDLE.exists() or not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError("Final C10 Stage 4 model/checkpoint is missing")
    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "pipeline_state.json"
    state = {
        "status": "validating",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "completed_stage": 4,
        "stop_reason": "",
    }
    if args.resume and state_path.exists():
        state.update(json.loads(state_path.read_text(encoding="utf-8")))

    if not run_frozen_validation(args, work):
        state.update({
            "status": "stopped",
            "stop_reason": "Final altitude model failed mixed frozen safety validation.",
        })
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[STOP] {state['stop_reason']}")
        return 2
    if args.dry_run:
        print("[dry-run] frozen validation and attack stages configured")
        return 0

    previous_checkpoint = SOURCE_CHECKPOINT
    result_path = work / "attack_stage_results.csv"
    results = (
        list(csv.DictReader(result_path.open("r", encoding="utf-8-sig", newline="")))
        if args.resume and result_path.exists() else []
    )
    for stage in ATTACK_STAGES:
        existing = [
            row for row in results
            if int(row["stage"]) == stage["index"]
            and str(row.get("passed")).lower() == "true"
        ]
        if existing:
            previous_checkpoint = Path(existing[0]["checkpoint_final"])
            print(f"[resume] stage {stage['index']} already passed")
            continue
        reward = dict(C10)
        reward.update(stage["reward"])
        tag = f"{args.campaign}_stage{stage['index']}_{stage['name']}_C10"
        experiment = experiment_for(
            base,
            tag=tag,
            iterations=stage["iterations"],
            seed=args.seed + stage["index"],
            lr=args.lr,
            reward=reward,
            scenario_name="level_5000",
            jitter=True,
            checkpoint_frequency=args.checkpoint_frequency,
            restore_checkpoint=str(previous_checkpoint),
        )
        experiment["env_config"]["initial_scenario"] = mixed_scenario_pool()
        experiment["notes"] = (
            "Safe attack follow-up: keep C10 altitude reward while reducing range."
        )
        yaml_path = work / "generated" / f"stage{stage['index']}_{stage['name']}.yaml"
        write_yaml(yaml_path, experiment)
        code = execute([sys.executable, str(RUN_EXPERIMENT), str(yaml_path)])
        metrics = attack_window_metrics(tag, args.evaluation_window)
        passed = code == 0 and passed_attack_gate(metrics, stage)
        row = {
            "stage": stage["index"],
            "name": stage["name"],
            "status": "success" if code == 0 else "failed",
            **metrics,
            "crash_gate": stage["crash_max"],
            "altitude_gate_m": stage["altitude_min"],
            "mean_distance_gate_m": stage["mean_distance_max"],
            "wez_episode_rate_gate": stage["wez_episode_rate_min"],
            "passed": passed,
            "checkpoint_final": str(checkpoint_final(tag)),
            "model_bundle": str(model_bundle(tag)),
        }
        results = [old for old in results if int(old["stage"]) != stage["index"]]
        results.append(row)
        results.sort(key=lambda value: int(value["stage"]))
        save_rows(result_path, results)
        if not passed:
            state.update({
                "status": "stopped",
                "stop_reason": f"Attack stage {stage['index']} failed its safety/engagement gate.",
            })
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"[STOP] {state['stop_reason']}")
            return 2
        previous_checkpoint = checkpoint_final(tag)
        state["completed_stage"] = stage["index"]
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    state.update({
        "status": "completed",
        "final_checkpoint": str(previous_checkpoint),
        "stop_reason": "",
    })
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[done] safe attack follow-up completed: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
