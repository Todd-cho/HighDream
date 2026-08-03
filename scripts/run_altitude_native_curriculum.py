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

from scripts.run_altitude_reward_search import (
    build_experiment,
    dashboard_path,
    load_yaml,
    number,
    summarize,
    write_yaml,
)
from student.altitude_safety_pipeline_curriculum import (
    CANDIDATES,
    scenario_pool,
)


RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

STAGES = [
    {
        "index": 0,
        "name": "level_hold",
        "scenario": "level_5000",
        "iterations": 200,
        "crash_max": 0.20,
        "min_altitude_mean_min": 3000.0,
        "jitter": False,
    },
    {
        "index": 1,
        "name": "light_recovery",
        "scenario": "light_disturbance_4500",
        "iterations": 150,
        "crash_max": 0.25,
        "min_altitude_mean_min": 2500.0,
        "jitter": True,
    },
    {
        "index": 2,
        "name": "mild_dive_recovery",
        "scenario": "mild_dive_4000",
        "iterations": 180,
        "crash_max": 0.30,
        "min_altitude_mean_min": 2000.0,
        "jitter": True,
    },
    {
        "index": 3,
        "name": "low_dive_recovery",
        "scenario": "low_dive_3500",
        "iterations": 220,
        "crash_max": 0.35,
        "min_altitude_mean_min": 1500.0,
        "jitter": True,
    },
    {
        "index": 4,
        "name": "banked_descent_recovery",
        "scenario": "banked_descent_3800",
        "iterations": 260,
        "crash_max": 0.40,
        "min_altitude_mean_min": 1200.0,
        "jitter": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh SAC policy with native-checkpoint stage transfer "
            "and strict altitude-safety gates."
        )
    )
    parser.add_argument(
        "--campaign",
        default="altitude_native_curriculum_v1",
    )
    parser.add_argument(
        "--base",
        default="experiments/altitude_reward_search_base.yaml",
    )
    parser.add_argument("--screen-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=261100)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--checkpoint-frequency", type=int, default=25)
    parser.add_argument("--min-episodes", type=int, default=5)
    parser.add_argument(
        "--evaluation-window",
        type=int,
        default=10,
        help="Use only the most recent completed episodes for safety gates.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def experiment_for(
    base: dict[str, Any],
    *,
    tag: str,
    iterations: int,
    seed: int,
    lr: float,
    reward: dict[str, float],
    scenario_name: str,
    jitter: bool,
    checkpoint_frequency: int,
    restore_checkpoint: str = "",
) -> dict[str, Any]:
    experiment = build_experiment(
        base,
        tag,
        iterations,
        reward,
        validation=False,
    )
    experiment.setdefault("env", {})["target_mode"] = "loiter"
    experiment.setdefault("env_config", {})["initial_scenario"] = scenario_pool(
        scenario_name,
        jitter,
    )
    experiment.setdefault("algo", {})["lr"] = lr
    runtime = experiment.setdefault("runtime", {})
    runtime.update({
        "seed": seed,
        "save_native_checkpoint": True,
        "native_checkpoint_frequency": checkpoint_frequency,
        "save_lightweight_bundle": True,
        "lightweight_bundle_frequency": 0,
    })
    runtime.pop("init_bundle", None)
    runtime.pop("restart_from_bundle", None)
    if restore_checkpoint:
        runtime["restore_checkpoint"] = restore_checkpoint
    else:
        runtime.pop("restore_checkpoint", None)
    experiment.setdefault("engagement_log", {})["enabled"] = False
    experiment["notes"] = (
        "Fresh SAC altitude curriculum with full native-checkpoint transfer."
    )
    return experiment


def execute_yaml(yaml_path: Path, dry_run: bool) -> int:
    command = [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)]
    if dry_run:
        command.append("--dry-run")
    print(f"[run] {' '.join(command)}")
    creationflags = (
        BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" and not dry_run else 0
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        creationflags=creationflags,
    )
    return completed.returncode


def episode_metrics(tag: str, evaluation_window: int) -> dict[str, Any]:
    run_dir = ROOT / "artifacts" / "logs" / "highdream" / tag
    rows: list[dict[str, str]] = []
    for path in sorted(run_dir.glob("episode_summary*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    all_altitudes = [
        value
        for row in rows
        if (value := number(row.get("minimum_altitude_m"))) is not None
    ]
    evaluation_rows = rows[-evaluation_window:]
    altitudes = [
        value
        for row in evaluation_rows
        if (value := number(row.get("minimum_altitude_m"))) is not None
    ]
    crashes = sum(
        row.get("outcome") == "crash" for row in evaluation_rows
    )
    all_crashes = sum(row.get("outcome") == "crash" for row in rows)
    return {
        "episodes": len(rows),
        "evaluation_episodes": len(evaluation_rows),
        "episode_crash_rate": (
            crashes / len(evaluation_rows) if evaluation_rows else None
        ),
        "minimum_altitude_mean_m": mean(altitudes) if altitudes else None,
        "minimum_altitude_worst_m": min(altitudes) if altitudes else None,
        "full_episode_crash_rate": (
            all_crashes / len(rows) if rows else None
        ),
        "full_minimum_altitude_mean_m": (
            mean(all_altitudes) if all_altitudes else None
        ),
        "nose_down_crashes": sum(
            row.get("primary_cause") == "altitude_nose_down_crash"
            for row in evaluation_rows
        ),
        "roll_instability_crashes": sum(
            row.get("primary_cause") == "altitude_roll_instability_crash"
            for row in evaluation_rows
        ),
    }


def checkpoint_final(tag: str) -> Path:
    return (
        ROOT / "artifacts" / "checkpoints" / "highdream" / tag
        / "checkpoint_final"
    )


def model_bundle(tag: str) -> Path:
    return ROOT / "artifacts" / "models" / "highdream" / tag


def passed_gate(
    metrics: dict[str, Any],
    *,
    min_episodes: int,
    crash_max: float,
    min_altitude_mean_min: float,
) -> bool:
    crash = number(metrics.get("episode_crash_rate"))
    altitude = number(metrics.get("minimum_altitude_mean_m"))
    return bool(
        metrics.get("evaluation_episodes", 0) >= min_episodes
        and crash is not None
        and crash <= crash_max
        and altitude is not None
        and altitude >= min_altitude_mean_min
    )


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if min(
        args.screen_iterations,
        args.checkpoint_frequency,
        args.min_episodes,
        args.evaluation_window,
    ) < 1:
        raise ValueError("iteration, checkpoint, and episode values must be positive")
    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "pipeline_state.json"
    state = {
        "campaign": args.campaign,
        "status": "screening",
        "selected_candidate": "",
        "completed_stage": -1,
        "stop_reason": "",
    }
    if args.resume and state_path.exists():
        state.update(json.loads(state_path.read_text(encoding="utf-8")))

    screen_rows = (
        list(csv.DictReader(
            (work / "screen_results.csv").open(
                "r", encoding="utf-8-sig", newline=""
            )
        ))
        if args.resume and (work / "screen_results.csv").exists()
        else []
    )
    completed_candidates = {
        row["candidate"] for row in screen_rows if row.get("status") == "success"
    }
    for offset, (candidate, reward) in enumerate(CANDIDATES.items()):
        if candidate in completed_candidates:
            print(f"[resume] screen candidate {candidate} already complete")
            continue
        tag = f"{args.campaign}_screen_{candidate}_{args.screen_iterations}iter"
        experiment = experiment_for(
            base,
            tag=tag,
            iterations=args.screen_iterations,
            seed=args.seed,
            lr=args.lr,
            reward=reward,
            scenario_name="level_5000",
            jitter=False,
            checkpoint_frequency=args.checkpoint_frequency,
        )
        yaml_path = work / "generated" / f"screen_{candidate}.yaml"
        write_yaml(yaml_path, experiment)
        code = execute_yaml(yaml_path, args.dry_run)
        if args.dry_run:
            continue
        metrics = {
            **summarize(dashboard_path(experiment)),
            **episode_metrics(tag, args.evaluation_window),
        }
        row = {
            "candidate": candidate,
            "status": "success" if code == 0 else "failed",
            "return_code": code,
            "tag": tag,
            **reward,
            **metrics,
            "passed": passed_gate(
                metrics,
                min_episodes=args.min_episodes,
                crash_max=STAGES[0]["crash_max"],
                min_altitude_mean_min=STAGES[0]["min_altitude_mean_min"],
            ),
            "checkpoint_final": str(checkpoint_final(tag)),
        }
        screen_rows = [
            old for old in screen_rows if old.get("candidate") != candidate
        ]
        screen_rows.append(row)
        save_rows(work / "screen_results.csv", screen_rows)

    # Older runs used all training episodes for the gate, which unfairly
    # included the initial random-policy crashes.  Always recompute completed
    # screen rows from the latest evaluation window before selecting a model.
    if not args.dry_run:
        refreshed_rows: list[dict[str, Any]] = []
        for row in screen_rows:
            if row.get("status") != "success":
                refreshed_rows.append(row)
                continue
            metrics = episode_metrics(
                str(row["tag"]),
                args.evaluation_window,
            )
            row.update(metrics)
            row["passed"] = passed_gate(
                metrics,
                min_episodes=args.min_episodes,
                crash_max=STAGES[0]["crash_max"],
                min_altitude_mean_min=STAGES[0]["min_altitude_mean_min"],
            )
            refreshed_rows.append(row)
        screen_rows = refreshed_rows
        save_rows(work / "screen_results.csv", screen_rows)

    if args.dry_run:
        print("[dry-run] screen commands generated; later stages require real results.")
        return 0

    eligible = [
        row for row in screen_rows
        if str(row.get("status")) == "success"
        and str(row.get("passed")).lower() == "true"
        and Path(row["checkpoint_final"]).exists()
    ]
    if not eligible:
        state.update({
            "status": "stopped",
            "stop_reason": (
                "No fresh-from-scratch candidate passed level_hold. "
                "Do not advance to harder scenarios."
            ),
        })
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[STOP] {state['stop_reason']}")
        print(f"[results] {work / 'screen_results.csv'}")
        return 2

    eligible.sort(key=lambda row: (
        float(row["episode_crash_rate"]),
        -float(row["minimum_altitude_mean_m"]),
        float(row.get("selection_score") or 1e9),
    ))
    selected_row = eligible[0]
    selected = selected_row["candidate"]
    state.update({
        "status": "curriculum",
        "selected_candidate": selected,
        "completed_stage": 0,
        "stop_reason": "",
    })
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[selected] {selected} from native checkpoint")

    previous_checkpoint = Path(selected_row["checkpoint_final"])
    stage_rows: list[dict[str, Any]] = []
    stage_results_path = work / "stage_results.csv"
    if args.resume and stage_results_path.exists():
        stage_rows = list(csv.DictReader(
            stage_results_path.open("r", encoding="utf-8-sig", newline="")
        ))
        completed = [
            row for row in stage_rows
            if str(row.get("passed")).lower() == "true"
        ]
        if completed:
            last = max(completed, key=lambda row: int(row["stage"]))
            previous_checkpoint = Path(last["checkpoint_final"])

    for stage in STAGES[1:]:
        index = int(stage["index"])
        existing = [
            row for row in stage_rows
            if int(row["stage"]) == index
            and str(row.get("passed")).lower() == "true"
        ]
        if existing:
            print(f"[resume] stage {index} already passed")
            previous_checkpoint = Path(existing[0]["checkpoint_final"])
            continue
        tag = f"{args.campaign}_stage{index}_{stage['name']}_{selected}"
        experiment = experiment_for(
            base,
            tag=tag,
            iterations=int(stage["iterations"]),
            seed=args.seed + index,
            lr=args.lr,
            reward=CANDIDATES[selected],
            scenario_name=str(stage["scenario"]),
            jitter=bool(stage["jitter"]),
            checkpoint_frequency=args.checkpoint_frequency,
            restore_checkpoint=str(previous_checkpoint),
        )
        yaml_path = work / "generated" / f"stage{index}_{stage['name']}.yaml"
        write_yaml(yaml_path, experiment)
        code = execute_yaml(yaml_path, False)
        metrics = {
            **summarize(dashboard_path(experiment)),
            **episode_metrics(tag, args.evaluation_window),
        }
        passed = passed_gate(
            metrics,
            min_episodes=args.min_episodes,
            crash_max=float(stage["crash_max"]),
            min_altitude_mean_min=float(stage["min_altitude_mean_min"]),
        )
        row = {
            "stage": index,
            "name": stage["name"],
            "scenario": stage["scenario"],
            "candidate": selected,
            "status": "success" if code == 0 else "failed",
            **metrics,
            "crash_gate": stage["crash_max"],
            "altitude_gate_m": stage["min_altitude_mean_min"],
            "passed": passed,
            "checkpoint_final": str(checkpoint_final(tag)),
            "model_bundle": str(model_bundle(tag)),
        }
        stage_rows = [
            old for old in stage_rows if int(old["stage"]) != index
        ]
        stage_rows.append(row)
        stage_rows.sort(key=lambda value: int(value["stage"]))
        save_rows(stage_results_path, stage_rows)
        if code != 0 or not passed:
            state.update({
                "status": "stopped",
                "completed_stage": index - 1,
                "stop_reason": (
                    f"Stage {index} {stage['name']} did not pass the safety gate."
                ),
            })
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[STOP] {state['stop_reason']}")
            return 2
        previous_checkpoint = checkpoint_final(tag)
        state["completed_stage"] = index
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    state.update({
        "status": "completed",
        "completed_stage": 4,
        "final_checkpoint": str(previous_checkpoint),
        "stop_reason": "",
    })
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] native altitude curriculum completed: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
