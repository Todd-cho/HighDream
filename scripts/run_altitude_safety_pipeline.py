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
    PARAMETER_KEYS,
    add_episode_summary_metrics,
    build_experiment,
    dashboard_path,
    load_yaml,
    number,
    read_csv,
    run_experiment,
    summarize,
    write_csv,
    write_validation_summary,
    write_yaml,
)
from student.altitude_safety_pipeline_curriculum import (
    CANDIDATES,
    scenario_pool,
)


R08_BUNDLE = (
    ROOT / "artifacts" / "models" / "highdream"
    / "altitude_c05_refine_v1_validation_R08_r1_150iter"
)
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
FROZEN_EVALUATOR = ROOT / "scripts" / "evaluate_altitude_bundle.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end altitude safety evaluation and curriculum pipeline."
    )
    parser.add_argument(
        "--campaign",
        default="altitude_safety_pipeline_v1",
    )
    parser.add_argument(
        "--base",
        default="experiments/altitude_reward_search_base.yaml",
    )
    parser.add_argument("--frozen-episodes", type=int, default=5)
    parser.add_argument("--easy-iterations", type=int, default=100)
    parser.add_argument("--easy-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260900)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_frozen_evaluation(args: argparse.Namespace, work: Path) -> None:
    output_dir = work / "01_frozen_r08"
    report = output_dir / "frozen_evaluation_summary.csv"
    if args.resume and report.exists():
        print(f"[resume] frozen R08 evaluation already complete: {report}")
        return
    command = [
        sys.executable,
        str(FROZEN_EVALUATOR),
        "--bundle",
        str(R08_BUNDLE),
        "--output-dir",
        str(output_dir),
        "--episodes-per-scenario",
        str(args.frozen_episodes),
        "--seed",
        str(args.seed),
    ]
    print(f"[stage 1] {' '.join(command)}")
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def easy_experiment(
    base: dict[str, Any],
    tag: str,
    iterations: int,
    seed: int,
    lr: float,
    reward: dict[str, float],
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
        "level_5000",
        False,
    )
    experiment.setdefault("algo", {})["lr"] = lr
    runtime = experiment.setdefault("runtime", {})
    runtime["seed"] = seed
    runtime["init_bundle"] = str(R08_BUNDLE.relative_to(ROOT))
    experiment.setdefault("engagement_log", {})["enabled"] = False
    experiment["notes"] = (
        "Altitude pipeline easy level-hold comparison from frozen R08 actor."
    )
    return experiment


def run_easy_comparison(
    args: argparse.Namespace,
    base: dict[str, Any],
    work: Path,
) -> tuple[str, dict[str, Any] | None]:
    stage_dir = work / "02_easy_comparison"
    results_path = stage_dir / "easy_results.csv"
    existing = read_csv(results_path) if args.resume else []
    completed = {
        row["trial"] for row in existing if row.get("status") == "success"
    }
    results: list[dict[str, Any]] = list(existing)
    trial_number = 0
    for repeat in range(1, args.easy_repeats + 1):
        repeat_seed = args.seed + 100 + repeat - 1
        for candidate, reward in CANDIDATES.items():
            trial_number += 1
            trial = f"{trial_number:03d}"
            if trial in completed:
                print(f"[resume] easy comparison {trial} already complete")
                continue
            tag = (
                f"{args.campaign}_easy_{candidate}_r{repeat}_"
                f"seed{repeat_seed}_{args.easy_iterations}iter"
            )
            yaml_path = stage_dir / "generated" / f"{trial}_{candidate}_r{repeat}.yaml"
            experiment = easy_experiment(
                base,
                tag,
                args.easy_iterations,
                repeat_seed,
                args.lr,
                reward,
            )
            write_yaml(yaml_path, experiment)
            code = run_experiment(
                yaml_path,
                stage_dir / "logs" / f"{trial}.log",
                args.dry_run,
            )
            if args.dry_run:
                continue
            metrics = add_episode_summary_metrics(
                summarize(dashboard_path(experiment)),
                experiment,
            )
            row = {
                "trial": trial,
                "candidate": candidate,
                "repeat": repeat,
                "label": f"easy_level_{candidate}",
                "tag": tag,
                "status": "success" if code == 0 else "failed",
                "return_code": code,
                **reward,
                **metrics,
            }
            results = [old for old in results if old.get("trial") != trial]
            results.append(row)
            results.sort(key=lambda item: int(item["trial"]))
            write_csv(results_path, results)

    if args.dry_run:
        return "R08", None
    write_validation_summary(stage_dir / "easy_validation_summary.csv", results)
    successful = [row for row in results if row.get("status") == "success"]
    candidate_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        group = [row for row in successful if row["candidate"] == candidate]
        if not group:
            continue
        crash_values = [
            value for row in group
            if (value := number(row.get("crash_rate"))) is not None
        ]
        altitude_values = [
            value for row in group
            if (value := number(row.get("minimum_altitude_mean_m"))) is not None
        ]
        score_values = [
            value for row in group
            if (value := number(row.get("selection_score"))) is not None
        ]
        crash_mean = mean(crash_values) if crash_values else 1.0
        altitude_mean = mean(altitude_values) if altitude_values else 0.0
        candidate_rows.append({
            "candidate": candidate,
            "successful_repeats": len(group),
            "crash_rate_mean": crash_mean,
            "minimum_altitude_mean_m": altitude_mean,
            "selection_score_mean": mean(score_values) if score_values else 1e9,
            "passed": crash_mean <= 0.20 and altitude_mean >= 3000.0,
        })
    candidate_rows.sort(
        key=lambda row: (
            not row["passed"],
            row["selection_score_mean"],
            row["crash_rate_mean"],
            -row["minimum_altitude_mean_m"],
        )
    )
    summary_path = stage_dir / "easy_candidate_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    selected = candidate_rows[0]["candidate"]
    selected_runs = [
        row for row in successful if row["candidate"] == selected
    ]
    selected_run = min(
        selected_runs,
        key=lambda row: number(row.get("selection_score")) or 1e9,
    )
    selection = {
        "candidate": selected,
        "passed_gate": bool(candidate_rows[0]["passed"]),
        "source_tag": selected_run["tag"],
        "source_bundle": str(
            Path("artifacts") / "models" / "highdream" / selected_run["tag"]
        ),
    }
    (stage_dir / "selected_candidate.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[stage 2] selected {selected}; "
        f"gate={'PASS' if selection['passed_gate'] else 'FALLBACK (no candidate passed)'}"
    )
    return selected, selection


def curriculum_experiment(
    base: dict[str, Any],
    tag: str,
    lr: float,
    init_bundle: str,
    resume: bool,
) -> dict[str, Any]:
    experiment = copy.deepcopy(base)
    experiment["name"] = "altitude_safety_progressive_curriculum"
    experiment["script"] = "train_curriculum"
    experiment["output"] = {"name": "highdream", "tag": tag}
    experiment.setdefault("env", {})["target_mode"] = "loiter"
    experiment.setdefault("algo", {})["lr"] = lr
    experiment["runtime"] = {
        "num_env_runners": 1,
        "init_bundle": init_bundle,
        "resume": resume,
        "save_lightweight_bundle": True,
        "save_native_checkpoint": False,
    }
    experiment["curriculum"] = {
        "stages_module": "student.altitude_safety_pipeline_curriculum",
    }
    experiment["engagement_log"] = {
        "enabled": False,
    }
    experiment["notes"] = "Progressive altitude safety curriculum after easy-candidate selection."
    return experiment


def summarize_curriculum(work: Path, tag: str) -> Path:
    curriculum_dir = ROOT / "artifacts" / "curriculum" / "highdream" / tag
    rows = []
    for stage_index in range(5):
        episode_rows: list[dict[str, str]] = []
        for path in sorted(
            curriculum_dir.glob(
                f"episode_summary_stage_{stage_index:02d}_runner_*_env_*.csv"
            )
        ):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                episode_rows.extend(csv.DictReader(handle))
        altitudes = [
            float(row["minimum_altitude_m"])
            for row in episode_rows
            if number(row.get("minimum_altitude_m")) is not None
        ]
        crashes = sum(row.get("outcome") == "crash" for row in episode_rows)
        rows.append({
            "stage": stage_index,
            "episodes": len(episode_rows),
            "crash_rate": crashes / len(episode_rows) if episode_rows else "",
            "minimum_altitude_mean_m": (
                sum(altitudes) / len(altitudes) if altitudes else ""
            ),
            "minimum_altitude_worst_m": min(altitudes) if altitudes else "",
            "passed_episode_gate": (
                crashes / len(episode_rows) <= [0.20, 0.25, 0.30, 0.35, 0.40][stage_index]
                if episode_rows else False
            ),
        })
    path = work / "03_curriculum" / "curriculum_episode_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_progressive_curriculum(
    args: argparse.Namespace,
    base: dict[str, Any],
    work: Path,
    selected: str,
    selection: dict[str, Any] | None,
) -> None:
    stage_dir = work / "03_curriculum"
    tag = f"{args.campaign}_curriculum_{selected}"
    init_bundle = (
        selection["source_bundle"]
        if selection is not None
        else str(R08_BUNDLE.relative_to(ROOT))
    )
    experiment = curriculum_experiment(
        base,
        tag,
        args.lr,
        init_bundle,
        args.resume,
    )
    yaml_path = stage_dir / "progressive_curriculum.yaml"
    write_yaml(yaml_path, experiment)
    command = [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)]
    if args.dry_run:
        command.append("--dry-run")
    env = dict(os.environ)
    env["ALTITUDE_PIPELINE_CANDIDATE"] = selected
    print(f"[stage 3] candidate={selected}: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.dry_run:
        path = summarize_curriculum(work, tag)
        print(f"[done] curriculum episode summary: {path}")


def main() -> int:
    args = parse_args()
    if min(
        args.frozen_episodes,
        args.easy_iterations,
        args.easy_repeats,
    ) < 1:
        raise ValueError("episode, iteration, and repeat counts must be positive")
    if not R08_BUNDLE.exists():
        raise FileNotFoundError(f"R08 bundle not found: {R08_BUNDLE}")
    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign
    work.mkdir(parents=True, exist_ok=True)

    run_frozen_evaluation(args, work)
    selected, selection = run_easy_comparison(args, base, work)
    run_progressive_curriculum(
        args,
        base,
        work,
        selected,
        selection,
    )
    print(f"[done] altitude safety pipeline: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
