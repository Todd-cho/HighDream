from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from dogfight.envs.initial_scenario import validate_scenario_pool
from student.my_reward import MY_REWARD_CONFIG
from student.reward_search_curriculum import (
    SCREEN_SCENARIO_POOL,
    VALIDATION_SCENARIO_POOL,
)


PARAMETER_KEYS = [
    "altitude_soft_floor_m",
    "altitude_hard_floor_m",
    "low_altitude_penalty",
    "very_low_altitude_penalty",
    "altitude_bonus_high_min_m",
    "altitude_bonus_high_max_m",
    "altitude_bonus_mid_min_m",
    "altitude_bonus_high",
    "altitude_bonus_mid",
    "nose_down_altitude_m",
    "nose_down_pitch_deg",
    "nose_down_penalty",
    "crash_penalty",
]

METRIC_KEYS = [
    "crash_rate",
    "episode_length",
    "altitude_penalty_steps",
    "altitude_penalty_rate",
    "safety_reward",
    "mean_range_m",
    "min_range_m",
    "wez_steps",
    "saturation_rate",
    "episode_summary_rows",
    "minimum_altitude_mean_m",
    "minimum_altitude_worst_m",
    "low_altitude_episode_rate",
    "nose_down_crash_rate",
    "roll_instability_crash_rate",
    "valid_rows",
    "selection_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled altitude reward search using the Release-260721 scenario pool."
    )
    parser.add_argument("--phase", choices=("screen", "validate"), required=True)
    parser.add_argument("--base", default="experiments/altitude_reward_search_base.yaml")
    parser.add_argument("--campaign", default="altitude_reward_search_v2")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260721)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def check_environment() -> None:
    missing = [name for name in ("yaml", "ray", "torch") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Training environment is missing: {', '.join(missing)}")
    screen = validate_scenario_pool(SCREEN_SCENARIO_POOL)
    validation = validate_scenario_pool(VALIDATION_SCENARIO_POOL)
    if len(screen) != 3 or len(validation) != 4:
        raise RuntimeError("Expected three screen and four validation scenarios")


def current_baseline() -> dict[str, float]:
    return {key: float(MY_REWARD_CONFIG[key]) for key in PARAMETER_KEYS}


def candidate_parameters(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    baseline = current_baseline()
    baseline["_label"] = "old_final04_reference"
    candidates.append(baseline)
    seen = {tuple(baseline[key] for key in PARAMETER_KEYS)}

    while len(candidates) < count:
        soft = float(rng.choice((2600, 3000, 3400, 3800)))
        hard = float(rng.choice(tuple(value for value in (1000, 1400, 1800, 2200) if value < soft)))
        high_bonus = round(rng.uniform(0.40, 0.80), 2)
        mid_bonus = round(rng.uniform(0.12, min(0.38, high_bonus - 0.08)), 2)
        low_penalty = round(rng.uniform(0.9, 2.5), 2)
        params: dict[str, Any] = {
            "altitude_soft_floor_m": soft,
            "altitude_hard_floor_m": hard,
            "low_altitude_penalty": low_penalty,
            "very_low_altitude_penalty": round(rng.uniform(max(2.5, low_penalty + 0.8), 6.0), 2),
            # Avoid rewarding the same altitude band that the soft-floor term penalizes.
            "altitude_bonus_high_min_m": soft,
            "altitude_bonus_high_max_m": 9000.0,
            "altitude_bonus_mid_min_m": hard,
            "altitude_bonus_high": high_bonus,
            "altitude_bonus_mid": mid_bonus,
            "nose_down_altitude_m": soft + float(rng.choice((300, 600, 900))),
            "nose_down_pitch_deg": float(rng.choice((-10, -8, -6))),
            "nose_down_penalty": round(rng.uniform(-1.8, -0.7), 2),
            "crash_penalty": float(rng.choice((-200, -240, -280))),
            "_label": "structured_random",
        }
        signature = tuple(params[key] for key in PARAMETER_KEYS)
        if signature not in seen:
            seen.add(signature)
            candidates.append(params)
    return candidates


def build_experiment(
    base: dict[str, Any], tag: str, iterations: int, params: dict[str, Any], validation: bool
) -> dict[str, Any]:
    experiment = copy.deepcopy(base)
    experiment.setdefault("output", {})["tag"] = tag
    experiment.setdefault("runtime", {})["iterations"] = iterations
    env_config = experiment.setdefault("env_config", {})
    pool = VALIDATION_SCENARIO_POOL if validation else SCREEN_SCENARIO_POOL
    env_config["initial_scenario"] = copy.deepcopy(pool)
    env_config["randomization"] = {"enabled": False}
    reward = dict(MY_REWARD_CONFIG)
    reward.update({key: params[key] for key in PARAMETER_KEYS})
    env_config["reward"] = reward
    if validation:
        engagement = experiment.setdefault("engagement_log", {})
        engagement.update({"enabled": True, "interval": iterations, "steps": 1800, "episodes": 2})
    return experiment


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_experiment(yaml_path: Path, log_path: Path, dry_run: bool) -> int:
    command = [sys.executable, str(RUN_EXPERIMENT), str(yaml_path)]
    if dry_run:
        command.append("--dry-run")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[reward-search] {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := number(row.get(key))) is not None]
    return mean(values) if values else None


def dashboard_path(experiment: dict[str, Any]) -> Path:
    root = resolve(experiment.get("dashboard", {}).get("logdir", "artifacts/dashboard"))
    run_name = f"{experiment['output']['name']}_{experiment['output']['tag']}"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_name)
    return root / safe / "metrics.jsonl"


def summarize(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if number(row.get("episode/crash_rate")) is not None:
                rows.append(row)
    evaluation = rows[len(rows) // 2 :] if rows else []
    length = average(evaluation, "episode/length")
    altitude_steps = average(evaluation, "dogfight/altitude_penalty_steps")
    result = {
        "crash_rate": average(evaluation, "episode/crash_rate"),
        "episode_length": length,
        "altitude_penalty_steps": altitude_steps,
        "altitude_penalty_rate": (
            altitude_steps / length if altitude_steps is not None and length and length > 0 else None
        ),
        "safety_reward": average(evaluation, "reward/safety"),
        "mean_range_m": average(evaluation, "dogfight/distance_mean"),
        "min_range_m": average(evaluation, "dogfight/distance_min"),
        "wez_steps": average(evaluation, "dogfight/wez_steps"),
        "saturation_rate": average(evaluation, "action/saturation_rate"),
        "valid_rows": len(evaluation),
    }
    result["selection_score"] = selection_score(result)
    return result


def add_episode_summary_metrics(
    result: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    run_dir = (
        ROOT
        / "artifacts"
        / "logs"
        / str(experiment["output"]["name"])
        / str(experiment["output"]["tag"])
    )
    rows: list[dict[str, str]] = []
    for path in sorted(run_dir.glob("episode_summary*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    altitudes = [
        value
        for row in rows
        if (value := number(row.get("minimum_altitude_m"))) is not None
    ]
    causes = [str(row.get("primary_cause", "")) for row in rows]
    count = len(rows)
    result.update({
        "episode_summary_rows": count,
        "minimum_altitude_mean_m": mean(altitudes) if altitudes else None,
        "minimum_altitude_worst_m": min(altitudes) if altitudes else None,
        "low_altitude_episode_rate": (
            sum(value < 1000.0 for value in altitudes) / len(altitudes)
            if altitudes else None
        ),
        "nose_down_crash_rate": (
            causes.count("altitude_nose_down_crash") / count if count else None
        ),
        "roll_instability_crash_rate": (
            causes.count("altitude_roll_instability_crash") / count
            if count else None
        ),
    })
    result["selection_score"] = selection_score(result)
    return result


def selection_score(row: dict[str, Any]) -> float:
    crash = number(row.get("crash_rate"))
    valid = number(row.get("valid_rows")) or 0
    if crash is None or valid < 1:
        return 1_000_000.0
    altitude_rate = number(row.get("altitude_penalty_rate")) or 0.0
    mean_range = number(row.get("mean_range_m")) or 10_000.0
    min_range = number(row.get("min_range_m")) or 10_000.0
    saturation = number(row.get("saturation_rate")) or 0.0
    wez = number(row.get("wez_steps")) or 0.0
    low_altitude_rate = number(row.get("low_altitude_episode_rate")) or 0.0
    nose_crash_rate = number(row.get("nose_down_crash_rate")) or 0.0
    roll_crash_rate = number(row.get("roll_instability_crash_rate")) or 0.0
    worst_altitude = number(row.get("minimum_altitude_worst_m"))
    score = crash * 1000.0 + altitude_rate * 200.0
    score += low_altitude_rate * 400.0
    score += nose_crash_rate * 300.0 + roll_crash_rate * 300.0
    if worst_altitude is not None:
        score += max(0.0, 1000.0 - worst_altitude) / 2.0
    score += max(0.0, mean_range - 5000.0) / 20.0
    score += max(0.0, min_range - 2500.0) / 10.0
    score += max(0.0, saturation - 0.05) * 2000.0
    score -= min(10.0, wez) * 2.0
    return round(score, 6)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["trial", "candidate", "repeat", "label", "tag", "status", "return_code", *PARAMETER_KEYS, *METRIC_KEYS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def eligible_candidates(screen_path: Path, top_k: int) -> list[dict[str, str]]:
    rows = [
        row for row in read_csv(screen_path)
        if row.get("status") == "success" and (number(row.get("valid_rows")) or 0) >= 1
    ]
    rows.sort(
        key=lambda row: (
            score if (score := number(row.get("selection_score"))) is not None
            else 1_000_000.0
        )
    )
    return rows[:top_k]


def write_validation_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    candidates = sorted({str(row["candidate"]) for row in successful})
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        group = [row for row in successful if str(row["candidate"]) == candidate]
        if not group:
            continue
        summary: dict[str, Any] = {
            "candidate": candidate,
            "successful_repeats": len(group),
            "label": group[0].get("label", ""),
            **{key: group[0].get(key) for key in PARAMETER_KEYS},
        }
        for key in METRIC_KEYS:
            values = [value for row in group if (value := number(row.get(key))) is not None]
            summary[f"{key}_mean"] = mean(values) if values else None
        crashes = [value for row in group if (value := number(row.get("crash_rate"))) is not None]
        summary["crash_rate_std"] = pstdev(crashes) if len(crashes) > 1 else 0.0 if crashes else None
        summary["crash_rate_worst"] = max(crashes) if crashes else None
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            score if (score := number(row.get("selection_score_mean"))) is not None
            else 1_000_000.0
        )
    )
    fields = [
        "rank", "candidate", "successful_repeats", "label", *PARAMETER_KEYS,
        *[f"{key}_mean" for key in METRIC_KEYS], "crash_rate_std", "crash_rate_worst",
    ]
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def execute(args: argparse.Namespace, base: dict[str, Any], work: Path) -> int:
    validation = args.phase == "validate"
    iterations = args.iterations or (50 if validation else 30)
    if validation:
        selected = eligible_candidates(work / "screen_results.csv", args.top_k)
        if not selected:
            raise RuntimeError("No valid screen candidates. Run --phase screen first.")
        jobs: list[tuple[str, int, dict[str, Any]]] = []
        for candidate in selected:
            params = {key: float(candidate[key]) for key in PARAMETER_KEYS}
            params["_label"] = candidate.get("label", "selected")
            for repeat in range(1, args.repeats + 1):
                jobs.append((candidate["candidate"], repeat, params))
        results_path = work / "validation_results.csv"
    else:
        jobs = [(f"C{index:02d}", 1, params) for index, params in enumerate(candidate_parameters(args.trials, args.seed), 1)]
        results_path = work / "screen_results.csv"

    existing = read_csv(results_path) if args.resume else []
    completed = {row["trial"] for row in existing if row.get("status") == "success"}
    results: list[dict[str, Any]] = list(existing)

    for index, (candidate, repeat, original) in enumerate(jobs, 1):
        trial = f"{index:03d}"
        params = dict(original)
        label = str(params.pop("_label", "candidate"))
        tag = f"{args.campaign}_{args.phase}_{candidate}_r{repeat}_{iterations}iter"
        yaml_path = work / "generated" / args.phase / f"{trial}_{candidate}_r{repeat}.yaml"
        experiment = build_experiment(base, tag, iterations, params, validation)
        write_yaml(yaml_path, experiment)
        if trial in completed:
            print(f"[resume] skip completed {trial}")
            continue
        code = run_experiment(yaml_path, work / "logs" / args.phase / f"{trial}.log", args.dry_run)
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
            "label": label,
            "tag": tag,
            "status": "success" if code == 0 else "failed",
            "return_code": code,
            **params,
            **metrics,
        }
        results = [old for old in results if old.get("trial") != trial]
        results.append(row)
        results.sort(key=lambda item: int(item["trial"]))
        write_csv(results_path, results)

    print(f"[done] results: {results_path}")
    if validation and not args.dry_run:
        write_validation_summary(work / "validation_summary.csv", results)
        print(f"[done] candidate summary: {work / 'validation_summary.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.trials < 1 or args.top_k < 1 or args.repeats < 1:
        raise ValueError("trials, top-k, and repeats must be positive")
    check_environment()
    base_path = resolve(args.base)
    base = load_yaml(base_path)
    work = ROOT / "artifacts" / args.campaign
    return execute(args, base, work)


if __name__ == "__main__":
    raise SystemExit(main())
