"""Summarize an altitude curriculum run using only the Python standard library."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    "reward_mean",
    "ep_len_mean",
    "crash_rate",
    "win_rate",
    "ep_wez_steps",
    "ep_mean_distance",
    "ep_min_distance",
    "ep_altitude_penalty_steps",
    "action_sat_rate",
)


def number(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "", "nan") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    state_path = args.run_dir / "curriculum_state.json"
    log_path = args.run_dir / "training_log.csv"
    if not state_path.exists() or not log_path.exists():
        print(f"[summary] Missing state or training log under {args.run_dir}")
        return 1

    state = json.loads(state_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with log_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["stage"]].append(row)

    summaries = []
    for stage, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        tail = rows[-max(1, args.window):]
        item: dict[str, object] = {
            "stage": int(stage),
            "iterations": len(rows),
            "window": len(tail),
            "status": state.get("stages", {}).get(stage, {}).get("status", "unknown"),
            "advance_reason": state.get("stages", {}).get(stage, {}).get("advance_reason"),
        }
        for metric in METRICS:
            values = [v for row in tail if (v := number(row.get(metric))) is not None]
            item[metric] = round(sum(values) / len(values), 6) if values else None
        summaries.append(item)

    payload = {
        "run_status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "total_iterations": state.get("total_iterations_elapsed"),
        "window": args.window,
        "stages": summaries,
    }
    json_path = args.run_dir / "curriculum_summary.json"
    csv_path = args.run_dir / "curriculum_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["stage", "iterations", "window", "status", "advance_reason", *METRICS]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"[summary] status={payload['run_status']} total_iterations={payload['total_iterations']}")
    for item in summaries:
        print(
            f"  stage={item['stage']} status={item['status']} "
            f"crash={item['crash_rate']} win={item['win_rate']} "
            f"wez={item['ep_wez_steps']} reward={item['reward_mean']}"
        )
    print(f"[summary] CSV:  {csv_path}")
    print(f"[summary] JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
