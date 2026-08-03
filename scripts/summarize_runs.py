from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METRICS = [
    "reward_mean",
    "ep_len_mean",
    "crash_rate",
    "timeout_rate",
    "ep_wez_steps",
    "ep_mean_distance",
    "ep_min_distance",
    "ep_altitude_penalty_steps",
    "final_ata_deg",
    "final_aa_deg",
    "action_sat_rate",
]


def to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        return None
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8-sig", newline="")))
    valid = [row for row in rows if to_float(row.get("reward_mean")) is not None]
    result: dict[str, Any] = {
        "tag": run_dir.name,
        "valid_rows": len(valid),
        "log": str(log_path),
    }
    if not valid:
        return result

    last = valid[-1]
    result["episodes"] = last.get("episodes", "")
    for metric in METRICS:
        values = [to_float(row.get(metric)) for row in valid]
        values = [value for value in values if value is not None]
        result[f"last_{metric}"] = to_float(last.get(metric))
        result[f"avg_{metric}"] = sum(values) / len(values) if values else None
    return result


def score(row: dict[str, Any]) -> tuple:
    crash = row.get("last_crash_rate")
    avg_crash = row.get("avg_crash_rate")
    length = row.get("last_ep_len_mean") or 0.0
    min_range = row.get("last_ep_min_distance") or 0.0
    mean_range = row.get("last_ep_mean_distance") or 999999.0
    far_over = max(0.0, mean_range - 6000.0)
    return (
        999.0 if crash is None else crash,
        999.0 if avg_crash is None else avg_crash,
        -length,
        far_over,
        -min(min_range, 500.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DogFight training runs.")
    parser.add_argument("--pattern", default="raf_*", help="Run directory glob under artifacts/logs/<name>.")
    parser.add_argument("--output-name", default="lt")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_root = ROOT / "artifacts" / "logs" / args.output_name
    rows = []
    for run_dir in sorted(logs_root.glob(args.pattern)):
        if run_dir.is_dir():
            row = summarize_run(run_dir)
            if row:
                rows.append(row)
    rows.sort(key=score)
    if args.top:
        rows = rows[: args.top]

    columns = [
        "tag",
        "valid_rows",
        "episodes",
        "last_crash_rate",
        "avg_crash_rate",
        "last_timeout_rate",
        "last_ep_len_mean",
        "last_reward_mean",
        "last_ep_min_distance",
        "last_ep_mean_distance",
        "last_ep_altitude_penalty_steps",
        "last_final_ata_deg",
        "last_final_aa_deg",
        "last_action_sat_rate",
        "log",
    ]

    if args.csv:
        out = Path(args.csv)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {out}")

    print(",".join(columns[:-1]))
    for row in rows:
        values = []
        for column in columns[:-1]:
            value = row.get(column, "")
            if isinstance(value, float):
                value = round(value, 4)
            values.append(str(value))
        print(",".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
