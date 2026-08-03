from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize altitude-safety episode CSV files."
    )
    parser.add_argument(
        "run_dir",
        help="Run log directory or path relative to the release root.",
    )
    parser.add_argument("--csv", default="", help="Optional output CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    paths = sorted(run_dir.glob("episode_summary*.csv"))
    if not paths:
        raise FileNotFoundError(f"No episode summary CSV found under {run_dir}")

    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    altitudes = [
        value
        for row in rows
        if (value := number(row.get("minimum_altitude_m"))) is not None
    ]
    causes = Counter(str(row.get("primary_cause", "")) for row in rows)
    outcomes = Counter(str(row.get("outcome", "")) for row in rows)
    scenarios = Counter(str(row.get("scenario_name", "")) for row in rows)

    summary = {
        "episodes": len(rows),
        "minimum_altitude_mean_m": round(mean(altitudes), 3) if altitudes else "",
        "minimum_altitude_worst_m": round(min(altitudes), 3) if altitudes else "",
        "low_altitude_below_1000_rate": (
            round(sum(value < 1000.0 for value in altitudes) / len(altitudes), 6)
            if altitudes else ""
        ),
        "crash_rate": (
            round(outcomes.get("crash", 0) / len(rows), 6) if rows else ""
        ),
        "nose_down_crashes": causes.get("altitude_nose_down_crash", 0),
        "roll_instability_crashes": causes.get(
            "altitude_roll_instability_crash", 0
        ),
        "low_altitude_crashes": causes.get("low_altitude_crash", 0),
        "scenarios": "; ".join(
            f"{name or '(unknown)'}={count}"
            for name, count in sorted(scenarios.items())
        ),
        "causes": "; ".join(
            f"{name or '(unknown)'}={count}"
            for name, count in causes.most_common()
        ),
    }
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.csv:
        output = Path(args.csv)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
