"""Compare live Unreal CSV candidates with one stable metric definition.

Uses sim_time_s when present and falls back to frame_index / 60 for legacy
logs.  Standard-library only so it works in the lightweight live environment.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else math.nan


def load(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("frame_index") and row.get("ata_deg")]


def summarize(path: Path) -> dict[str, float | str]:
    rows = load(path)
    if not rows:
        raise ValueError("no complete telemetry rows")

    if "sim_time_s" in rows[0] and rows[0].get("sim_time_s"):
        absolute_time = [number(row, "sim_time_s") for row in rows]
    else:
        absolute_time = [number(row, "frame_index") / 60.0 for row in rows]
    start_time = absolute_time[0]
    time_s = [value - start_time for value in absolute_time]

    ata = [number(row, "ata_deg") for row in rows]
    distance = [number(row, "distance_m") for row in rows]
    altitude = [number(row, "own_alt_m") for row in rows]
    own_speed = [number(row, "own_speed_mps") for row in rows]
    target_speed = [number(row, "enemy_speed_mps") for row in rows]

    post_indices = [i for i, value in enumerate(time_s) if value >= 5.0] or list(range(len(rows)))

    def time_below(limit: float) -> float:
        total = 0.0
        for i in range(1, len(rows)):
            if ata[i] <= limit:
                total += max(0.0, time_s[i] - time_s[i - 1])
        return total

    return {
        "file": path.name,
        "duration_s": time_s[-1],
        "post5_min_ata": min(ata[i] for i in post_indices),
        "time_ata90_s": time_below(90.0),
        "time_ata50_s": time_below(50.0),
        "time_ata20_s": time_below(20.0),
        "max_distance_m": max(distance),
        "end_distance_m": distance[-1],
        "altitude_delta_m": altitude[-1] - altitude[0],
        "max_abs_altitude_delta_m": max(abs(value - altitude[0]) for value in altitude),
        "max_own_speed_mps": max(own_speed),
        "end_speed_delta_mps": own_speed[-1] - target_speed[-1],
    }


def rank_key(item: dict[str, float | str]) -> tuple[float, ...]:
    # Engagement geometry first, then separation/safety.  Negate beneficial
    # accumulated times because sorted() is ascending.
    return (
        -float(item["time_ata20_s"]),
        -float(item["time_ata50_s"]),
        -float(item["time_ata90_s"]),
        float(item["post5_min_ata"]),
        float(item["max_distance_m"]),
        float(item["max_abs_altitude_delta_m"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    summaries = []
    for path in args.logs:
        try:
            summaries.append(summarize(path))
        except (OSError, ValueError, csv.Error) as exc:
            print(f"SKIP {path}: {exc}")

    summaries.sort(key=rank_key)
    header = (
        "rank file duration minATA>5s ATA<90s ATA<50s ATA<20s "
        "maxDist endDist altDelta maxSpeed endSpeedDiff"
    )
    print(header)
    for rank, item in enumerate(summaries, 1):
        print(
            f"{rank:>4} {item['file']} "
            f"{item['duration_s']:.1f} {item['post5_min_ata']:.1f} "
            f"{item['time_ata90_s']:.1f} {item['time_ata50_s']:.1f} {item['time_ata20_s']:.1f} "
            f"{item['max_distance_m']:.0f} {item['end_distance_m']:.0f} "
            f"{item['altitude_delta_m']:+.0f} {item['max_own_speed_mps']:.1f} "
            f"{item['end_speed_delta_mps']:+.1f}"
        )


if __name__ == "__main__":
    main()
