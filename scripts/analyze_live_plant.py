"""Identify achievable live-aircraft turn performance from telemetry CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError):
        return math.nan


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, float]] = []
    files_used = 0
    for path in args.paths:
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                file_rows = []
                for raw in csv.DictReader(handle):
                    item = {k: number(raw, k) for k in (
                        "sim_time_s", "own_speed_mps", "own_roll_deg",
                        "roll_cmd", "pitch_cmd", "roll_rate_degps",
                        "yaw_rate_degps", "own_vz_mps", "own_alt_m",
                    )}
                    if all(math.isfinite(item[k]) for k in (
                        "sim_time_s", "own_speed_mps", "own_roll_deg",
                        "roll_cmd", "pitch_cmd", "yaw_rate_degps",
                    )):
                        file_rows.append(item)
                if file_rows:
                    rows.extend(file_rows)
                    files_used += 1
        except (OSError, csv.Error):
            continue

    print(f"files={files_used} rows={len(rows)}")
    if not rows:
        return

    speed_bins = [(0, 180), (180, 210), (210, 240), (240, 999)]
    bank_bins = [(0, 45), (45, 65), (65, 75), (75, 91)]
    print("\nSUSTAINED TURN: signed yaw rate in commanded bank direction (deg/s)")
    print("speed bank n p50 p75 p90 opposite%")
    for slo, shi in speed_bins:
        for blo, bhi in bank_bins:
            vals = []
            opposite = 0
            for row in rows:
                speed = row["own_speed_mps"]
                bank = abs(row["own_roll_deg"])
                roll_rate = row["roll_rate_degps"]
                if not (slo <= speed < shi and blo <= bank < bhi):
                    continue
                if math.isfinite(roll_rate) and abs(roll_rate) > 20:
                    continue
                signed = row["yaw_rate_degps"] * (1.0 if row["own_roll_deg"] >= 0 else -1.0)
                vals.append(signed)
                opposite += signed < 0
            if vals:
                print(f"{slo:3}-{shi:<3} {blo:2}-{bhi:<2} {len(vals):6} "
                      f"{percentile(vals,50):5.1f} {percentile(vals,75):5.1f} "
                      f"{percentile(vals,90):5.1f} {100*opposite/len(vals):6.1f}")

    # Cross-correlation: command versus roll rate reveals actuator/control delay.
    print("\nROLL COMMAND RESPONSE: correlation roll_cmd(t) vs roll_rate(t+lag)")
    # Treat concatenated data conservatively: only adjacent samples with plausible dt.
    for lag_s in np.arange(0.0, 1.01, 0.1):
        lag_steps = int(round(lag_s * 10))
        x, y = [], []
        for idx in range(len(rows) - lag_steps):
            a, b = rows[idx], rows[idx + lag_steps]
            dt = b["sim_time_s"] - a["sim_time_s"]
            if abs(dt - lag_s) > 0.08 or not math.isfinite(b["roll_rate_degps"]):
                continue
            x.append(a["roll_cmd"])
            y.append(b["roll_rate_degps"])
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 20 else math.nan
        print(f"lag={lag_s:3.1f}s n={len(x):7} corr={corr:6.3f}")

    high = [r for r in rows if abs(r["own_roll_deg"]) >= 70]
    print("\nHIGH-BANK COST (|bank|>=70 deg)")
    for label, values in (
        ("speed", [r["own_speed_mps"] for r in high]),
        ("pitch_cmd", [r["pitch_cmd"] for r in high]),
        ("vertical_speed", [r["own_vz_mps"] for r in high if math.isfinite(r["own_vz_mps"])]),
    ):
        print(f"{label}: n={len(values)} p10={percentile(values,10):.2f} "
              f"p50={percentile(values,50):.2f} p90={percentile(values,90):.2f}")


if __name__ == "__main__":
    main()
