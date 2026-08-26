"""Convert a live --log-csv (run_unreal_inference.py / ProviderCommandPolicy)
into a Tacview-importable CSV pair, in the exact format
single_agent_env.py.make_tacviewLog() already writes for JSBSim replays
(Time,Longitude,Latitude,Altitude,Roll (deg),Pitch (deg),Yaw (deg),Health) --
so an actual LIVE session against the real competition opponent can be
watched frame-by-frame, not just read as CSV numbers.

The live log stores position as local N/E offsets in meters (own_n/own_e/
enemy_n/enemy_e), not lat/lon, so this converts them to lat/lon with a flat-
earth approximation around an arbitrary fixed reference point. The absolute
location is meaningless (never was one on record for the live match) --
only the RELATIVE geometry between the two aircraft matters for watching a
merge, and this preserves that exactly.

Usage:
  python scripts/adhoc_live_log_to_tacview.py --log-csv artifacts\\logs\\live_terminal_stabilizer_rollcommit_run1.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REF_LAT_DEG = 37.9235564
REF_LON_DEG = 128.1818813
EARTH_RADIUS_M = 6371000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-csv", required=True, help="Path to the live --log-csv file.")
    parser.add_argument("--out-dir", default=None, help="Default: same directory as --log-csv.")
    return parser.parse_args()


def ned_to_latlon(n_m: float, e_m: float) -> tuple[float, float]:
    lat = REF_LAT_DEG + math.degrees(n_m / EARTH_RADIUS_M)
    lon = REF_LON_DEG + math.degrees(
        e_m / (EARTH_RADIUS_M * math.cos(math.radians(REF_LAT_DEG)))
    )
    return lat, lon


def gf(row: dict[str, str], key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_csv)
    out_dir = Path(args.out_dir) if args.out_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = log_path.stem

    with log_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[error] log file has no rows")
        return 1

    ownship_path = out_dir / f"{stem}_tacview_ownship_(F-16)[Blue].csv"
    target_path = out_dir / f"{stem}_tacview_target_(F-16)[Red].csv"

    with ownship_path.open("w", encoding="utf-8") as fo, target_path.open("w", encoding="utf-8") as ft:
        header = "Time,Longitude,Latitude,Altitude,Roll (deg),Pitch (deg),Yaw (deg),Health\n"
        fo.write(header)
        ft.write(header)
        n_written = 0
        for row in rows:
            t = gf(row, "sim_time_s")
            own_n, own_e, own_alt = gf(row, "own_n"), gf(row, "own_e"), gf(row, "own_alt_m")
            own_roll, own_pitch, own_yaw = gf(row, "own_roll_deg"), gf(row, "own_pitch_deg"), gf(row, "own_yaw_deg")
            enemy_n, enemy_e, enemy_alt = gf(row, "enemy_n"), gf(row, "enemy_e"), gf(row, "enemy_alt_m")
            enemy_roll, enemy_pitch, enemy_yaw = gf(row, "enemy_roll_deg"), gf(row, "enemy_pitch_deg"), gf(row, "enemy_yaw_deg")
            if None in (t, own_n, own_e, own_alt, own_roll, own_pitch, own_yaw):
                continue
            own_lat, own_lon = ned_to_latlon(own_n, own_e)
            fo.write(f"{t},{own_lon},{own_lat},{own_alt},{own_roll},{own_pitch},{own_yaw},\n")
            if None not in (enemy_n, enemy_e, enemy_alt, enemy_roll, enemy_pitch, enemy_yaw):
                enemy_lat, enemy_lon = ned_to_latlon(enemy_n, enemy_e)
                ft.write(f"{t},{enemy_lon},{enemy_lat},{enemy_alt},{enemy_roll},{enemy_pitch},{enemy_yaw},\n")
            n_written += 1

    print(f"[done] {n_written} frames written")
    print(f"  ownship: {ownship_path}")
    print(f"  target:  {target_path}")
    print("  Tacview: File > Import > CSV, map Longitude/Latitude/Altitude/Roll/Pitch/Yaw, import both files into the same timeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
