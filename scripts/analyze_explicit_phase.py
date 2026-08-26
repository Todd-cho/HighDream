"""Summarize EP1+ phase ownership, conflicts, and saturation from a live CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    with args.log.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("frame_index")]
    if not rows:
        raise SystemExit("no complete telemetry rows")

    def counts(field: str) -> Counter[str]:
        return Counter(row.get(field, "") or "blank" for row in rows)

    print(f"file={args.log.name} frames={len(rows)} spawn={rows[0].get('spawn_yaw_class', '')}")
    for field in ("engagement_phase", "bank_owner", "pitch_owner", "throttle_owner"):
        rendered = ", ".join(
            f"{name}:{count}({100.0 * count / len(rows):.1f}%)"
            for name, count in counts(field).most_common()
        )
        print(f"{field}: {rendered}")
    conflicts = sum(_truthy(row.get("proposal_conflict", "")) for row in rows)
    saturated = sum(_truthy(row.get("proposal_saturated", "")) for row in rows)
    print(f"conflict={conflicts}({100.0 * conflicts / len(rows):.1f}%)")
    print(f"saturated={saturated}({100.0 * saturated / len(rows):.1f}%)")
    print("top_bank_combinations:")
    for name, count in counts("active_bank_proposals").most_common(10):
        print(f"  {name}: {count} ({100.0 * count / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
