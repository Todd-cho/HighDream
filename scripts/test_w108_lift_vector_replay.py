"""Counterfactual one-step replay smoke test for the W108 lift-vector overlay.

This does not claim to reproduce a new closed-loop trajectory.  It feeds the
same recorded W100 telemetry to W100 and W108 and verifies isolation, numeric
stability, activation, and bounded command generation before a live run.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_unreal_inference as inference
from dogfight.ai.action_provider import ActionContext
from dogfight.sim.state_schema import StateIndex


def _provider(mode: str):
    old_argv = sys.argv
    try:
        sys.argv = ["replay", "--mode", mode, "--team-name", "blue"]
        return inference.build_action_provider(inference.parse_args())
    finally:
        sys.argv = old_argv


def _state(row: dict[str, str], prefix: str) -> np.ndarray:
    state = np.zeros(51, dtype=np.float32)
    state[StateIndex.N] = float(row[f"{prefix}_n"])
    state[StateIndex.E] = float(row[f"{prefix}_e"])
    # CSV altitude is up-positive; controller state position uses NED.
    state[StateIndex.D] = -float(row[f"{prefix}_alt_m"])
    state[StateIndex.ROLL] = float(row[f"{prefix}_roll_deg"])
    state[StateIndex.PITCH] = float(row[f"{prefix}_pitch_deg"])
    state[StateIndex.YAW] = float(row[f"{prefix}_yaw_deg"])
    state[StateIndex.KCAS] = float(row[f"{prefix}_speed_mps"])
    state[StateIndex.SIM_TIME] = float(row["sim_time_s"])
    state[StateIndex.ALT] = float(row[f"{prefix}_alt_m"])
    return state


def replay(path: Path, mode: str) -> tuple[int, float, float, float]:
    provider = _provider(mode)
    active = 0
    max_abs_action = 0.0
    max_accel = 0.0
    max_logged_action_error = 0.0
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            own = _state(row, "own")
            target = _state(row, "enemy")
            result = provider.compute_action(ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=own,
                target_state=target,
            ))
            if not np.all(np.isfinite(result.action)):
                raise AssertionError(f"{mode}: non-finite action")
            if mode == "w100" and row.get("provider_roll_cmd", "") != "":
                logged_action = np.array([
                    float(row["provider_roll_cmd"]),
                    float(row["provider_pitch_cmd"]),
                    float(row["provider_yaw_cmd"]),
                    float(row["provider_throttle_cmd"]),
                ])
                max_logged_action_error = max(
                    max_logged_action_error,
                    float(np.max(np.abs(result.action - logged_action))),
                )
            max_abs_action = max(
                max_abs_action, float(np.max(np.abs(result.action)))
            )
            if result.info.get("lift_vector_active", False):
                active += 1
                max_accel = max(
                    max_accel,
                    float(result.info.get("lift_vector_accel", 0.0)),
                )
    return active, max_abs_action, max_accel, max_logged_action_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    w100 = replay(args.csv, "w100")
    w108 = replay(args.csv, "w108")
    assert w100[0] == 0, "W100 isolation failed: overlay unexpectedly active"
    assert w108[0] > 0, "W108 overlay never activated on the reference log"
    assert w108[1] <= 1.00001, "W108 emitted an out-of-bounds command"
    assert w108[2] <= 35.00001, "W108 exceeded configured acceleration limit"
    assert w100[3] <= 1e-5, (
        f"W100 historical behaviour changed: max error={w100[3]:.8f}"
    )
    print(
        f"W100 active={w100[0]} max_action={w100[1]:.3f} "
        f"max_logged_error={w100[3]:.8f}"
    )
    print(
        f"W108 active={w108[0]} max_action={w108[1]:.3f} "
        f"max_accel={w108[2]:.3f}"
    )


if __name__ == "__main__":
    main()
