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


def replay(path: Path, mode: str, *, capture_actions: bool = False):
    provider = _provider(mode)
    active = 0
    max_abs_action = 0.0
    max_accel = 0.0
    max_logged_action_error = 0.0
    previous_bank_sign = 0
    bank_sign_flips = 0
    max_abs_pitch = 0.0
    max_authority = 0.0
    first_active_time = -1.0
    first_merge_pass_time = -1.0
    opening_boost_frames = 0
    terminal_track_frames = 0
    actions: list[np.ndarray] = []
    explicit_phase_frames = 0
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
            if capture_actions:
                actions.append(np.asarray(result.action, dtype=np.float64).copy())
            if result.info.get("engagement_phase", ""):
                explicit_phase_frames += 1
            if (
                mode == "w100"
                and "w100_raw" in path.name
                and row.get("provider_roll_cmd", "") != ""
            ):
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
            max_abs_pitch = max(max_abs_pitch, abs(float(result.action[1])))
            max_authority = max(
                max_authority,
                float(result.info.get("lift_vector_authority", 0.0)),
            )
            if result.info.get("lift_vector_active", False):
                active += 1
                if first_active_time < 0.0:
                    first_active_time = float(row["sim_time_s"])
                max_accel = max(
                    max_accel,
                    float(result.info.get("lift_vector_accel", 0.0)),
                )
                target_bank = float(result.info.get("target_bank", 0.0))
                bank_sign = 1 if target_bank > 10.0 else -1 if target_bank < -10.0 else 0
                if (
                    bank_sign != 0
                    and previous_bank_sign != 0
                    and bank_sign != previous_bank_sign
                ):
                    bank_sign_flips += 1
                if bank_sign != 0:
                    previous_bank_sign = bank_sign
            if (
                first_merge_pass_time < 0.0
                and result.info.get("first_merge_passed", False)
            ):
                first_merge_pass_time = float(row["sim_time_s"])
            if result.info.get("opening_pull_boost_active", False):
                opening_boost_frames += 1
            if result.info.get("terminal_track_active", False):
                terminal_track_frames += 1
    summary = (
        active, max_abs_action, max_accel, max_logged_action_error,
        bank_sign_flips, max_abs_pitch, max_authority,
        first_active_time, first_merge_pass_time,
        opening_boost_frames,
        terminal_track_frames,
    )
    return (summary, np.vstack(actions), explicit_phase_frames) if capture_actions else summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    w100 = replay(args.csv, "w100")
    w108 = replay(args.csv, "w108")
    w109 = replay(args.csv, "w109")
    w110 = replay(args.csv, "w110")
    w111 = replay(args.csv, "w111")
    w112 = replay(args.csv, "w112")
    w113 = replay(args.csv, "w113")
    w111_capture, w111_actions, _ = replay(args.csv, "w111", capture_actions=True)
    ep1_capture, ep1_actions, ep1_phase_frames = replay(args.csv, "ep1", capture_actions=True)
    assert w100[0] == 0, "W100 isolation failed: overlay unexpectedly active"
    assert w108[0] > 0, "W108 overlay never activated on the reference log"
    assert w108[1] <= 1.00001, "W108 emitted an out-of-bounds command"
    assert w108[2] <= 35.00001, "W108 exceeded configured acceleration limit"
    assert w109[0] > 0, "W109 overlay never activated on the reference log"
    assert w109[1] <= 1.00001, "W109 emitted an out-of-bounds command"
    assert w109[2] <= 25.00001, "W109 exceeded configured acceleration limit"
    assert w109[4] < w108[4], "W109 did not reduce lift-vector bank chatter"
    assert w110[0] > 0, "W110 overlay never activated on the reference log"
    assert w110[5] <= 0.92001, "W110 exceeded its pitch authority budget"
    assert w110[6] <= 1.00001, "W110 authority weights exceed unity"
    assert w111[0] > 0, "W111 never enabled post-merge lift guidance"
    assert w111[5] <= 0.95001, "W111 exceeded its pitch authority budget"
    assert w111[2] <= 25.00001, "W111 exceeded acceleration authority"
    assert w111[8] >= 0.0, "W111 failed to detect the first merge pass"
    assert w111[7] >= w111[8], "W111 lift guidance activated before merge"
    assert w112[9] > 0, "W112 opening pull boost never activated"
    assert w112[5] <= 0.95001, "W112 exceeded pitch authority budget"
    assert w113[9] == 0, "W113 unexpectedly inherited opening boost"
    assert w113[10] > 0, "W113 side-shot terminal gate never activated"
    assert np.array_equal(w111_actions, ep1_actions), "EP1 changed W111 commands"
    assert ep1_phase_frames == len(ep1_actions), "EP1 phase telemetry is incomplete"
    assert ep1_capture == w111_capture, "EP1 changed W111 replay summary"
    assert w100[3] <= 1e-5, (
        f"W100 historical behaviour changed: max error={w100[3]:.8f}"
    )
    print(
        f"W100 active={w100[0]} max_action={w100[1]:.3f} "
        f"max_logged_error={w100[3]:.8f}"
    )
    print(
        f"W108 active={w108[0]} max_action={w108[1]:.3f} "
        f"max_accel={w108[2]:.3f} bank_flips={w108[4]}"
    )
    print(
        f"W109 active={w109[0]} max_action={w109[1]:.3f} "
        f"max_accel={w109[2]:.3f} bank_flips={w109[4]}"
    )
    print(
        f"W110 active={w110[0]} max_action={w110[1]:.3f} "
        f"max_accel={w110[2]:.3f} bank_flips={w110[4]} "
        f"max_pitch={w110[5]:.3f} max_authority={w110[6]:.3f}"
    )
    print(
        f"W111 active={w111[0]} max_action={w111[1]:.3f} "
        f"max_accel={w111[2]:.3f} bank_flips={w111[4]} "
        f"max_pitch={w111[5]:.3f} max_authority={w111[6]:.3f} "
        f"merge={w111[8]:.2f}s first_active={w111[7]:.2f}s"
    )
    print(
        f"W112 active={w112[0]} max_pitch={w112[5]:.3f} "
        f"bank_flips={w112[4]} opening_boost={w112[9]}frames "
        f"merge={w112[8]:.2f}s first_active={w112[7]:.2f}s"
    )
    print(
        f"W113 active={w113[0]} max_pitch={w113[5]:.3f} "
        f"bank_flips={w113[4]} terminal={w113[10]}frames "
        f"opening_boost={w113[9]}frames"
    )
    print(f"EP1 parity=exact frames={len(ep1_actions)} phase_rows={ep1_phase_frames}")


if __name__ == "__main__":
    main()
