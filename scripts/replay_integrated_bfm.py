"""Replay a live CSV through IntegratedBFMController without sim dynamics.

This validates state transitions and command sanity, not counterfactual flight
performance. Legacy logs reconstruct target yaw from position differences.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogfight.ai.action_provider import ActionContext  # noqa: E402
from dogfight.ai.integrated_bfm_controller import (  # noqa: E402
    IntegratedBFMConfig,
    IntegratedBFMController,
)
from dogfight.sim.state_schema import StateIndex  # noqa: E402


def value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else default


def make_state(
    row: dict[str, str],
    own: bool,
    sim_time: float,
    fallback_target_yaw: float = 0.0,
) -> np.ndarray:
    state = np.zeros(46, dtype=np.float64)
    if own:
        state[StateIndex.N] = value(row, "own_n")
        state[StateIndex.E] = value(row, "own_e")
        state[StateIndex.D] = value(row, "own_alt_m")
        state[StateIndex.ROLL] = value(row, "own_roll_deg")
        state[StateIndex.PITCH] = value(row, "own_pitch_deg")
        state[StateIndex.YAW] = value(row, "own_yaw_deg") % 360.0
        state[6] = value(row, "own_vn_mps")
        state[7] = value(row, "own_ve_mps")
        state[8] = value(row, "own_vz_mps")
        state[StateIndex.KCAS] = value(row, "own_speed_mps")
        state[StateIndex.ALT] = value(row, "own_alt_m")
    else:
        state[StateIndex.N] = value(row, "enemy_n")
        state[StateIndex.E] = value(row, "enemy_e")
        state[StateIndex.D] = value(row, "enemy_alt_m")
        state[StateIndex.ROLL] = value(row, "enemy_roll_deg")
        state[StateIndex.PITCH] = value(row, "enemy_pitch_deg")
        state[StateIndex.YAW] = value(row, "enemy_yaw_deg", fallback_target_yaw) % 360.0
        state[6] = value(row, "enemy_vn_mps")
        state[7] = value(row, "enemy_ve_mps")
        state[8] = value(row, "enemy_vz_mps")
        state[StateIndex.KCAS] = value(row, "enemy_speed_mps")
        state[StateIndex.ALT] = value(row, "enemy_alt_m")
    state[StateIndex.SIM_TIME] = sim_time
    state[StateIndex.HEALTH] = 1.0
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--rear-commit-deg", type=float, default=180.0)
    args = parser.parse_args()
    with args.log.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("frame_index")]
    if not rows:
        raise SystemExit("no telemetry rows")

    controller = IntegratedBFMController(IntegratedBFMConfig(
        controller_name="replay",
        guidance_rear_commit_deg=args.rear_commit_deg,
    ))
    actions: list[np.ndarray] = []
    infos: list[dict] = []
    previous_enemy: tuple[float, float, float] | None = None
    fallback_yaw = 0.0
    start_time = None

    for row in rows:
        sim_time = value(row, "sim_time_s", value(row, "frame_index") / 60.0)
        if start_time is None:
            start_time = sim_time
        if previous_enemy is not None:
            prev_n, prev_e, prev_t = previous_enemy
            dt = sim_time - prev_t
            if dt > 1e-6:
                dn = value(row, "enemy_n") - prev_n
                de = value(row, "enemy_e") - prev_e
                if abs(dn) + abs(de) > 1e-6:
                    fallback_yaw = math.degrees(math.atan2(de, dn)) % 360.0
        own = make_state(row, True, sim_time)
        target = make_state(row, False, sim_time, fallback_yaw)
        result = controller.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=own,
            target_state=target,
        ))
        actions.append(np.asarray(result.action, dtype=np.float64))
        infos.append(result.info)
        previous_enemy = (value(row, "enemy_n"), value(row, "enemy_e"), sim_time)

    action_array = np.vstack(actions)
    if not np.isfinite(action_array).all():
        raise SystemExit("FAIL: non-finite action")
    states: dict[str, int] = {}
    for info in infos:
        states[info["state"]] = states.get(info["state"], 0) + 1
    target_banks = [float(info["target_bank"]) for info in infos]
    sign_flips = sum(a * b < 0.0 for a, b in zip(target_banks, target_banks[1:]))
    duration = value(rows[-1], "sim_time_s", value(rows[-1], "frame_index") / 60.0) - (
        start_time or 0.0
    )
    print(f"file={args.log.name} rows={len(rows)} duration={duration:.1f}s")
    print(f"states={states}")
    print(
        "action_range "
        f"roll=[{action_array[:,0].min():+.3f},{action_array[:,0].max():+.3f}] "
        f"pitch=[{action_array[:,1].min():+.3f},{action_array[:,1].max():+.3f}] "
        f"throttle=[{action_array[:,3].min():.3f},{action_array[:,3].max():.3f}]"
    )
    print(
        f"target_bank_sign_flips={sign_flips} "
        f"aim_az=[{min(float(i['aim_az']) for i in infos):+.1f},"
        f"{max(float(i['aim_az']) for i in infos):+.1f}] "
        f"threat_ata_min={min(float(i['threat_ata']) for i in infos):.1f}"
    )


if __name__ == "__main__":
    main()
