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
    parser.add_argument("--mode", choices=["default", "w53", "w56", "w63", "w64", "w65", "w66", "w67", "w68", "w69", "w70", "w71", "w72", "w73", "w74", "w75", "w76", "w77", "w78", "w79", "w80", "w81", "w82", "w83", "w84", "w85", "w86", "w87", "w88", "w89", "w90", "w91", "w92", "w93", "w95", "w96", "w97", "w100", "w101"], default="default")
    args = parser.parse_args()
    with args.log.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("frame_index")]
    if not rows:
        raise SystemExit("no telemetry rows")

    if args.mode in {"w53", "w56", "w63", "w64", "w65", "w66", "w67", "w68", "w69", "w70", "w71", "w72", "w73", "w74", "w75", "w76", "w77", "w78", "w79", "w80", "w81", "w82", "w83", "w84", "w85", "w86", "w87", "w88", "w89", "w90", "w91", "w92", "w93", "w95", "w96", "w97", "w100", "w101"}:
        from types import SimpleNamespace
        from run_unreal_inference import build_action_provider
        controller = build_action_provider(SimpleNamespace(mode=args.mode))
    else:
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
    rear_active = sum(bool(info.get("rear_rate_match_active", False)) for info in infos)
    print(f"rear_rate_match_active_rows={rear_active}")
    engagement_modes: dict[str, int] = {}
    for info in infos:
        mode = str(info.get("engagement_mode", ""))
        engagement_modes[mode] = engagement_modes.get(mode, 0) + 1
    print(f"engagement_modes={engagement_modes}")
    advantage_reasons: dict[str, int] = {}
    for info in infos:
        reason = str(info.get("advantage_manager_reason", ""))
        if reason:
            advantage_reasons[reason] = advantage_reasons.get(reason, 0) + 1
    if advantage_reasons:
        scores = [float(info.get("advantage_score", 0.0)) for info in infos]
        print(
            f"advantage_reasons={advantage_reasons} "
            f"score=[{min(scores):+.2f},{max(scores):+.2f}]"
        )
    lag_energy_rows = [
        i for i, info in enumerate(infos)
        if bool(info.get("lag_energy_preserve_active", False))
    ]
    if lag_energy_rows:
        lag_throttles = action_array[lag_energy_rows, 3]
        print(
            f"lag_energy_rows={len(lag_energy_rows)} "
            f"throttle=[{lag_throttles.min():.2f},"
            f"{np.median(lag_throttles):.2f},{lag_throttles.max():.2f}]"
        )
    print(
        "integrated_active_rows "
        f"overshoot={sum(bool(i.get('overshoot_control_active', False)) for i in infos)} "
        f"vertical={sum(bool(i.get('vertical_follow_active', False)) for i in infos)} "
        f"energy={sum(bool(i.get('energy_deficit', False)) for i in infos)} "
        f"mutual={sum(bool(i.get('mutual_commit_active', False)) for i in infos)} "
        f"terminal={sum(bool(i.get('terminal_track_active', False)) for i in infos)}"
    )
    rate_bank_rows = [
        i for i in infos if bool(i.get("rear_rate_match_active", False))
    ]
    if rate_bank_rows:
        mismatches = sum(
            float(i["target_bank"]) * float(i["target_yaw_rate"]) < 0.0
            for i in rate_bank_rows
            if abs(float(i["target_yaw_rate"])) >= 0.5
        )
        print(
            "rear_rate_bank "
            f"active={sum(bool(i.get('integrated_rate_bank_active', False)) for i in rate_bank_rows)} "
            f"sign_mismatch={mismatches} "
            f"mean_abs_bank={np.mean([abs(float(i['target_bank'])) for i in rate_bank_rows]):.1f}"
        )
    print(
        "predictive "
        f"active={sum(bool(i.get('predictive_guidance_active', False)) for i in infos)} "
        f"override={sum(bool(i.get('predictive_override_active', False)) for i in infos)} "
        f"spiral_recovery={sum(bool(i.get('spiral_recovery_active', False)) for i in infos)} "
        f"overbank={sum(bool(i.get('rate_deficit_overbank_active', False)) for i in infos)}"
    )
    planner_manoeuvres: dict[str, int] = {}
    for info in infos:
        manoeuvre = str(info.get("planner3d_manoeuvre", "inactive"))
        planner_manoeuvres[manoeuvre] = planner_manoeuvres.get(manoeuvre, 0) + 1
    print(
        "planner3d "
        f"active={sum(bool(i.get('planner3d_active', False)) for i in infos)} "
        f"replanned={sum(bool(i.get('planner3d_replanned', False)) for i in infos)} "
        f"manoeuvres={planner_manoeuvres}"
    )
    sequence_phases: dict[str, int] = {}
    for info in infos:
        phase = str(info.get("sequential_maneuver_phase", "none"))
        sequence_phases[phase] = sequence_phases.get(phase, 0) + 1
    print(f"sequential_phases={sequence_phases}")


if __name__ == "__main__":
    main()
