# -*- coding: utf-8 -*-
"""Copy of student/my_reward.py (do not edit the shared file) plus delta-based
range/ata shaping: 이전 스텝 대비 거리·ata가 줄어들면 보상, 늘어나면 페널티.

Motivation (2026-08-06, stage6/6a/6b/6c 4연속 실패 분석): 기존 range/ata reward는
전부 "지금 특정 밴드 안에 있는가"만 보는 정적(static) 보상이라, target이 직선비행
일 때만 우연히 통했고(stage6b) target이 선회하자(stage6c) 조준 능력이 바로 퇴행함.
"지금 좁혀지고 있는가"를 매 스텝 직접 평가하는 delta 항을 추가해서 이 문제를 분리
검증한다. 나머지 로직/파라미터는 my_reward.py와 완전히 동일함.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex, position_ned


MY_REWARD_CONFIG = {
    "step_penalty": -0.003,
    "survival_bonus": 0.0,
    "too_close_m": 460.0,
    "ideal_range_min_m": 450.0,
    "ideal_range_max_m": 1100.0,
    "range_scale": 0.8,
    "overshoot_penalty": 4.0,
    "overshoot_quadratic_scale": 1.5,
    "inside_min_range_penalty": -3.0,
    "ata_scale": 0.12,
    "aa_scale": 0.03,
    "wez_bonus": 0.5,
    "damage_scale": 20.0,
    "altitude_soft_floor_m": 3800.0,
    "altitude_hard_floor_m": 1800.0,
    "low_altitude_penalty": 1.76,
    "very_low_altitude_penalty": 5.49,
    "altitude_bonus_high_min_m": 2100.0,
    "altitude_bonus_high_max_m": 9000.0,
    "altitude_bonus_mid_min_m": 1500.0,
    "altitude_bonus_high": 0.72,
    "altitude_bonus_mid": 0.43,
    "nose_down_altitude_m": 3200.0,
    "nose_down_pitch_deg": -6.0,
    "nose_down_penalty": -0.9,
    "roll_limit_deg": 80.0,
    "roll_limit_penalty": 0.15,
    "pitch_down_limit_deg": -12.0,
    "pitch_down_penalty": 1.2,
    "pitch_up_limit_deg": 35.0,
    "pitch_up_penalty": 0.25,
    "attack_range_min_m": 250.0,
    "attack_range_max_m": 1500.0,
    "attack_range_bonus": 0.2,
    "far_range_penalty_start_m": 3500.0,
    "far_range_penalty": 0.25,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "crash_penalty": -260.0,
    # New: delta shaping. 0.0 by default (opt-in via YAML) so this module is a
    # strict superset of my_reward.py until these are explicitly set.
    "range_delta_scale": 0.0,
    # New (2026-08-20): asymmetric anti-evasion penalty. Applied ON TOP of
    # range_delta_scale ONLY when distance is increasing (moving away),
    # never when closing -- so "flee" costs more per meter than "approach"
    # earns, directly targeting the fleeing/standoff habit confirmed both in
    # JSBSim eval (huge mean_distance, far_range-dominant episodes) and a
    # live DogFightViewer match (LOS averaged 90-100deg, opponent closed the
    # gap on its own while ownship orbited away). 0.0 by default (opt-in).
    "evasion_penalty_scale": 0.0,
    "ata_delta_scale": 0.0,
    "range_delta_clip_m": 60.0,
    "ata_delta_clip_deg": 10.0,
    # New (2026-08-19): time pressure. 0.0 scale by default (opt-in). Added
    # after stage6pp_engageband_fix_100iter's ideal_range/too_close_m fix
    # (which moved the standoff sweet-spot inside the WEZ band) produced a
    # *new* degenerate equilibrium instead of engagement: the policy found a
    # stable orbit that satisfies the range reward forever without ever
    # entering WEZ angle or crashing/losing, so episodes stopped terminating
    # (Eps barely advanced across 100 iterations). This term makes NOT being
    # in the WEZ increasingly costly the longer an episode runs past a grace
    # period, to break that stalling equilibrium -- ramped and CLIPPED (not
    # accumulated) so it can't runaway like the 2026-08-06 delta-reward crash
    # incident (a per-step magnitude cap, not a growing sum).
    "time_pressure_start_s": 0.0,
    "time_pressure_ramp_s": 12.0,
    "time_pressure_scale": 0.0,
    # New (2026-08-20): sharply-peaked ATA precision bonus, additive on top
    # of the existing linear "ata" term. Motivation: the linear term
    # (1 - ata/90) has a constant gradient everywhere, so it rewards going
    # from 90deg->85deg exactly as much as 5deg->1deg -- but the WEZ needs
    # <=1-3deg, and a live DogFightViewer match showed LOS never converging
    # below 30deg while the scripted opponent (which has direct target
    # velocity telemetry, tactical19 gives our policy the same info -- see
    # observation.py) holds 0-1deg. This term is a Gaussian bump centered on
    # ata=0 with width ata_precision_width_deg, worth up to
    # ata_precision_scale at perfect alignment and decaying fast -- so
    # "close enough" (10-20deg) barely benefits, but the last few degrees
    # into the WEZ pay disproportionately more than the linear term alone
    # ever could. 0.0 by default (opt-in).
    "ata_precision_scale": 0.0,
    "ata_precision_width_deg": 8.0,
    # New (2026-08-20): directional pursuit-heading shaping, ported from
    # my_reward_delta_v1_wez_proximity.py's 2026-08-11 term (this session's
    # best-ever WEZ signal on the old stage6e/turnback line, never tried on
    # the current Stage4-based safe foundation). Rewards this step's actual
    # flight-path direction for pointing at the target's *current* position
    # every step, independent of whether range/ata have started improving
    # yet -- continuous "turn the nose toward the opponent", not a one-shot
    # merge bet. 0.0 by default (opt-in).
    "pursuit_heading_scale": 0.0,
    # Lead-pursuit shaping (ported 2026-08-20 from my_reward_delta_v1_lead_pursuit.py,
    # a 2026-08-12 module that predates today's tactical19/timeout-win-fix
    # changes -- reimplemented here so it stacks with those instead of
    # reverting them). Rewards the ownship's nose for pointing at where the
    # target WILL be (current position + measured per-step displacement *
    # lead_pursuit_horizon_steps), not where it currently is. Motivation:
    # every other angle term (ata, ata_delta, ata_precision, pursuit_heading)
    # scores against the target's CURRENT position, so even a perfect policy
    # under those signals only achieves pure pursuit -- which swings to a
    # large angle right at the merge against ANY moving target (geometry, not
    # a training failure). Matches this session's live-test finding
    # (2026-08-20: v5/v6b/v7/v8 all close to <50m at times but 0 frames ever
    # satisfy WEZ-range + ata<=10deg simultaneously). 0.0 by default (opt-in).
    "lead_pursuit_scale": 0.0,
    "lead_pursuit_horizon_steps": 20,
    "lead_pursuit_velocity_clip_m": 40.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _wez_phase_coeff(distance: float, ata: float, wez_config: dict) -> float:
    """Same phase-selection logic as single_agent_env.py._phase_damage, but
    returning the matching phase's damage_coeff (0.0 if none match) instead
    of a damage amount -- used to grade the wez reward bonus by phase
    instead of a single Phase-1-only binary check."""
    min_range_m = float(wez_config["min_range_m"])
    phases = wez_config.get("phases") or [
        {
            "angle_deg": float(wez_config["angle_deg"]),
            "max_range_m": float(wez_config["max_range_m"]),
            "damage_coeff": 1.0,
        }
    ]
    for phase in phases:
        max_range_m = float(phase["max_range_m"])
        half_angle_deg = float(phase["angle_deg"]) / 2.0
        if min_range_m <= distance <= max_range_m and ata <= half_angle_deg:
            return float(phase.get("damage_coeff", 1.0))
    return 0.0


# Module-level previous-step cache. Safe only for a single sequential env
# instance (this project always runs num-env-runners=1 / num-envs-per-env-runner=1
# for these pilots) -- reset whenever the previous call ended the episode, so a
# fresh episode's first step never sees a stale delta from the last one.
_prev_distance: float | None = None
_prev_ata: float | None = None
_prev_position: np.ndarray | None = None
_prev_target_position: np.ndarray | None = None


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    global _prev_distance, _prev_ata, _prev_position, _prev_target_position

    cfg = {**MY_REWARD_CONFIG, **reward_config}
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    aa = abs(float(geo_info._get_aspect_angle(ownship_state, target_state, False)))
    altitude = float(ownship_state[StateIndex.ALT])
    roll = float(ownship_state[StateIndex.ROLL])
    pitch = float(ownship_state[StateIndex.PITCH])

    components: dict[str, float] = {
        "step": float(cfg["step_penalty"]),
        "survival": float(cfg.get("survival_bonus", 0.0)),
    }

    ideal_min = float(cfg["ideal_range_min_m"])
    ideal_max = float(cfg["ideal_range_max_m"])
    ideal_mid = 0.5 * (ideal_min + ideal_max)
    ideal_half_width = max(1.0, 0.5 * (ideal_max - ideal_min))
    range_score = 1.0 - abs(distance - ideal_mid) / ideal_half_width
    components["range"] = float(cfg["range_scale"]) * max(-4.0, range_score)

    if distance < float(cfg["too_close_m"]):
        ratio = _clamp01(1.0 - distance / max(1.0, float(cfg["too_close_m"])))
        components["overshoot"] = -float(cfg["overshoot_penalty"]) * (
            ratio + float(cfg["overshoot_quadratic_scale"]) * ratio * ratio
        )
    else:
        components["overshoot"] = 0.0
    components["inside_min_range"] = (
        float(cfg["inside_min_range_penalty"])
        if distance < float(wez_config["min_range_m"])
        else 0.0
    )

    components["ata"] = float(cfg["ata_scale"]) * max(-1.0, 1.0 - ata / 90.0)
    ata_precision_scale = float(cfg["ata_precision_scale"])
    if ata_precision_scale != 0.0:
        width = max(1.0, float(cfg["ata_precision_width_deg"]))
        components["ata_precision"] = ata_precision_scale * float(np.exp(-0.5 * (ata / width) ** 2))
    else:
        components["ata_precision"] = 0.0
    components["aa"] = float(cfg["aa_scale"]) * max(-1.0, 1.0 - aa / 180.0)
    # Graduated WEZ bonus (2026-08-19, replaces the old Phase-1-only binary
    # check): the competition's own 3-phase damage cone (Phase1 LOS<1deg
    # coeff1.0, Phase2 LOS<2deg coeff0.3, Phase3 LOS<3deg coeff0.1 -- see
    # single_agent_env.py._phase_damage, same wez_config["phases"] passed
    # here) already deals real damage in Phase 2/3, not just Phase 1, but
    # the old binary wez_bonus only fired for Phase 1's narrow +-1deg cone --
    # leaving a huge reward cliff between "far away" and "essentially
    # boresighted" with almost no signal for the intermediate, still-lethal
    # phases. Reusing the exact same phase table (not an invented threshold)
    # gives partial credit proportional to each phase's own damage_coeff.
    wez_phase_coeff = _wez_phase_coeff(distance, ata, wez_config)
    in_wez = wez_phase_coeff > 0.0
    components["wez"] = float(cfg["wez_bonus"]) * wez_phase_coeff
    components["damage"] = float(cfg["damage_scale"]) * (
        float(target_damage) - float(ownship_damage)
    )

    if float(cfg["altitude_bonus_high_min_m"]) <= altitude <= float(
        cfg["altitude_bonus_high_max_m"]
    ):
        components["altitude"] = float(cfg["altitude_bonus_high"])
    elif float(cfg["altitude_bonus_mid_min_m"]) <= altitude < float(
        cfg["altitude_bonus_high_min_m"]
    ):
        components["altitude"] = float(cfg["altitude_bonus_mid"])
    else:
        components["altitude"] = 0.0

    components["nose_down"] = (
        float(cfg["nose_down_penalty"])
        if altitude < float(cfg["nose_down_altitude_m"])
        and pitch < float(cfg["nose_down_pitch_deg"])
        else 0.0
    )

    safety = 0.0
    if altitude < float(cfg["altitude_soft_floor_m"]):
        safety -= float(cfg["low_altitude_penalty"]) * _clamp01(
            (float(cfg["altitude_soft_floor_m"]) - altitude)
            / max(
                1.0,
                float(cfg["altitude_soft_floor_m"])
                - float(cfg["altitude_hard_floor_m"]),
            )
        )
    if altitude < float(cfg["altitude_hard_floor_m"]):
        safety -= float(cfg["very_low_altitude_penalty"])
    components["safety"] = safety

    control = 0.0
    if abs(roll) > float(cfg["roll_limit_deg"]):
        control -= float(cfg["roll_limit_penalty"]) * _clamp01(
            (abs(roll) - float(cfg["roll_limit_deg"]))
            / max(1.0, 180.0 - float(cfg["roll_limit_deg"]))
        )
    if pitch < float(cfg["pitch_down_limit_deg"]):
        control -= float(cfg["pitch_down_penalty"]) * _clamp01(
            (float(cfg["pitch_down_limit_deg"]) - pitch)
            / max(1.0, 90.0 + float(cfg["pitch_down_limit_deg"]))
        )
    if pitch > float(cfg["pitch_up_limit_deg"]):
        control -= float(cfg["pitch_up_penalty"]) * _clamp01(
            (pitch - float(cfg["pitch_up_limit_deg"]))
            / max(1.0, 90.0 - float(cfg["pitch_up_limit_deg"]))
        )
    components["control_stability"] = control

    attack_min = float(cfg["attack_range_min_m"])
    attack_max = float(cfg["attack_range_max_m"])
    components["attack_range"] = (
        float(cfg["attack_range_bonus"])
        if attack_min <= distance <= attack_max
        else 0.0
    )
    far_start = float(cfg["far_range_penalty_start_m"])
    far_span = float(cfg.get("far_range_penalty_span_m", far_start * 6.0))
    components["far_range"] = (
        -float(cfg["far_range_penalty"])
        * _clamp01((distance - far_start) / max(1.0, far_span))
        if distance > far_start
        else 0.0
    )

    # Delta shaping: reward closing distance / reducing ata step-over-step,
    # independent of the absolute band. Clipped so a single-step teleport
    # (episode reset, geometry-guard snap) can't dominate the reward.
    #
    # Safety gate (2026-08-06 fix): stage6d_delta_reward_400iter collapsed into
    # repeated nose-down crashes at ~285m alt after ~iter100 -- accumulated
    # range_delta reward over many steps (up to range_delta_scale*range_delta_clip_m
    # per step) exceeded the one-time crash_penalty, making "dive fast toward
    # the target, eat the crash" net-positive. Gating delta reward off below
    # altitude_soft_floor_m removes that incentive: closing distance only pays
    # while altitude is still in the safe band, so diving into the danger zone
    # can no longer be paid for by delta reward, only by the (much smaller,
    # already-existing) safety/altitude penalties.
    range_delta_reward = 0.0
    ata_delta_reward = 0.0
    safe_altitude = altitude >= float(cfg["altitude_soft_floor_m"])
    if _prev_distance is not None and safe_altitude:
        range_delta = _prev_distance - distance  # positive = closed distance
        range_delta = max(
            -float(cfg["range_delta_clip_m"]),
            min(float(cfg["range_delta_clip_m"]), range_delta),
        )
        range_delta_reward = float(cfg["range_delta_scale"]) * range_delta
        if range_delta < 0.0:
            # Moving away: charge the extra evasion penalty on top of the
            # normal (negative) range_delta_reward already applied above.
            # range_delta is negative here, so this SUBTRACTS further.
            range_delta_reward += float(cfg["evasion_penalty_scale"]) * range_delta
    if _prev_ata is not None and safe_altitude:
        ata_delta = _prev_ata - ata  # positive = pointed more at target
        ata_delta = max(
            -float(cfg["ata_delta_clip_deg"]),
            min(float(cfg["ata_delta_clip_deg"]), ata_delta),
        )
        ata_delta_reward = float(cfg["ata_delta_scale"]) * ata_delta
    components["range_delta"] = range_delta_reward
    components["ata_delta"] = ata_delta_reward

    # Time pressure: once elapsed sim time passes time_pressure_start_s,
    # ramp a per-step penalty up to -time_pressure_scale over
    # time_pressure_ramp_s while NOT in the WEZ, then hold flat (bounded,
    # not accumulating) so stalling outside the WEZ gets steadily less
    # attractive than closing in, without ever being able to outweigh the
    # one-time terminal rewards/penalties the way an unbounded sum could.
    sim_time = float(ownship_state[StateIndex.SIM_TIME])
    time_pressure_scale = float(cfg["time_pressure_scale"])
    if time_pressure_scale > 0.0 and not in_wez:
        start_s = float(cfg["time_pressure_start_s"])
        ramp_s = max(1.0, float(cfg["time_pressure_ramp_s"]))
        ramp = _clamp01((sim_time - start_s) / ramp_s)
        components["time_pressure"] = -time_pressure_scale * ramp
    else:
        components["time_pressure"] = 0.0

    # Directional pursuit-heading shaping (2026-08-20, ported from
    # my_reward_delta_v1_wez_proximity.py's 2026-08-11 term): rewards this
    # step's actual flight-path direction for pointing at the target's
    # *current* position, continuously and independent of whether ata/range
    # have already improved -- see MY_REWARD_CONFIG comment for rationale.
    # Same safe_altitude gate as range_delta/ata_delta above.
    pursuit_heading_reward = 0.0
    current_position = np.asarray(position_ned(ownship_state), dtype=np.float64)
    if (
        safe_altitude
        and float(cfg["pursuit_heading_scale"]) != 0.0
        and _prev_position is not None
    ):
        displacement = current_position - _prev_position
        displacement_norm = float(np.linalg.norm(displacement))
        target_position = np.asarray(position_ned(target_state), dtype=np.float64)
        los_vector = target_position - current_position
        los_norm = float(np.linalg.norm(los_vector))
        if displacement_norm > 1e-3 and los_norm > 1e-3:
            motion_dir = displacement / displacement_norm
            los_dir = los_vector / los_norm
            closure_alignment = float(np.clip(np.dot(motion_dir, los_dir), -1.0, 1.0))
            pursuit_heading_reward = float(cfg["pursuit_heading_scale"]) * closure_alignment
    components["pursuit_heading"] = pursuit_heading_reward

    # Lead-pursuit shaping: predict the target's position lead_pursuit_horizon_steps
    # ahead from its measured per-step displacement, and reward the ownship's
    # nose (ATA) for pointing at that PREDICTED point instead of the target's
    # current position. Same safe_altitude gate as the other delta/heading
    # terms above. Per-step target displacement is clipped so an episode
    # boundary or scripted_pursuit target reset can't explode the prediction.
    lead_pursuit_reward = 0.0
    target_position = np.asarray(position_ned(target_state), dtype=np.float64)
    lead_pursuit_scale = float(cfg["lead_pursuit_scale"])
    if (
        safe_altitude
        and lead_pursuit_scale != 0.0
        and _prev_target_position is not None
    ):
        target_displacement = target_position - _prev_target_position
        clip_m = float(cfg["lead_pursuit_velocity_clip_m"])
        disp_norm = float(np.linalg.norm(target_displacement))
        if disp_norm > clip_m:
            target_displacement = target_displacement * (clip_m / disp_norm)
        horizon = float(cfg["lead_pursuit_horizon_steps"])
        predicted_target_position = target_position + target_displacement * horizon
        synthetic_target_state = np.zeros_like(target_state)
        synthetic_target_state[:3] = predicted_target_position
        lead_ata = abs(
            float(
                geo_info._get_antenna_train_angle(
                    ownship_state, synthetic_target_state, False
                )
            )
        )
        lead_pursuit_reward = lead_pursuit_scale * max(-1.0, 1.0 - lead_ata / 90.0)
    components["lead_pursuit"] = lead_pursuit_reward

    terminal_reward = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in {"ownship altitude below min", "FDM Update Fail"}:
            terminal_reward = float(cfg["crash_penalty"])
        elif target_health <= 0.0 < ownship_health:
            terminal_reward = float(cfg["win_reward"])
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = float(cfg["loss_reward"])
        elif truncated:
            # Competition rule (slide 11): "200s -- deal more damage than you
            # take to win, even without a kill." A timeout used to always
            # fall through to draw_reward here regardless of who actually
            # took more damage -- matching single_agent_env.py's own
            # _classify_outcome bug (fixed 2026-08-20 alongside this) -- so
            # the policy never got a terminal signal for the single most
            # common real winning condition (almost every training episode
            # ends by timeout, and a live opponent routinely wins on points
            # without a kill). Mirrors _classify_outcome's binary comparison
            # exactly so reward and logged outcome never disagree.
            # Higher remaining health = took LESS damage = ahead on points.
            if ownship_health > target_health:
                terminal_reward = float(cfg["win_reward"])
            elif target_health > ownship_health:
                terminal_reward = float(cfg["loss_reward"])
            else:
                terminal_reward = float(cfg["draw_reward"])
        else:
            terminal_reward = float(cfg["draw_reward"])
    components["terminal"] = terminal_reward

    if terminated or truncated:
        _prev_distance = None
        _prev_ata = None
        _prev_position = None
        _prev_target_position = None
    else:
        _prev_distance = distance
        _prev_ata = ata
        _prev_position = current_position
        _prev_target_position = target_position

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
