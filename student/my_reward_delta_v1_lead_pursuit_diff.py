# -*- coding: utf-8 -*-
"""Copy of student/my_reward_delta_v1_lead_pursuit.py with the lead-pursuit
term redesigned from a raw score into a DIFFERENTIAL score, to fix the
redundancy diagnosed 2026-08-12 in scripts/adhoc_trajectory_probe_stage6k.py.

Background: my_reward_delta_v1_lead_pursuit.py's lead_pursuit_reward scored
the predicted future target position the same way the existing static "ata"
component scores the current one:
    lead_pursuit_reward = lead_pursuit_scale * max(-1.0, 1.0 - lead_ata / 90.0)
Step-level trajectory replay of the resulting checkpoint
(stage6k_lead_pursuit_scale020_50iter) showed this is almost always
REDUNDANT with the existing ata term rather than adding new information: at
t=13.5s of a representative episode, current ata was 0.1deg (already
near-perfect) while the back-computed lead_ata was ~15.9deg (worse -- the
target's own banking motion was rotating the predicted point away from
alignment) -- yet the raw-score design still paid +0.165 reward at that
exact moment, on top of the static ata term's own near-maximum payout for
the same near-zero ata. Because lead_ata tracks current ata fairly closely
except during genuine target maneuvers, this raw scoring roughly DOUBLED the
existing angle-correction incentive most of the time instead of teaching
anything lead-specific. The resulting checkpoint committed to a much longer,
harder knife-edge bank (~85-90deg sustained for 13+ seconds, vs the shorter
recoveries seen in every prior checkpoint) trying to satisfy both
near-duplicate signals -- and at ~90deg bank the wings can't supply vertical
lift, so altitude bled continuously and every one of 20 frozen-eval episodes
ended up pinned against the safety-override hard floor (minimum_altitude
397-618m, vs stage6j's 1750-4184m) for zero WEZ/damage improvement.

This version replaces the raw score with a DIFFERENTIAL one that only pays
when aiming at the predicted point is actually BETTER than aiming at the
target's current position:
    diff_deg = max(0.0, ata - lead_ata)   # positive only when the lead point is more aligned than the live one
    diff_deg = min(diff_deg, lead_pursuit_diff_clip_deg)   # clipped, same rationale as range_delta_clip_m/ata_delta_clip_deg
    lead_pursuit_reward = lead_pursuit_scale * diff_deg / 90.0
When the target isn't maneuvering in a way that makes leading pay off
(lead_ata >= ata, the common case per the trajectory replay above), this is
exactly 0.0 -- no double-counting with the static ata term, which already
fully rewards good *current* alignment on its own. It only produces gradient
in the specific situation lead pursuit is meant to teach: the target's
motion means aiming ahead of it is measurably better than chasing its
current bearing. lead_pursuit_scale is set higher than the raw version's
0.2 (0.3 as a starting pilot value) because this term is 0 most of the time
by construction, so a comparable *typical* nonzero payout needs a larger
coefficient than a term that pays out continuously.

Otherwise identical to my_reward_delta_v1_lead_pursuit.py -- same
target-velocity extrapolation (single-step displacement x
lead_pursuit_horizon_steps, clipped), same reuse of GeoMathUtil's
antenna-train-angle geometry via a synthetic position-only target array, same
altitude gating (inactive below altitude_soft_floor_m) and bounded per-step
magnitude for the same crash-incentive-avoidance reasons documented there.

lead_pursuit_scale defaults to 0.0 (opt-in via YAML), so this module is a
strict superset of my_reward_delta_v1.py until explicitly set.
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

from GeoMathUtil import GeometryInfo
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
    "range_delta_scale": 0.0,
    "ata_delta_scale": 0.0,
    "range_delta_clip_m": 60.0,
    "ata_delta_clip_deg": 10.0,
    # Predictive lead-pursuit shaping, differential version (2026-08-12) --
    # see module docstring. 0.0 by default (opt-in via YAML).
    "lead_pursuit_scale": 0.0,
    "lead_pursuit_horizon_steps": 20,
    "lead_pursuit_velocity_clip_m": 40.0,
    "lead_pursuit_diff_clip_deg": 45.0,
}

_geo = GeometryInfo()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# Module-level previous-step cache. Safe only for a single sequential env
# instance (this project always runs num-env-runners=1 / num-envs-per-env-runner=1
# for these pilots) -- reset whenever the previous call ended the episode, so a
# fresh episode's first step never sees a stale delta from the last one.
_prev_distance: float | None = None
_prev_ata: float | None = None
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
    global _prev_distance, _prev_ata, _prev_target_position

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
    components["aa"] = float(cfg["aa_scale"]) * max(-1.0, 1.0 - aa / 180.0)
    in_wez = (
        float(wez_config["min_range_m"]) <= distance <= float(wez_config["max_range_m"])
        and ata <= float(wez_config["angle_deg"]) / 2.0
    )
    components["wez"] = float(cfg["wez_bonus"]) if in_wez else 0.0
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
    if _prev_ata is not None and safe_altitude:
        ata_delta = _prev_ata - ata  # positive = pointed more at target
        ata_delta = max(
            -float(cfg["ata_delta_clip_deg"]),
            min(float(cfg["ata_delta_clip_deg"]), ata_delta),
        )
        ata_delta_reward = float(cfg["ata_delta_scale"]) * ata_delta
    components["range_delta"] = range_delta_reward
    components["ata_delta"] = ata_delta_reward

    # Predictive lead-pursuit shaping, DIFFERENTIAL version (2026-08-12): only
    # pays when the target's extrapolated future position is a BETTER aim
    # point than its current one -- see module docstring for the full
    # rationale (fixes the raw-scored version's redundancy with the static
    # "ata" component above, diagnosed via step-level trajectory replay).
    lead_pursuit_reward = 0.0
    target_position = np.asarray(position_ned(target_state), dtype=np.float64)
    if (
        safe_altitude
        and float(cfg["lead_pursuit_scale"]) != 0.0
        and _prev_target_position is not None
    ):
        target_velocity_step = target_position - _prev_target_position
        clip_m = float(cfg["lead_pursuit_velocity_clip_m"])
        speed = float(np.linalg.norm(target_velocity_step))
        if speed > clip_m and speed > 1e-9:
            target_velocity_step = target_velocity_step * (clip_m / speed)
        horizon = float(cfg["lead_pursuit_horizon_steps"])
        predicted_target_position = target_position + target_velocity_step * horizon
        # Only position (index 0:3) is read by _get_antenna_train_angle for
        # the target-side argument -- pad the rest with zeros, they are
        # never consulted (verified against GeoMathUtil.py's implementation).
        synthetic_target_ned = np.zeros(6, dtype=np.float64)
        synthetic_target_ned[0:3] = predicted_target_position
        lead_ata = abs(float(
            _geo._get_antenna_train_angle(ownship_state, synthetic_target_ned, False)
        ))
        diff_deg = max(0.0, ata - lead_ata)  # positive only if leading beats aiming at the live position
        diff_deg = min(diff_deg, float(cfg["lead_pursuit_diff_clip_deg"]))
        lead_pursuit_reward = float(cfg["lead_pursuit_scale"]) * diff_deg / 90.0
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
        else:
            terminal_reward = float(cfg["draw_reward"])
    components["terminal"] = terminal_reward

    if terminated or truncated:
        _prev_distance = None
        _prev_ata = None
        _prev_target_position = None
    else:
        _prev_distance = distance
        _prev_ata = ata
        _prev_target_position = target_position

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
