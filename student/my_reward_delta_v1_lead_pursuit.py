# -*- coding: utf-8 -*-
"""Copy of student/my_reward_delta_v1.py plus a predictive "lead pursuit"
shaping term: reward the ownship's nose for pointing at where the opponent
WILL be, not where it currently is.

Motivation (2026-08-12, stage6j_scripted_pursuit_opponent_50iter result):
swapping the training opponent from passive "loiter" to the genuinely active
ScriptedPursuitActionProvider, with reward left completely unchanged, did NOT
teach any lead-pursuit behavior -- frozen eval wez_episode_rate/target_damage
stayed at 0%/0% (identical to before retraining), and safety_override fired
in 100% of episodes (vs the untrained baseline's occasional trigger),
suggesting the policy got *more* unstable trying to chase a moving target
with only reactive signals. Confirms the earlier trajectory analysis
(stage6e_delta_down_400iter vs scripted_pursuit, seed 269001): the ownship
tracks well at range (t=5.1s, 1866m, ata=16.8deg) but ata degrades as it
closes (t=7.5s, 452m, ata=42.4deg), passing almost perpendicular at closest
approach (t=8.1s, 269.5m, ata=94.5deg) -- textbook pure pursuit (nose always
chasing the target's *current* bearing) with no lead.

Every existing angle-based reward term in this project (the static "ata"
component, ata_delta, ata_recovery, pursuit_heading in the wez_proximity
module) scores against the target's *current* position. None of them can, by
construction, teach lead: even a perfect policy under those signals would
still point at the current bearing, and pure pursuit against ANY moving
target inevitably swings to a large angle right at the merge (geometry, not a
training failure) -- which is exactly the pattern observed.

This module adds a term that scores against a PREDICTED future position
instead:
    target_velocity_per_step = target_position(t) - target_position(t-1)   # NED, actual measured displacement, not commanded heading
    predicted_target_position = target_position(t) + target_velocity_per_step * lead_pursuit_horizon_steps
    lead_ata = antenna_train_angle(ownship_state, predicted_target_position)   # reuses GeoMathUtil's existing, already-verified geometry --
                                                                                # only position (index 0:3) is read for the target side, roll/pitch/yaw
                                                                                # are not, so a synthetic array with only position filled in is valid input.
    lead_pursuit_reward = lead_pursuit_scale * max(-1.0, 1.0 - lead_ata / 90.0)
Same normalization/shape as the existing static "ata" component (see
components["ata"] below), just scored against the extrapolated point instead
of the live one -- directly, densely rewards "aim ahead of the target's
motion" every step, which the BFM lead-pursuit concept (rules deck's BT
tutorial) is built on, rather than requiring the policy to discover it via a
sparse/absent signal.

lead_pursuit_horizon_steps is a fixed step count (not an adaptive
time-to-intercept estimate, to avoid a new division-by-near-zero-closure-rate
failure mode as a first version) -- default 20 steps. This env runs at
step_ratio=6/sim_hz=60 = 0.1s of sim time per RL step (confirmed via
stage6i_postmerge_recovery.py's "150 steps = 15s" post-merge window), so 20
steps = ~2s of lead, a reasonable fraction of the ~3s the trajectory probe
showed between "well-aimed at range" (t=5.1s) and "closest approach"
(t=8.1s). Per-step target displacement is clipped
(lead_pursuit_velocity_clip_m, default 40m -- above the target's real
per-step displacement at its ~250-260kcas cruise speed, ~25-26m/step) so an
episode-boundary position snap or scripted_pursuit target reset can't
momentarily explode the extrapolated point.

Gated by the same safe_altitude check as range_delta/ata_delta/pursuit_heading
(see my_reward_delta_v1.py's 2026-08-06 fix note): closing/aiming shaping
only pays while altitude is still in the safe band, so this cannot become a
new dive-and-crash incentive. Bounded per-step magnitude (scale * [-1, 1]),
same as ata/pursuit_heading, for the same reason.

lead_pursuit_scale defaults to 0.0 (opt-in via YAML), so this module is a
strict superset of my_reward_delta_v1.py until explicitly set. Everything
else is byte-for-byte identical to that module.
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
    # New: delta shaping. 0.0 by default (opt-in via YAML) so this module is a
    # strict superset of my_reward.py until these are explicitly set.
    "range_delta_scale": 0.0,
    "ata_delta_scale": 0.0,
    "range_delta_clip_m": 60.0,
    "ata_delta_clip_deg": 10.0,
    # New (2026-08-12): predictive lead-pursuit shaping -- see module
    # docstring. 0.0 by default (opt-in via YAML).
    "lead_pursuit_scale": 0.0,
    "lead_pursuit_horizon_steps": 20,
    "lead_pursuit_velocity_clip_m": 40.0,
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

    # Predictive lead-pursuit shaping (2026-08-12): rewards the ownship's
    # NOSE (not flight path -- that's pursuit_heading, a different, already
    # existing term) for pointing at where the target's own measured
    # per-step motion says it will be lead_pursuit_horizon_steps from now,
    # rather than where it is right now. See module docstring for full
    # rationale (this is the term diagnosed as entirely missing from every
    # prior reward module in this project).
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
        lead_pursuit_reward = float(cfg["lead_pursuit_scale"]) * max(
            -1.0, 1.0 - lead_ata / 90.0
        )
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
