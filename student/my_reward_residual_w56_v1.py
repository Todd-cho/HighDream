# -*- coding: utf-8 -*-
"""Reward module for the W56-based residual RL pilot
(RL_TRAINING_ADDENDUM_W56_KO section 7 priority order):

  1) actual damage exchange: target_damage - own_damage (dominant)
  2) our WEZ occupancy time - opponent's WEZ occupancy time on us
  3) alignment (ATA) reward, gated by threat_ata so a mutual head-on where
     the opponent also has us boresighted is not rewarded for tight ATA
     (addendum's W57 case: ATA 0.76deg at 706m, but ~70% damage taken vs
     ~30% dealt -- an unfavorable trade the old ATA-only reward could not see)
  4) appropriate range/closure maintenance
  5) overshoot / minimum-range violation suppression
  6) safe altitude and energy
  (residual magnitude/delta penalty is applied by ResidualW56Env.step()
  itself, not here -- this hook's signature has no access to the RL action)

Unconditional lead-pursuit reward is deliberately NOT implemented here per
both handoff documents' repeated prohibition.
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

from dogfight.sim.state_schema import StateIndex


# Module-level previous-step cache, matching student/my_reward_delta_v1.py's
# established pattern. Safe only for a single sequential env instance (this
# project always runs num-env-runners=1 for these pilots).
_prev_distance: float | None = None


MY_REWARD_CONFIG = {
    "damage_scale": 25.0,
    "wez_bonus": 1.0,
    "opponent_wez_penalty": 1.2,
    "ata_scale": 0.10,
    "ata_precision_scale": 0.15,
    "ata_precision_width_deg": 8.0,
    "favorable_threat_ata_low_deg": 20.0,
    "favorable_threat_ata_high_deg": 60.0,
    "aa_scale": 0.02,
    "ideal_range_min_m": 450.0,
    "ideal_range_max_m": 1100.0,
    "range_scale": 0.6,
    "too_close_m": 460.0,
    "overshoot_penalty": 4.0,
    "overshoot_quadratic_scale": 1.5,
    "inside_min_range_penalty": -3.0,
    "far_range_penalty_start_m": 3500.0,
    "far_range_penalty": 0.25,
    "closure_target_delta_m": 6.0,
    "closure_scale": 0.02,
    "closure_delta_clip_m": 60.0,
    "altitude_soft_floor_m": 3800.0,
    "altitude_hard_floor_m": 1800.0,
    "low_altitude_penalty": 1.76,
    "very_low_altitude_penalty": 5.49,
    "step_penalty": -0.003,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "crash_penalty": -260.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _wez_phase_coeff(distance: float, angle: float, wez_config: dict) -> float:
    """Same phase table single_agent_env.py uses for real WEZ damage, applied
    to either our ATA (our WEZ) or the opponent's ATA-on-us / threat_ata
    (opponent's WEZ on us) depending on the caller."""
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
        if min_range_m <= distance <= max_range_m and angle <= half_angle_deg:
            return float(phase.get("damage_coeff", 1.0))
    return 0.0


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
    global _prev_distance
    cfg = {**MY_REWARD_CONFIG, **reward_config}
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    aa = abs(float(geo_info._get_aspect_angle(ownship_state, target_state, False)))
    threat_ata = abs(float(geo_info._get_antenna_train_angle(target_state, ownship_state, False)))
    altitude = float(ownship_state[StateIndex.ALT])

    components: dict[str, float] = {"step": float(cfg["step_penalty"])}

    # 1) Damage exchange -- dominant term.
    components["damage"] = float(cfg["damage_scale"]) * (
        float(target_damage) - float(ownship_damage)
    )

    # 2) Our WEZ occupancy vs. the opponent's WEZ occupancy on us.
    our_phase = _wez_phase_coeff(distance, ata, wez_config)
    their_phase = _wez_phase_coeff(distance, threat_ata, wez_config)
    components["wez"] = float(cfg["wez_bonus"]) * our_phase
    components["opponent_wez"] = -float(cfg["opponent_wez_penalty"]) * their_phase

    # 3) Alignment, gated by how favorable threat_ata is. A tight ATA solution
    # earned while the opponent also has us boresighted (small threat_ata) is
    # not rewarded -- see module docstring / addendum W57 finding.
    low = float(cfg["favorable_threat_ata_low_deg"])
    high = float(cfg["favorable_threat_ata_high_deg"])
    favorability = _clamp01((threat_ata - low) / max(1.0, high - low))
    components["ata"] = (
        float(cfg["ata_scale"]) * max(-1.0, 1.0 - ata / 90.0) * favorability
    )
    width = max(1.0, float(cfg["ata_precision_width_deg"]))
    components["ata_precision"] = (
        float(cfg["ata_precision_scale"])
        * float(np.exp(-0.5 * (ata / width) ** 2))
        * favorability
    )
    components["aa"] = float(cfg["aa_scale"]) * max(-1.0, 1.0 - aa / 180.0) * favorability

    # 4) Range and closure maintenance.
    ideal_min = float(cfg["ideal_range_min_m"])
    ideal_max = float(cfg["ideal_range_max_m"])
    ideal_mid = 0.5 * (ideal_min + ideal_max)
    ideal_half_width = max(1.0, 0.5 * (ideal_max - ideal_min))
    range_score = 1.0 - abs(distance - ideal_mid) / ideal_half_width
    components["range"] = float(cfg["range_scale"]) * max(-4.0, range_score)
    far_start = float(cfg["far_range_penalty_start_m"])
    components["far_range"] = (
        -float(cfg["far_range_penalty"]) * _clamp01((distance - far_start) / max(1.0, far_start))
        if distance > far_start
        else 0.0
    )
    # Closure management, via a per-step distance-delta proxy (this hook gets
    # no dt/velocity directly -- same module-level-cache approach as
    # student/my_reward_delta_v1.py's range_delta term). Only shaped inside
    # engagement range: penalize closing much faster than a comfortable
    # per-step delta so residual doesn't fly a high-closure pass-through.
    if _prev_distance is not None and distance <= far_start:
        closing_delta = _prev_distance - distance  # positive = closed distance
        closing_delta = float(np.clip(
            closing_delta,
            -float(cfg["closure_delta_clip_m"]),
            float(cfg["closure_delta_clip_m"]),
        ))
        target_delta = float(cfg["closure_target_delta_m"])
        components["closure"] = -float(cfg["closure_scale"]) * max(0.0, closing_delta - target_delta)
    else:
        components["closure"] = 0.0

    # 5) Overshoot / minimum-range violation suppression.
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

    # 6) Safe altitude and energy.
    safety = 0.0
    if altitude < float(cfg["altitude_soft_floor_m"]):
        safety -= float(cfg["low_altitude_penalty"]) * _clamp01(
            (float(cfg["altitude_soft_floor_m"]) - altitude)
            / max(1.0, float(cfg["altitude_soft_floor_m"]) - float(cfg["altitude_hard_floor_m"]))
        )
    if altitude < float(cfg["altitude_hard_floor_m"]):
        safety -= float(cfg["very_low_altitude_penalty"])
    components["safety"] = safety

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
        elif ownship_health > target_health:
            terminal_reward = float(cfg["win_reward"])
        elif target_health > ownship_health:
            terminal_reward = float(cfg["loss_reward"])
        else:
            terminal_reward = float(cfg["draw_reward"])
    components["terminal"] = terminal_reward

    _prev_distance = None if (terminated or truncated) else distance

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
