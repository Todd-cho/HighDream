"""Two-hour W97 residual recovery reward: learn the live ATA 45--100 band."""

from __future__ import annotations

import numpy as np

from student.my_reward_residual_w56_v1 import (
    MY_REWARD_CONFIG as _BASE_CONFIG,
    compute_reward as _base_reward,
)


MY_REWARD_CONFIG = dict(_BASE_CONFIG)
_prev_ata: float | None = None


def compute_reward(
    ownship_state, target_state, ownship_damage, target_damage, geo_info,
    wez_config, reward_config, terminated, truncated, end_condition,
):
    global _prev_ata
    reward, components = _base_reward(
        ownship_state, target_state, ownship_damage, target_damage, geo_info,
        wez_config, reward_config, terminated, truncated, end_condition,
    )
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(
        ownship_state, target_state, False
    )))
    threat = abs(float(geo_info._get_antenna_train_angle(
        target_state, ownship_state, False
    )))

    range_quality = float(np.exp(-0.5 * ((distance - 1300.0) / 1500.0) ** 2))
    alignment = float(np.cos(np.deg2rad(min(180.0, ata)) / 2.0) ** 4)
    defensive_margin = float(np.sin(np.deg2rad(min(180.0, threat)) / 2.0) ** 2)
    components["coupled_attack_wide"] = 0.55 * range_quality * alignment * defensive_margin

    if _prev_ata is not None and ata <= 105.0 and distance <= 3800.0:
        progress = float(np.clip(_prev_ata - ata, -6.0, 6.0))
        # Potential-difference shaping: rewards reducing ATA, not camping at
        # the widened 90-degree residual gate.
        components["ata_progress_45_100"] = 0.14 * progress
    else:
        components["ata_progress_45_100"] = 0.0

    _prev_ata = None if (terminated or truncated) else ata
    extra = components["coupled_attack_wide"] + components["ata_progress_45_100"]
    return float(reward + extra), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
