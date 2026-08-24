"""W97 residual reward: trade first, then learn ATA 20--45 acquisition."""
from __future__ import annotations

import numpy as np

from student.my_reward_residual_w56_v1 import (
    MY_REWARD_CONFIG as _W56_REWARD_CONFIG,
    compute_reward as _base_reward,
)


# Required by student_hooks.load_reward_hook(). Base shaping is intentionally
# inherited; W97-specific coupled/progress terms are calculated below.
MY_REWARD_CONFIG = dict(_W56_REWARD_CONFIG)


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
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    threat = abs(float(geo_info._get_antenna_train_angle(target_state, ownship_state, False)))

    # Coupled gun-quality potential: useful only at relevant range and when
    # the opponent is not simultaneously nose-on. This mirrors W97's planner
    # objective and avoids teaching a suicidal head-on ATA solution.
    range_quality = float(np.exp(-0.5 * ((distance - 1000.0) / 1000.0) ** 2))
    attack_angle = float(np.cos(np.deg2rad(min(180.0, ata)) / 2.0) ** 4)
    safety_angle = float(np.sin(np.deg2rad(min(180.0, threat)) / 2.0) ** 2)
    components["coupled_attack"] = 0.75 * range_quality * attack_angle * safety_angle

    # Directly reinforce progress through the previously untrained 20--45deg
    # band. Potential difference prevents reward for merely camping at 45deg.
    if _prev_ata is not None and ata <= 50.0 and distance <= 2800.0:
        progress = float(np.clip(_prev_ata - ata, -8.0, 8.0))
        components["ata_progress_20_45"] = 0.10 * progress
    else:
        components["ata_progress_20_45"] = 0.0

    _prev_ata = None if (terminated or truncated) else ata
    extra = components["coupled_attack"] + components["ata_progress_20_45"]
    return float(reward + extra), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
