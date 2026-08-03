# -*- coding: utf-8 -*-
"""Altitude-first curriculum followed by pursuit, WEZ, and BT dogfight."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage


# Official 260721 student template name.  Stages 0-1 use this altitude-recovery
# pool; keep RECOVERY_POOL below as a compatibility alias for older references.
MY_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "level_7000",
            "weight": 0.40,
            "ownship": [1000.0, 0.0, -7000.0, 0.0, 0.0, 0.0, 280.0],
            "target": [7000.0, 0.0, -7000.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
        {
            "name": "mild_dive_6000",
            "weight": 0.30,
            "ownship": [1000.0, 0.0, -6000.0, 15.0, -8.0, 15.0, 280.0],
            "target": [7000.0, 0.0, -6500.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
        {
            "name": "banked_5000",
            "weight": 0.30,
            "ownship": [1000.0, 0.0, -5000.0, -30.0, 3.0, -20.0, 270.0],
            "target": [6500.0, 500.0, -5500.0, 0.0, 0.0, 170.0, 250.0],
            "target_mode": "autopilot",
        },
    ],
    "ownship_randomization": {
        "enabled": True,
        "radius": 150.0,
        "r_roll": 5.0,
        "r_pitch": 3.0,
        "r_heading": 10.0,
        "speed_mps": 8.0,
    },
    "target_randomization": {
        "enabled": True,
        "radius": 100.0,
        "r_heading": 5.0,
        "speed_mps": 5.0,
    },
}

RECOVERY_POOL = MY_SCENARIO_POOL


ATTACK_POOL = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "autopilot_headon",
            "weight": 0.40,
            "ownship": [1000.0, 0.0, -7000.0, 0.0, 0.0, 0.0, 290.0],
            "target": [6000.0, 0.0, -7000.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
        {
            "name": "loiter_offset_left",
            "weight": 0.30,
            "ownship": [1000.0, -800.0, -6500.0, 0.0, 0.0, 20.0, 280.0],
            "target": [5000.0, 800.0, -6800.0, 0.0, 0.0, 180.0, 250.0],
            "target_mode": "loiter",
            "target_loiter": {"enabled": True, "bank": 25.0, "pitch": 0.0},
        },
        {
            "name": "loiter_offset_right",
            "weight": 0.30,
            "ownship": [1000.0, 800.0, -6500.0, 0.0, 0.0, -20.0, 280.0],
            "target": [5000.0, -800.0, -6800.0, 0.0, 0.0, 180.0, 250.0],
            "target_mode": "loiter",
            "target_loiter": {"enabled": True, "bank": -25.0, "pitch": 0.0},
        },
    ],
    "ownship_randomization": {
        "enabled": True,
        "radius": 300.0,
        "r_roll": 7.0,
        "r_pitch": 4.0,
        "r_heading": 15.0,
        "speed_mps": 10.0,
    },
    "target_randomization": {
        "enabled": True,
        "radius": 250.0,
        "r_heading": 10.0,
        "speed_mps": 8.0,
    },
}


def get_stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(
            index=0,
            name="altitude_survival",
            description="Learn level flight without pursuit pressure.",
            target_mode="autopilot",
            episode_step_limit=3600,
            max_iterations=300,
            checkpoint_interval=10,
            reward_overrides={
                "step_penalty": 0.0,
                "survival_bonus": 0.02,
                "range_scale": 0.0,
                "overshoot_penalty": 0.0,
                "inside_min_range_penalty": 0.0,
                "ata_scale": 0.0,
                "aa_scale": 0.0,
                "wez_bonus": 0.0,
                "damage_scale": 0.0,
                "altitude_bonus_high": 0.30,
                "altitude_bonus_mid": 0.12,
                "nose_down_altitude_m": 4500.0,
                "nose_down_pitch_deg": -6.0,
                "nose_down_penalty": -1.2,
                "attack_range_bonus": 0.0,
                "far_range_penalty": 0.0,
                "win_reward": 0.0,
                "loss_reward": -100.0,
                "draw_reward": 0.0,
                "crash_penalty": -260.0,
            },
            randomization={"enabled": False},
            env_overrides={"initial_scenario": RECOVERY_POOL},
            advance_conditions={"crash_rate_max": 0.20},
            advance_window=10,
        ),
        CurriculumStage(
            index=1,
            name="attitude_recovery",
            description="Recover from mild dive and bank while preserving altitude.",
            target_mode="autopilot",
            episode_step_limit=4800,
            max_iterations=300,
            checkpoint_interval=10,
            reward_overrides={
                "step_penalty": -0.001,
                "survival_bonus": 0.01,
                "range_scale": 0.05,
                "ata_scale": 0.02,
                "aa_scale": 0.0,
                "wez_bonus": 0.0,
                "damage_scale": 0.0,
                "altitude_bonus_high": 0.40,
                "altitude_bonus_mid": 0.20,
                "nose_down_altitude_m": 4200.0,
                "nose_down_pitch_deg": -6.0,
                "nose_down_penalty": -1.0,
                "attack_range_bonus": 0.0,
                "far_range_penalty": 0.05,
                "win_reward": 0.0,
                "loss_reward": -100.0,
                "draw_reward": 0.0,
            },
            randomization={"enabled": False},
            env_overrides={"initial_scenario": RECOVERY_POOL},
            advance_conditions={"crash_rate_max": 0.25},
            advance_window=10,
        ),
        CurriculumStage(
            index=2,
            name="safe_pursuit",
            description="Close on an altitude-holding target without diving away.",
            target_mode="autopilot",
            episode_step_limit=7200,
            max_iterations=400,
            checkpoint_interval=10,
            reward_overrides={
                "step_penalty": -0.002,
                "survival_bonus": 0.005,
                "range_scale": 0.35,
                "ata_scale": 0.08,
                "aa_scale": 0.02,
                "wez_bonus": 0.10,
                "damage_scale": 0.0,
                "altitude_bonus_high": 0.55,
                "altitude_bonus_mid": 0.25,
                "attack_range_bonus": 0.10,
                "far_range_penalty_start_m": 4000.0,
                "far_range_penalty": 0.20,
                "win_reward": 20.0,
                "loss_reward": -100.0,
                "draw_reward": -5.0,
            },
            randomization={"enabled": False},
            env_overrides={"initial_scenario": ATTACK_POOL},
            advance_conditions={
                "crash_rate_max": 0.30,
                "ep_min_distance_max": 2500.0,
            },
            advance_window=10,
        ),
        CurriculumStage(
            index=3,
            name="safe_wez",
            description="Build repeatable WEZ contact against loiter targets.",
            target_mode="loiter",
            episode_step_limit=10800,
            max_iterations=500,
            checkpoint_interval=10,
            reward_overrides={
                "step_penalty": -0.003,
                "survival_bonus": 0.0,
                "range_scale": 0.60,
                "ata_scale": 0.12,
                "aa_scale": 0.03,
                "wez_bonus": 0.75,
                "damage_scale": 10.0,
                "altitude_bonus_high": 0.65,
                "altitude_bonus_mid": 0.30,
                "attack_range_bonus": 0.20,
                "far_range_penalty": 0.25,
                "win_reward": 100.0,
                "loss_reward": -100.0,
                "draw_reward": -20.0,
            },
            randomization={"enabled": False},
            env_overrides={"initial_scenario": ATTACK_POOL},
            advance_conditions={
                "crash_rate_max": 0.35,
                "ep_wez_steps_min": 3.0,
            },
            advance_window=10,
        ),
        CurriculumStage(
            index=4,
            name="bt_dogfight",
            description="Final engagement against the behavior-tree opponent.",
            target_mode="behavior_tree",
            episode_step_limit=18000,
            max_iterations=800,
            checkpoint_interval=10,
            reward_overrides={},
            randomization={
                "enabled": True,
                "radius": 1000.0,
                "r_roll": 10.0,
                "r_pitch": 5.0,
                "r_heading": 90.0,
            },
            advance_conditions={},
            advance_window=10,
        ),
    ]


__all__ = ["MY_SCENARIO_POOL", "RECOVERY_POOL", "ATTACK_POOL", "get_stages"]
