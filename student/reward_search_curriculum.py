# -*- coding: utf-8 -*-
"""Controlled altitude-recovery environment for reward parameter search.

This is deliberately a single-stage curriculum.  Reward candidates must see
the same initial-state distribution; the full five-stage curriculum is used
only after a reward configuration has been selected.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage


SCREEN_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "level_5000",
            "weight": 0.60,
            "ownship": [1000.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 280.0],
            "target": [6000.0, 0.0, -5000.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
        {
            "name": "light_disturbance_4500",
            "weight": 0.25,
            "ownship": [1000.0, 0.0, -4500.0, 8.0, -3.0, 8.0, 280.0],
            "target": [6000.0, 0.0, -5000.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
        {
            "name": "mild_dive_4000",
            "weight": 0.15,
            "ownship": [1000.0, 0.0, -4000.0, 10.0, -8.0, 10.0, 280.0],
            "target": [6000.0, 0.0, -4800.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
    ],
    # Small within-pool variation prevents memorizing one exact reset while the
    # stage-level randomization remains disabled as recommended by Release 260721.
    "ownship_randomization": {
        "enabled": True,
        "radius": 80.0,
        "r_roll": 3.0,
        "r_pitch": 2.0,
        "r_heading": 5.0,
        "speed_mps": 5.0,
    },
    "target_randomization": {
        "enabled": True,
        "radius": 80.0,
        "r_heading": 3.0,
        "speed_mps": 4.0,
    },
}


VALIDATION_SCENARIO_POOL = {
    **SCREEN_SCENARIO_POOL,
    "scenarios": [
        {
            "name": "level_5000",
            "weight": 0.30,
            "ownship": [1000.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 280.0],
            "target": [6000.0, 0.0, -5000.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
        {
            "name": "mild_dive_4000",
            "weight": 0.35,
            "ownship": [1000.0, 0.0, -4000.0, 10.0, -8.0, 10.0, 280.0],
            "target": [6000.0, 0.0, -4800.0, 0.0, 0.0, 180.0, 260.0],
            "target_mode": "autopilot",
        },
        {
            "name": "low_dive_3500",
            "weight": 0.20,
            "ownship": [1000.0, 0.0, -3500.0, 12.0, -12.0, -10.0, 275.0],
            "target": [5500.0, 0.0, -4600.0, 0.0, 0.0, 180.0, 255.0],
            "target_mode": "autopilot",
        },
        {
            "name": "banked_descent_3800",
            "weight": 0.15,
            "ownship": [1000.0, 0.0, -3800.0, 35.0, -8.0, 20.0, 275.0],
            "target": [5500.0, 500.0, -4600.0, 0.0, 0.0, 180.0, 255.0],
            "target_mode": "autopilot",
        },
    ],
}


CONFIRM_SCENARIO_POOL = {
    **VALIDATION_SCENARIO_POOL,
    "scenarios": [
        {
            **scenario,
            "target_mode": "loiter",
            "target_loiter": {"enabled": True, "bank": 0.0, "pitch": 0.0},
        }
        for scenario in VALIDATION_SCENARIO_POOL["scenarios"]
    ],
}

# Official template-compatible name: the easy screening distribution.
MY_SCENARIO_POOL = SCREEN_SCENARIO_POOL


def get_stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(
            index=0,
            name="altitude_reward_search",
            description="Easy level-flight-first distribution for reward screening.",
            target_mode="autopilot",
            episode_step_limit=3600,
            max_iterations=20,
            checkpoint_interval=10,
            reward_overrides={},
            randomization={"enabled": False},
            env_overrides={"initial_scenario": SCREEN_SCENARIO_POOL},
            advance_conditions={},
            advance_window=10,
        ),
        CurriculumStage(
            index=1,
            name="altitude_reward_validation",
            description="Moderate dive and bank recovery validation distribution.",
            target_mode="autopilot",
            episode_step_limit=3600,
            max_iterations=50,
            checkpoint_interval=10,
            reward_overrides={},
            randomization={"enabled": False},
            env_overrides={"initial_scenario": VALIDATION_SCENARIO_POOL},
            advance_conditions={},
            advance_window=10,
        ),
        CurriculumStage(
            index=2,
            name="altitude_reward_confirm",
            description="Final C05/C07/C11 check with a level loiter target.",
            target_mode="loiter",
            episode_step_limit=3600,
            max_iterations=100,
            checkpoint_interval=10,
            reward_overrides={},
            randomization={"enabled": False},
            env_overrides={"initial_scenario": CONFIRM_SCENARIO_POOL},
            advance_conditions={},
            advance_window=10,
        ),
    ]


__all__ = [
    "MY_SCENARIO_POOL",
    "SCREEN_SCENARIO_POOL",
    "VALIDATION_SCENARIO_POOL",
    "CONFIRM_SCENARIO_POOL",
    "get_stages",
]
