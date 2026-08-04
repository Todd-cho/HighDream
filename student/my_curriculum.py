# -*- coding: utf-8 -*-
"""Minimal student curriculum template.

Use this file only when you want to replace the built-in curriculum.
Run with:
  python train_curriculum.py --stages-module student.my_curriculum

For the full built-in curriculum and two-circle head-on reference, see:
  src/dogfight/ai/curriculum.py
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

MY_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "head_on_bt",
            "weight": 0.5,
            "ownship": [1000.0, 0.0, -7000.0, 0.0, 0.0, 0.0, 300.0],
            "target": [6000.0, 0.0, -7000.0, 0.0, 0.0, 180.0, 300.0],
            "target_mode": "behavior_tree",
        },
        {
            "name": "offset_loiter",
            "weight": 0.5,
            "ownship": [1000.0, -1000.0, -6800.0, 0.0, 0.0, 30.0, 280.0],
            "target": [5000.0, 1000.0, -7200.0, 0.0, 0.0, 180.0, 270.0],
            "target_mode": "loiter",
            "target_loiter": {"enabled": True, "bank": 35.0, "pitch": 0.0},
        },
    ],
    "ownship_randomization": {
        "enabled": True,
        "radius": 200.0,
        "r_roll": 5.0,
        "r_pitch": 5.0,
        "r_heading": 15.0,
        "speed_mps": 10.0,
    },
    "target_randomization": {
        "enabled": True,
        "radius": 200.0,
        "r_roll": 5.0,
        "r_pitch": 5.0,
        "r_heading": 15.0,
        "speed_mps": 10.0,
    },
    "shared_randomization": {
        "enabled": True,
        "n_m": 500.0,
        "e_m": 500.0,
        "d_m": 100.0,
        "speed_mps": 5.0,
    },
}


def get_stages() -> list[CurriculumStage]:
    """Return a small runnable curriculum example."""
    return [
        CurriculumStage(
            index=0,
            name="student_fixed_target",
            description="Minimal fixed-target warmup stage.",
            target_mode="fixed",
            episode_step_limit=3600,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={},
            randomization={
                "enabled": True,
                "radius": 500.0,
                "r_roll": 5.0,
                "r_pitch": 5.0,
                "r_heading": 30.0,
            },
            advance_conditions={},
            advance_window=5,
        ),
        CurriculumStage(
            index=1,
            name="student_bt_dogfight",
            description="Minimal behavior-tree opponent stage.",
            target_mode="behavior_tree",
            episode_step_limit=18000,
            max_iterations=20,
            checkpoint_interval=5,
            reward_overrides={},
            randomization={"enabled": False},
            env_overrides={"initial_scenario": MY_SCENARIO_POOL},
            advance_conditions={},
            advance_window=5,
        ),
    ]


__all__ = ["MY_SCENARIO_POOL", "get_stages"]
