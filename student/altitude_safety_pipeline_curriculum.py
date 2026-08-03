from __future__ import annotations

import copy
import os

from dogfight.ai.curriculum import CurriculumStage


R08 = {
    "altitude_soft_floor_m": 3000.0,
    "altitude_hard_floor_m": 1000.0,
    "low_altitude_penalty": 0.90,
    "very_low_altitude_penalty": 2.88,
    "altitude_bonus_high_min_m": 3000.0,
    "altitude_bonus_high_max_m": 9000.0,
    "altitude_bonus_mid_min_m": 1000.0,
    "altitude_bonus_high": 0.60,
    "altitude_bonus_mid": 0.25,
    "nose_down_altitude_m": 3800.0,
    "nose_down_pitch_deg": -8.0,
    "nose_down_penalty": -1.40,
    "crash_penalty": -280.0,
}

C03 = {
    "altitude_soft_floor_m": 2600.0,
    "altitude_hard_floor_m": 1800.0,
    "low_altitude_penalty": 1.34,
    "very_low_altitude_penalty": 5.74,
    "altitude_bonus_high_min_m": 2600.0,
    "altitude_bonus_high_max_m": 9000.0,
    "altitude_bonus_mid_min_m": 1800.0,
    "altitude_bonus_high": 0.72,
    "altitude_bonus_mid": 0.22,
    "nose_down_altitude_m": 2900.0,
    "nose_down_pitch_deg": -8.0,
    "nose_down_penalty": -1.11,
    "crash_penalty": -280.0,
}

C10 = {
    "altitude_soft_floor_m": 3400.0,
    "altitude_hard_floor_m": 1800.0,
    "low_altitude_penalty": 1.15,
    "very_low_altitude_penalty": 3.45,
    "altitude_bonus_high_min_m": 3400.0,
    "altitude_bonus_high_max_m": 9000.0,
    "altitude_bonus_mid_min_m": 1800.0,
    "altitude_bonus_high": 0.50,
    "altitude_bonus_mid": 0.31,
    "nose_down_altitude_m": 4000.0,
    "nose_down_pitch_deg": -10.0,
    "nose_down_penalty": -1.13,
    "crash_penalty": -280.0,
}

CANDIDATES = {"R08": R08, "C03": C03, "C10": C10}

SCENARIOS = {
    "level_5000": {
        "name": "level_5000",
        "ownship": [1000.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 280.0],
        "target": [6000.0, 0.0, -5000.0, 0.0, 0.0, 180.0, 260.0],
    },
    "light_disturbance_4500": {
        "name": "light_disturbance_4500",
        "ownship": [1000.0, 0.0, -4500.0, 8.0, -3.0, 8.0, 280.0],
        "target": [6000.0, 0.0, -5000.0, 0.0, 0.0, 180.0, 260.0],
    },
    "mild_dive_4000": {
        "name": "mild_dive_4000",
        "ownship": [1000.0, 0.0, -4000.0, 10.0, -8.0, 10.0, 280.0],
        "target": [6000.0, 0.0, -4800.0, 0.0, 0.0, 180.0, 260.0],
    },
    "low_dive_3500": {
        "name": "low_dive_3500",
        "ownship": [1000.0, 0.0, -3500.0, 12.0, -12.0, -10.0, 275.0],
        "target": [5500.0, 0.0, -4600.0, 0.0, 0.0, 180.0, 255.0],
    },
    "banked_descent_3800": {
        "name": "banked_descent_3800",
        "ownship": [1000.0, 0.0, -3800.0, 35.0, -8.0, 20.0, 275.0],
        "target": [5500.0, 500.0, -4600.0, 0.0, 0.0, 180.0, 255.0],
    },
}


def scenario_pool(name: str, jitter: bool) -> dict:
    scenario = copy.deepcopy(SCENARIOS[name])
    scenario.update({
        "weight": 1.0,
        "target_mode": "loiter",
        "target_loiter": {"enabled": True, "bank": 0.0, "pitch": 0.0},
    })
    pool = {"mode": "scenario_pool", "scenarios": [scenario]}
    if jitter:
        pool.update({
            "ownship_randomization": {
                "enabled": True,
                "radius": 40.0,
                "r_roll": 2.0,
                "r_pitch": 1.0,
                "r_heading": 3.0,
                "speed_mps": 3.0,
            },
            "target_randomization": {
                "enabled": True,
                "radius": 40.0,
                "r_heading": 2.0,
                "speed_mps": 2.0,
            },
        })
    return pool


def get_stages() -> list[CurriculumStage]:
    candidate = os.environ.get("ALTITUDE_PIPELINE_CANDIDATE", "R08").upper()
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown ALTITUDE_PIPELINE_CANDIDATE: {candidate}")
    reward = CANDIDATES[candidate]
    definitions = [
        (0, "level_hold", "level_5000", 100, 0.20, 35.0, False),
        (1, "light_recovery", "light_disturbance_4500", 120, 0.25, 55.0, True),
        (2, "mild_dive_recovery", "mild_dive_4000", 150, 0.30, 80.0, True),
        (3, "low_dive_recovery", "low_dive_3500", 180, 0.35, 110.0, True),
        (4, "banked_descent_recovery", "banked_descent_3800", 200, 0.40, 130.0, True),
    ]
    return [
        CurriculumStage(
            index=index,
            name=stage_name,
            description=f"Altitude safety pipeline: {scenario_name} using {candidate}.",
            target_mode="loiter",
            episode_step_limit=3600,
            max_iterations=max_iterations,
            checkpoint_interval=25,
            reward_overrides=dict(reward),
            randomization={"enabled": False},
            env_overrides={
                "initial_scenario": scenario_pool(scenario_name, jitter),
            },
            advance_conditions={
                "crash_rate_max": crash_max,
                "ep_altitude_penalty_steps_max": altitude_steps_max,
            },
            advance_window=10,
        )
        for (
            index,
            stage_name,
            scenario_name,
            max_iterations,
            crash_max,
            altitude_steps_max,
            jitter,
        ) in definitions
    ]


__all__ = ["CANDIDATES", "SCENARIOS", "scenario_pool", "get_stages"]
