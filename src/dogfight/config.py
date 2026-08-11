from __future__ import annotations

import copy

FEET_TO_METER = 0.30480
METER_TO_FEET = 3.28084
KNOT_TO_METER_SEC = 0.51444

DEFAULT_ENV_CONFIG = {
    "sim_hz": 60,
    "step_ratio": None,
    "delta": None,
    "time_step": None,
    "max_engage_time": 300.0,
    "episode_step_limit": 18000,
    "min_altitude": 300.0,
    # Hard, non-learned recovery override for the ownship's own RL action
    # (single_agent_env.py._apply_safety_override): when altitude drops
    # below safety_override_altitude_m AND pitch is at/below
    # safety_override_pitch_deg (i.e. actually diving, not just briefly
    # low), the policy's pitch/roll/throttle commands are replaced with a
    # forced wings-level pull-up + full throttle, regardless of what the
    # trained policy outputs. Disabled by default (opt-in via env_config)
    # so it never changes behavior for existing trained checkpoints unless
    # explicitly turned on.
    "safety_override_enabled": False,
    "safety_override_altitude_m": 500.0,
    "safety_override_pitch_deg": -5.0,
    "safety_override_roll_deg": 45.0,
    # Predictive (time-to-impact) trigger, added after a fixed-altitude-only
    # trigger (500-1500m) still let some long, slowly-accelerating dives
    # crash: it engages far too late relative to how much energy/momentum
    # the dive has already built up. This estimates vertical rate from
    # consecutive altitude samples and triggers as soon as, at the current
    # rate, the ownship would reach safety_override_altitude_m within
    # safety_override_time_horizon_s -- regardless of current altitude, so a
    # fast dive from high altitude triggers early. safety_override_hard_floor_m
    # is an unconditional backstop independent of the rate estimate.
    "safety_override_time_horizon_s": 15.0,
    "safety_override_hard_floor_m": 400.0,
    "observation_mode": "classic12",
    "ownship_control_mode": "rl",
    "target_mode": "behavior_tree",
    "ownship_behavior_dll": None,
    "target_behavior_dll": "AIP_BASE_target.dll",
    "target_loiter": {"enabled": True, "bank": 30.0, "pitch": 0.0},
    "target_autopilot": {"heading_cmd": 180.0, "altitude_cmd": 7000.0, "speed_cmd": 250.0},
    # target_mode="pursuit_autopilot" (added 2026-08-11): scripted, non-learned
    # active opponent -- recomputes heading_cmd toward the ownship's live
    # position every step and drives the sim's own step_autopilot() loop with
    # it, instead of the (currently empty/no-op) native behavior_tree engine.
    "target_pursuit_autopilot": {"speed_cmd": 260.0},
    "reward": {
        "step_penalty": -0.01,
        "damage_scale": 20.0,
        "pursuit_scale": 0.3,
        "pursuit_half_angle_deg": 30.0,
        "pursuit_range_m": 3000.0,
        "low_altitude_penalty": 0.1,
        "win_reward": 100.0,
        "loss_reward": -100.0,
        "draw_reward": -30.0,
        "guard_fail_penalty": -50.0,
    },
    "wez": {
        # Phase 1 fields kept at top level for backward compatibility with
        # code that only checks a single binary WEZ band (observation.py's
        # in-WEZ feature, the wez_bonus term in student/my_reward*.py).
        "angle_deg": 2.0,
        "min_range_m": 500 * FEET_TO_METER,
        "max_range_m": 3000 * FEET_TO_METER,
        # Full 3-phase damage cone per the 2026 AI Pilot Top Gun Challenge
        # rules (slides 13/14): LOS<1deg/500-3000ft x1.0, LOS<2deg/500-3500ft
        # x0.3, LOS<3deg/500-4000ft x0.1. min_range_m is shared across phases.
        # Ordered tightest-first: single_agent_env.py.update_damage() takes
        # the first (narrowest) phase an aircraft qualifies for, matching the
        # rule "적이 하위 Phase 범위에 있으면 하위 phase 대미지 적용".
        "phases": [
            {"angle_deg": 2.0, "max_range_m": 3000 * FEET_TO_METER, "damage_coeff": 1.0},
            {"angle_deg": 4.0, "max_range_m": 3500 * FEET_TO_METER, "damage_coeff": 0.3},
            {"angle_deg": 6.0, "max_range_m": 4000 * FEET_TO_METER, "damage_coeff": 0.1},
        ],
    },
    "ownship": [1000.0, 0.0, -7000.0, 0.0, 0.0, 0.0, 300.0],
    "target": [6000.0, 0.0, -7000.0, 0.0, 0.0, 180.0, 300.0],
    "artifacts_dir": "artifacts/logs",
    # Per-episode position randomization (used by curriculum; disabled by default)
    "ownship_randomization": {
        "enabled": False,
        "radius": 0.0,      # NED position scatter radius (meters)
        "r_roll": 0.0,      # roll scatter (degrees)
        "r_pitch": 0.0,     # pitch scatter (degrees)
        "r_heading": 0.0,   # heading scatter (degrees)
    },
    "initial_scenario": {
        "mode": "default",
        # User-defined weighted scenario pool. Each aircraft vector is
        # [N, E, D, roll, pitch, heading, speed] in m, deg, and m/s.
        "scenarios": [],
        "ownship_randomization": {
            "enabled": False,
            "radius": 0.0,
            "r_roll": 0.0,
            "r_pitch": 0.0,
            "r_heading": 0.0,
            "speed_mps": 0.0,
        },
        "target_randomization": {
            "enabled": False,
            "radius": 0.0,
            "r_roll": 0.0,
            "r_pitch": 0.0,
            "r_heading": 0.0,
            "speed_mps": 0.0,
        },
        "shared_randomization": {
            "enabled": False,
            "n_m": 0.0,
            "e_m": 0.0,
            "d_m": 0.0,
            "speed_mps": 0.0,
        },
        "legacy_use_random_scenario": True,
        "legacy_use_first_scenario_only": False,
        "legacy_scenario_indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "legacy_randomization": {
            "aircraft_radius_m": 100.0,
            "roll_deg": 5.0,
            "pitch_deg": 5.0,
            "heading_deg": 5.0,
            "shared_n_m": 4000.0,
            "shared_e_m": 4000.0,
            "shared_d_m": 4000.0,
            "target_distance_n_m": 300.0,
            "speed_mps": 50.0,
            "loiter_bank_deg_range": [40.0, 70.0],
        },
        "alpha_deg": 0.0,
        "turn_diameter_ft": 6000.0,
        "separation_jitter_ft": [3000.0, 6000.0],
        "center_n_m": 3500.0,
        "center_e_m": 0.0,
        "altitude_m": 7000.0,
        "speed_mps_range": [250.0, 300.0],
        "vertical_pitch_choices_deg": [0.0, 10.0, -10.0],
        "roll_range_deg": [0.0, 180.0],
        "side_choices": [-1.0, 1.0],
    },
    "geometry_guard": {
        "enabled": False,
        "mode": "two_circle_headon",
        "alpha_deg": 0.0,
        "early_alpha_max_deg": 80.0,
        "mid_alpha_max_deg": 140.0,
        "crossed_ata_limit_deg": 90.0,
        "turn_margin_deg": 10.0,
    },
}


def merge_env_config(env_config: dict | None) -> dict:
    merged = copy.deepcopy(DEFAULT_ENV_CONFIG)
    if not env_config:
        return merged

    _deep_update(merged, env_config)
    return merged


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge env config dictionaries in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
