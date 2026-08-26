"""Goal-conditioned tactical observation for hierarchical 1v1 BFM."""
from __future__ import annotations

import math
import numpy as np

from dogfight.envs.observation import build_observation as build_builtin_observation, normalize
from dogfight.sim.state_schema import StateIndex

OBSERVATION_MODE = "bfm_goal27"
OBSERVATION_SIZE = 27
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0

MANEUVERS = {
    "energy_recover": -1.0,
    "defensive_break": -0.66,
    "high_yoyo": -0.33,
    "pitchback": 0.0,
    "lead_pursuit": 0.33,
    "terminal_track": 0.66,
    "wez_hold": 1.0,
}


def _signed(value: float, fallback: float = 1.0) -> float:
    return 1.0 if value > 1e-6 else -1.0 if value < -1e-6 else fallback


def select_bfm_goal(ownship_state, target_state, geo_info) -> dict:
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    threat_ata = abs(float(geo_info._get_antenna_train_angle(target_state, ownship_state, False)))
    los_az, los_el = geo_info._get_los_angle(ownship_state, target_state)
    own_speed = float(ownship_state[StateIndex.KCAS])
    target_speed = float(target_state[StateIndex.KCAS])
    turn_sign = _signed(float(los_az))

    if own_speed < 185.0:
        maneuver, turn_rate, gamma, speed = "energy_recover", 6.0 * turn_sign, -5.0, 235.0
    elif threat_ata < 25.0 and distance < 2000.0:
        maneuver, turn_rate, gamma, speed = "defensive_break", 18.0 * turn_sign, -3.0, 245.0
    elif distance < 650.0 and own_speed > target_speed + 20.0 and ata > 8.0:
        maneuver, turn_rate, gamma, speed = "high_yoyo", 14.0 * turn_sign, 15.0, target_speed
    elif ata <= 2.0 and 150.0 <= distance <= 1100.0:
        maneuver, turn_rate, gamma, speed = "wez_hold", 6.0 * turn_sign, float(np.clip(los_el, -5.0, 5.0)), target_speed
    elif ata < 15.0 and distance < 1600.0:
        maneuver, turn_rate, gamma, speed = "terminal_track", 10.0 * turn_sign, float(np.clip(los_el, -8.0, 8.0)), target_speed + 5.0
    elif ata > 70.0 and threat_ata > 55.0:
        maneuver, turn_rate, gamma, speed = "pitchback", 18.0 * turn_sign, 3.0, 225.0
    else:
        maneuver, turn_rate, gamma, speed = "lead_pursuit", 14.0 * turn_sign, float(np.clip(los_el, -12.0, 12.0)), 245.0

    return {
        "maneuver": maneuver,
        "target_turn_rate_degps": turn_rate,
        "target_gamma_deg": gamma,
        "target_speed_mps": speed,
        "threat_ata_deg": threat_ata,
    }


def build_observation(ownship_state, target_state, geo_info, wez_config=None):
    base = build_builtin_observation("tactical19", ownship_state, target_state, geo_info, wez_config)
    goal = select_bfm_goal(ownship_state, target_state, geo_info)
    own_speed = float(ownship_state[StateIndex.KCAS])
    target_speed = float(target_state[StateIndex.KCAS])
    own_alt = float(ownship_state[StateIndex.ALT])
    target_alt = float(target_state[StateIndex.ALT])
    g = 9.80665
    own_energy = 0.5 * own_speed * own_speed + g * own_alt
    target_energy = 0.5 * target_speed * target_speed + g * target_alt

    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    obs[:19] = base
    obs[19] = normalize(own_energy, 0.0, 180000.0)
    obs[20] = normalize(own_energy - target_energy, -80000.0, 80000.0)
    obs[21] = normalize(target_speed, 0.0, 600.0)
    obs[22] = normalize(goal["threat_ata_deg"], 0.0, 180.0)
    obs[23] = normalize(goal["target_turn_rate_degps"], -25.0, 25.0)
    obs[24] = normalize(goal["target_gamma_deg"], -25.0, 25.0)
    obs[25] = normalize(goal["target_speed_mps"], 120.0, 360.0)
    obs[26] = MANEUVERS[goal["maneuver"]]
    return np.clip(obs, -1.0, 1.0)


def describe_observation():
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [f"tactical19_{i}" for i in range(19)] + [
            "specific_energy_norm", "energy_advantage_norm", "target_speed_norm",
            "threat_ata_norm", "manager_turn_rate_goal", "manager_gamma_goal",
            "manager_speed_goal", "manager_maneuver_id",
        ],
        "description": "tactical19 plus explicit hierarchical BFM/energy goals; manager never writes actuator commands.",
    }
