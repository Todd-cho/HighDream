"""Constant-cost 3-D VPP pursuit guidance from the KAIST UCAV study.

The paper blends lag/pure/lead virtual pursuit points and combines velocity
pursuit guidance with an N=1 proportional-navigation LOS-rate compensation.
This module keeps that structure explicit and independent of the aircraft
attitude controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VPPGuidanceResult:
    point: np.ndarray
    blend: float
    mode: str
    turn_rate_degps: float
    gamma_deg: float
    closure_mps: float
    los_rate_degps: float
    t_cpa_s: float
    d_cpa_m: float
    turn_circle_active: bool
    projectile_tof_s: float
    apg_weight: float


def compute_vpp_guidance(
    own_position: np.ndarray,
    target_position: np.ndarray,
    own_velocity: np.ndarray,
    target_velocity: np.ndarray,
    *,
    ata_deg: float,
    threat_ata_deg: float,
    distance_m: float,
    previous_blend: float,
    dt: float,
    transition_rate_per_s: float = 0.8,
    lag_distance_m: float = 800.0,
    mutual_lateral_m: float = 500.0,
    projectile_speed_mps: float = 650.0,
    pursuit_gain: float = 0.5,
    navigation_constant: float = 1.0,
    turn_rate_limit_degps: float = 16.0,
    gamma_limit_deg: float = 24.0,
    target_yaw_rate_degps: float = 0.0,
    turn_circle_enabled: bool = False,
    turn_circle_horizon_s: float = 1.0,
    mode_override: str | None = None,
    target_acceleration: np.ndarray | None = None,
    own_forward: np.ndarray | None = None,
    ballistic_tof_enabled: bool = False,
    apg_enabled: bool = False,
    apg_gain: float = 0.7,
) -> VPPGuidanceResult:
    own_position = np.asarray(own_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    own_velocity = np.asarray(own_velocity, dtype=np.float64)
    target_velocity = np.asarray(target_velocity, dtype=np.float64)
    relative = target_position - own_position
    distance = max(float(np.linalg.norm(relative)), 1.0)
    los = relative / distance
    relative_velocity = target_velocity - own_velocity
    closure = -float(np.dot(los, relative_velocity))
    rel_speed_sq = float(np.dot(relative_velocity, relative_velocity))
    t_cpa = float(np.clip(
        -float(np.dot(relative, relative_velocity)) / max(rel_speed_sq, 1.0), 0.0, 8.0
    ))
    d_cpa = float(np.linalg.norm(relative + relative_velocity * t_cpa))

    target_speed = float(np.linalg.norm(target_velocity))
    if target_speed > 1.0:
        target_track = target_velocity / target_speed
    else:
        target_track = los

    # Paper lead VPP: predict the target over an approximate projectile TOF.
    if ballistic_tof_enabled:
        # rho(t) = -170.3 t^2 + 1021 t from the gun-lead study.
        disc = max(1021.0 * 1021.0 - 4.0 * 170.3 * distance, 0.0)
        tof = (1021.0 - math.sqrt(disc)) / (2.0 * 170.3)
    else:
        tof = distance / max(projectile_speed_mps, 1.0)
    tof = float(np.clip(tof, 0.15, 2.0))
    target_accel = (
        np.zeros(3, dtype=np.float64)
        if target_acceleration is None
        else np.asarray(target_acceleration, dtype=np.float64)
    )
    lead = target_position + target_velocity * tof + 0.5 * target_accel * tof * tof
    pure = target_position.copy()
    lag = target_position - target_track * lag_distance_m

    # Predict the target along its measured turn circle instead of assuming a
    # straight line. This is constant-cost and directly targets the recurring
    # level-turn tracking error seen in live fights.
    turn_circle_active = False
    yaw_rate_radps = math.radians(target_yaw_rate_degps)
    horizontal_speed = float(np.linalg.norm(target_velocity[:2]))
    if turn_circle_enabled and horizontal_speed > 30.0 and abs(yaw_rate_radps) >= math.radians(1.5):
        track_xy = target_velocity[:2] / horizontal_speed
        turn_sign = 1.0 if yaw_rate_radps >= 0.0 else -1.0
        left_normal = np.array([-track_xy[1], track_xy[0]])
        radius = float(np.clip(horizontal_speed / abs(yaw_rate_radps), 250.0, 6000.0))
        centre = target_position[:2] + turn_sign * left_normal * radius
        radial = target_position[:2] - centre
        horizon = float(np.clip(turn_circle_horizon_s, 0.2, 2.0))
        angle = yaw_rate_radps * horizon
        c, s = math.cos(angle), math.sin(angle)
        future_xy = centre + np.array([
            c * radial[0] - s * radial[1], s * radial[0] + c * radial[1]
        ])
        pure[:2] = future_xy
        future_track = np.array([
            c * target_track[0] - s * target_track[1],
            s * target_track[0] + c * target_track[1],
            target_track[2],
        ])
        lag = pure - future_track * lag_distance_m
        if not ballistic_tof_enabled:
            lead[:2] = (
                future_xy
                + future_track[:2] * horizontal_speed * min(tof, 0.8)
            )
        # With ballistic TOF enabled, retain the acceleration-predicted lead
        # computed above. Otherwise the turn-circle block makes W104's
        # projectile/acceleration equations dead code.
        turn_circle_active = True

    # A rear offset alone preserves a symmetric nose-on pass. Add an outside
    # horizontal displacement when the opponent can point back at us.
    if threat_ata_deg < 30.0:
        horizontal_track = target_track[:2]
        norm = float(np.linalg.norm(horizontal_track))
        if norm > 1e-6:
            horizontal_track /= norm
            side = np.array([-horizontal_track[1], horizontal_track[0]])
            cross_z = float(np.cross(
                np.r_[horizontal_track, 0.0], np.r_[los[:2], 0.0]
            )[2])
            side_sign = 1.0 if cross_z >= 0.0 else -1.0
            scale = 1.0 - threat_ata_deg / 30.0
            lag[:2] += side_sign * side * mutual_lateral_m * scale

    # Paper pursuit probabilities reduced to a continuous target coordinate:
    # 0=lag, 0.5=pure, 1=lead. Never jump directly between lead and lag.
    imminent_merge = closure > 80.0 and t_cpa < 3.0 and d_cpa < 1800.0
    safe_lead = (
        distance_m <= 1800.0 and ata_deg <= 20.0
        and threat_ata_deg >= 35.0 and not imminent_merge
    )
    if threat_ata_deg < 30.0 or imminent_merge or closure > 260.0:
        desired_blend, mode = 0.0, "lag"
    elif safe_lead:
        desired_blend, mode = 1.0, "lead"
    elif ata_deg >= 90.0:
        desired_blend, mode = 0.5, "pure"
    else:
        desired_blend, mode = 0.5, "pure"
    if mode_override == "lag":
        desired_blend, mode = 0.0, "lag"
    elif mode_override == "pure":
        desired_blend, mode = 0.5, "pure"
    elif mode_override == "lead":
        desired_blend, mode = 1.0, "lead"
    max_step = max(0.0, transition_rate_per_s * max(dt, 0.0))
    blend = float(previous_blend + np.clip(
        desired_blend - previous_blend, -max_step, max_step
    ))
    if blend <= 0.5:
        point = lag * (1.0 - 2.0 * blend) + pure * (2.0 * blend)
    else:
        point = pure * (2.0 - 2.0 * blend) + lead * (2.0 * blend - 1.0)

    vpp_relative = point - own_position
    vpp_range = max(float(np.linalg.norm(vpp_relative)), 1.0)
    desired_direction = vpp_relative / vpp_range
    own_speed = max(float(np.linalg.norm(own_velocity)), 1.0)
    own_direction = own_velocity / own_speed
    lateral_error = desired_direction - np.dot(desired_direction, own_direction) * own_direction
    pursuit_accel = pursuit_gain * own_speed * lateral_error

    apg_weight = 0.0
    if apg_enabled and own_forward is not None:
        nose = np.asarray(own_forward, dtype=np.float64)
        nose /= max(float(np.linalg.norm(nose)), 1e-6)
        nose_error = desired_direction - np.dot(desired_direction, nose) * nose
        # Blend velocity-pursuit into gun-axis pursuit only near lead.
        apg_weight = float(np.clip(2.0 * (blend - 0.5), 0.0, 1.0))
        apg_accel = apg_gain * own_speed * nose_error
        pursuit_accel = (1.0 - apg_weight) * pursuit_accel + apg_weight * apg_accel

    los_omega = np.cross(relative, relative_velocity) / (distance * distance)
    png_accel = navigation_constant * max(closure, 0.0) * np.cross(
        los_omega, own_direction
    )
    accel = pursuit_accel + png_accel
    horizontal_speed = max(float(np.linalg.norm(own_velocity[:2])), 1.0)
    horizontal_direction = own_velocity[:2] / horizontal_speed
    left_normal = np.array([-horizontal_direction[1], horizontal_direction[0]])
    turn_rate = math.degrees(float(np.dot(accel[:2], left_normal)) / horizontal_speed)
    turn_rate = float(np.clip(turn_rate, -turn_rate_limit_degps, turn_rate_limit_degps))
    horizontal_vpp_range = max(float(np.linalg.norm(vpp_relative[:2])), 1.0)
    gamma = math.degrees(math.atan2(-vpp_relative[2], horizontal_vpp_range))
    gamma = float(np.clip(gamma, -gamma_limit_deg, gamma_limit_deg))
    return VPPGuidanceResult(
        point=point,
        blend=blend,
        mode=mode,
        turn_rate_degps=turn_rate,
        gamma_deg=gamma,
        closure_mps=closure,
        los_rate_degps=math.degrees(float(np.linalg.norm(los_omega))),
        t_cpa_s=t_cpa,
        d_cpa_m=d_cpa,
        turn_circle_active=turn_circle_active,
        projectile_tof_s=tof,
        apg_weight=apg_weight,
    )
