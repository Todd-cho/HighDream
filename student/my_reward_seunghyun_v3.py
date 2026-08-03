# -*- coding: utf-8 -*-
"""[승현 개인 진단용 사본] my_reward.py의 range/far_range saturation 완화 테스트.

팀 공통 student/my_reward.py는 건드리지 않음. 이 파일은 그걸 그대로 복사한 뒤
range/far_range 계산식 두 군데만 고친 것 - 나머지 로직/파라미터는 전부 동일함.

바뀐 부분 (파라미터 값이 아니라 계산식 자체):
  1) range 컴포넌트: max(-1.0, range_score) -> max(-4.0, range_score)
     바닥을 -1.0에서 -4.0으로 낮춰서, ideal 범위 밖으로 더 멀리 나가도
     "더 나쁘다"는 신호가 좀 더 오래 살아있게 함.
  2) far_range 컴포넌트: 분모를 far_start 대신 far_start*6로 키움
     (예: far_start=1800이면 기존엔 distance=3600에서 이미 포화, 이제는
     distance=1800+10800=12600 근처까지 서서히 벌점이 커짐)
     -> 실제 관측되는 7000~13000m 구간에서도 "가까워지면 낫다"는
        gradient가 남아있도록 하는 게 목적.

far_range_penalty_span_m을 YAML env_config.reward에 추가하면 그 값을 쓰고,
없으면 far_start*6을 기본값으로 씀 (기존 실험 YAML 그대로 써도 동작함).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    # ---------------------------------------------------------------------
    # 1) Common pacing
    # ---------------------------------------------------------------------
    "step_penalty": -0.003,

    # ---------------------------------------------------------------------
    # 2) Range keeping
    # ---------------------------------------------------------------------
    "too_close_m": 460.0,
    "ideal_range_min_m": 450.0,
    "ideal_range_max_m": 1100.0,
    "range_scale": 0.8,
    "overshoot_penalty": 4.0,
    "overshoot_quadratic_scale": 1.5,
    "inside_min_range_penalty": -3.0,

    # ---------------------------------------------------------------------
    # 3) Aim / geometry
    # ---------------------------------------------------------------------
    "ata_scale": 0.12,
    "aa_scale": 0.03,

    # ---------------------------------------------------------------------
    # 4) Gun WEZ / damage
    # ---------------------------------------------------------------------
    "wez_bonus": 0.5,
    "damage_scale": 20.0,

    # ---------------------------------------------------------------------
    # 5) Altitude / flight safety
    # ---------------------------------------------------------------------
    "altitude_soft_floor_m": 2200.0,
    "altitude_hard_floor_m": 1000.0,
    "low_altitude_penalty": 1.0,
    "very_low_altitude_penalty": 2.2,
    "altitude_bonus_high_min_m": 1500.0,
    "altitude_bonus_high_max_m": 5000.0,
    "altitude_bonus_mid_min_m": 700.0,
    "altitude_bonus_high": 0.8,
    "altitude_bonus_mid": 0.8,

    "nose_down_altitude_m": 4500.0,
    "nose_down_pitch_deg": -5.0,
    "nose_down_penalty": -2.2,

    # ---------------------------------------------------------------------
    # 6) Control stability
    # ---------------------------------------------------------------------
    "roll_limit_deg": 80.0,
    "roll_limit_penalty": 0.15,
    "pitch_down_limit_deg": -12.0,
    "pitch_down_penalty": 1.2,
    "pitch_up_limit_deg": 35.0,
    "pitch_up_penalty": 0.25,

    # ---------------------------------------------------------------------
    # 7) Attack opportunity
    # ---------------------------------------------------------------------
    "attack_range_min_m": 250.0,
    "attack_range_max_m": 1500.0,
    "attack_range_bonus": 0.2,
    "far_range_penalty_start_m": 3500.0,
    "far_range_penalty": 0.25,

    # ---------------------------------------------------------------------
    # 8) Terminal outcome
    # ---------------------------------------------------------------------
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "crash_penalty": -200.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
    """Return reward components for early fixed-target gun training."""
    cfg = {**MY_REWARD_CONFIG, **reward_config}
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    aa = abs(float(geo_info._get_aspect_angle(ownship_state, target_state, False)))
    altitude = float(ownship_state[StateIndex.ALT])
    roll = float(ownship_state[StateIndex.ROLL])
    pitch = float(ownship_state[StateIndex.PITCH])

    components: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1) Common pacing
    # ------------------------------------------------------------------
    components["step"] = float(cfg["step_penalty"])

    # ------------------------------------------------------------------
    # 2) Range keeping [수정: 바닥을 -1.0 -> -4.0]
    # ------------------------------------------------------------------
    ideal_min = float(cfg["ideal_range_min_m"])
    ideal_max = float(cfg["ideal_range_max_m"])
    ideal_mid = 0.5 * (ideal_min + ideal_max)
    ideal_half_width = max(1.0, 0.5 * (ideal_max - ideal_min))
    range_score = 1.0 - abs(distance - ideal_mid) / ideal_half_width
    components["range"] = float(cfg["range_scale"]) * max(-4.0, range_score)

    if distance < float(cfg["too_close_m"]):
        too_close_ratio = 1.0 - distance / max(1.0, float(cfg["too_close_m"]))
        close_penalty = _clamp01(too_close_ratio)
        components["overshoot"] = -float(cfg["overshoot_penalty"]) * (
            close_penalty
            + float(cfg["overshoot_quadratic_scale"]) * close_penalty * close_penalty
        )
    else:
        components["overshoot"] = 0.0

    if distance < float(wez_config["min_range_m"]):
        components["inside_min_range"] = float(cfg["inside_min_range_penalty"])
    else:
        components["inside_min_range"] = 0.0

    # ------------------------------------------------------------------
    # 3) Aim / geometry
    # ------------------------------------------------------------------
    ata_score = 1.0 - ata / 90.0
    components["ata"] = float(cfg["ata_scale"]) * max(-1.0, ata_score)

    aa_score = 1.0 - aa / 180.0
    components["aa"] = float(cfg["aa_scale"]) * max(-1.0, aa_score)

    # ------------------------------------------------------------------
    # 4) Gun WEZ / damage
    # ------------------------------------------------------------------
    in_wez = (
        float(wez_config["min_range_m"]) <= distance <= float(wez_config["max_range_m"])
        and ata <= float(wez_config["angle_deg"]) / 2.0
    )
    components["wez"] = float(cfg["wez_bonus"]) if in_wez else 0.0

    components["damage"] = float(cfg["damage_scale"]) * (
        float(target_damage) - float(ownship_damage)
    )

    # ------------------------------------------------------------------
    # 5) Altitude / flight safety
    # ------------------------------------------------------------------
    altitude_bonus = 0.0
    if float(cfg["altitude_bonus_high_min_m"]) <= altitude <= float(cfg["altitude_bonus_high_max_m"]):
        altitude_bonus = float(cfg["altitude_bonus_high"])
    elif float(cfg["altitude_bonus_mid_min_m"]) <= altitude < float(cfg["altitude_bonus_high_min_m"]):
        altitude_bonus = float(cfg["altitude_bonus_mid"])
    components["altitude"] = altitude_bonus

    if altitude < float(cfg["nose_down_altitude_m"]) and pitch < float(cfg["nose_down_pitch_deg"]):
        components["nose_down"] = float(cfg["nose_down_penalty"])
    else:
        components["nose_down"] = 0.0

    safety = 0.0
    if altitude < float(cfg["altitude_soft_floor_m"]):
        safety -= float(cfg["low_altitude_penalty"]) * _clamp01(
            (float(cfg["altitude_soft_floor_m"]) - altitude)
            / max(1.0, float(cfg["altitude_soft_floor_m"]) - float(cfg["altitude_hard_floor_m"]))
        )
    if altitude < float(cfg["altitude_hard_floor_m"]):
        safety -= float(cfg["very_low_altitude_penalty"])
    components["safety"] = safety

    # ------------------------------------------------------------------
    # 6) Control stability
    # ------------------------------------------------------------------
    control_stability = 0.0
    abs_roll = abs(roll)
    if abs_roll > float(cfg["roll_limit_deg"]):
        control_stability -= float(cfg["roll_limit_penalty"]) * _clamp01(
            (abs_roll - float(cfg["roll_limit_deg"]))
            / max(1.0, 180.0 - float(cfg["roll_limit_deg"]))
        )
    if pitch < float(cfg["pitch_down_limit_deg"]):
        control_stability -= float(cfg["pitch_down_penalty"]) * _clamp01(
            (float(cfg["pitch_down_limit_deg"]) - pitch)
            / max(1.0, 90.0 + float(cfg["pitch_down_limit_deg"]))
        )
    if pitch > float(cfg["pitch_up_limit_deg"]):
        control_stability -= float(cfg["pitch_up_penalty"]) * _clamp01(
            (pitch - float(cfg["pitch_up_limit_deg"]))
            / max(1.0, 90.0 - float(cfg["pitch_up_limit_deg"]))
        )
    components["control_stability"] = control_stability

    # ------------------------------------------------------------------
    # 7) Attack opportunity
    # ------------------------------------------------------------------
    attack_min = float(cfg["attack_range_min_m"])
    attack_max = float(cfg["attack_range_max_m"])
    if attack_min <= distance <= attack_max:
        components["attack_range"] = float(cfg["attack_range_bonus"])
    else:
        components["attack_range"] = 0.0

    # [수정] far_range: 포화 지점을 far_start*6까지 늘림 (기존은 far_start*2에서 포화)
    far_start = float(cfg["far_range_penalty_start_m"])
    far_span = float(cfg.get("far_range_penalty_span_m", far_start * 6.0))
    if distance > far_start:
        components["far_range"] = -float(cfg["far_range_penalty"]) * _clamp01(
            (distance - far_start) / max(1.0, far_span)
        )
    else:
        components["far_range"] = 0.0

    # ------------------------------------------------------------------
    # 8) Terminal outcome
    # ------------------------------------------------------------------
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
        else:
            terminal_reward = float(cfg["draw_reward"])
    components["terminal"] = terminal_reward

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
