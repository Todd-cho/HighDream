# -*- coding: utf-8 -*-
"""Copy of student/my_reward_delta_v1_turnback.py plus a multiplicative
range-x-angle "WEZ proximity" shaping term.

Motivation (2026-08-09, Stage6f follow-up analysis): stage6f_turnback_200iter
clears 3/4 Stage6 safe_wez_geometry gates (crash 0%, altitude 2624m, distance
6245m) but wez_episode_rate is 0.0 across all Stage6f variants (control and
turnback alike) -- not a single evaluation episode ever hit the exact WEZ
condition (min_range_m <= distance <= max_range_m AND ata <= angle_deg/2)
even once. Pairing each episode's final_distance_m with final_ata_deg shows
why: episodes with excellent ata (as low as 1.8deg) are consistently still
far away (4000-16000m), while the only close approaches (min_distance_m as
low as 88m) happen with bad ata that only gets corrected afterward, by which
point distance has re-opened. The existing range and ata reward components
are additive (range_score + ata_score), so the policy can bank reward from
either dimension independently without ever needing both good range AND good
angle at the same instant -- there is no gradient pushing toward the joint
condition. The only signal that directly rewards the joint condition is
wez_bonus, which is a sparse all-or-nothing bonus that never fires here, so
it provides no gradient before the exact condition is first reached.

This module adds a dense, multiplicative proxy for that joint condition:
    range_closeness = clamp01(1 - |distance - wez_mid| / wez_half_width)
    ata_alignment    = clamp01(1 - ata / (wez_angle_deg / 2))
    wez_proximity_reward = wez_proximity_scale * range_closeness * ata_alignment
Multiplying (rather than adding, like range/ata already are) means a step
only earns a large proximity reward when BOTH factors are simultaneously
decent -- being perfectly aligned but far away, or very close but pointed
the wrong way, both collapse the product toward zero. wez_mid/wez_half_width
are derived from the actual wez_config passed into compute_reward (not a
hardcoded band), so this tracks the real WEZ geometry for the run.

Gated by the same safe_altitude check as range_delta/ata_delta/ata_recovery
(see my_reward_delta_v1.py's 2026-08-06 fix note) so it cannot become a new
dive-and-crash incentive: closing to WEZ range only pays while altitude is
still in the safe band.

wez_proximity_scale defaults to 0.0 (opt-in via YAML), so this module is a
strict superset of my_reward_delta_v1_turnback.py until explicitly set.
Everything else is identical to that module.

2026-08-10: optional wez_proximity_use_aa (default False) folds aa (aspect
angle) into the joint term as a third multiplicative factor -- range x ata x
aa instead of just range x ata -- using the same aa/180 normalization the
static "aa" reward component already uses.

2026-08-10: optional wez_hold shaping (wez_hold_scale / wez_hold_break_penalty_scale,
both 0.0 by default). Motivation: a 20-episode frozen (deterministic) evaluation of
a checkpoint trained with this module found the policy reliably closes to
350-800m (well inside attack_range_min/max_m) every episode, but then drifts
back out to 15000-46000m by episode end instead of holding position long
enough to also align ata -- range and ata alignment keep being achieved
sequentially, never simultaneously, so wez_episode_rate stayed 0% despite
consistently getting close. A likely cause: range_delta/ata_delta only pay
for the *motion* of closing, not for the *state* of being close -- once
distance stops decreasing, delta reward drops to ~0, but retreating and
re-approaching earns a fresh burst of positive delta reward again. That can
make repeated bail-and-reapproach cycles net *more* cumulative reward than
settling into one hold, which would explain the pattern directly.

wez_hold adds a duration-aware incentive on top of the existing (state-only,
not duration-aware) wez_proximity term:
  - reuses the same range_closeness x ata_alignment [x aa_alignment] "quality"
    score in [0, 1] already computed for wez_proximity;
  - tracks _hold_streak, the number of consecutive steps quality has stayed
    at/above wez_hold_quality_threshold;
  - while in that streak, pays wez_hold_scale * quality * (1 + wez_hold_growth_per_step
    * min(streak, wez_hold_growth_cap_steps)) each step -- reward per step
    for holding position *grows* with how long the hold has lasted, capped so
    a very long hold doesn't dominate other reward terms unboundedly;
  - if a streak of at least wez_hold_break_min_streak steps is broken (quality
    drops back below threshold), pays a one-time
    -wez_hold_break_penalty_scale * min(broken_streak, wez_hold_growth_cap_steps)
    penalty -- makes abandoning an established hold (e.g. to "reset" for
    another approach pass) cost something, directly countering the
    bail-and-reapproach incentive diagnosed above.
Gated by the same safe_altitude check as the other delta/proximity terms (streak
resets to 0 whenever altitude drops out of the safe band, same as everywhere
else in this module) so it cannot become a new dive-and-crash incentive.
Both scales default to 0.0 (opt-in via YAML), so this module remains a strict
superset of its prior behavior until explicitly set.

2026-08-11: optional pursuit_heading shaping (pursuit_heading_scale, 0.0 by
default). Motivation: raising max_engage_time from 120s to the real
competition value (200s, see slide 11 of the rules deck) on top of a
wez_hold-trained checkpoint made mean_distance/final_distance *worse*, not
better (mean_distance 6951m -> 12556m, n=3) -- ruling out "not enough time to
turn back" as the primary cause. range_delta already rewards the *outcome* of
closing distance step-over-step, but gives no *directional* signal for how to
get there -- from far away, after a bad merge, the policy has to stumble onto
a heading that happens to reduce range through undirected exploration before
range_delta ever pays out again. That is a much harder credit-assignment
problem than staying oriented once already close, which likely explains why
far-tail distances kept growing even though a return-and-close maneuver was,
in principle, already rewarded once achieved.

This term rewards the *direction* of travel directly, independent of whether
distance happens to be decreasing yet:
    motion_dir = normalize(current_position - previous_position)   # NED, this env-step's actual displacement
    los_dir    = normalize(target_position - current_position)     # NED, current line-of-sight to target
    pursuit_heading_reward = pursuit_heading_scale * dot(motion_dir, los_dir)
dot(motion_dir, los_dir) is in [-1, 1]: +1 means this step's actual flight
path point directly at the target's current position (pure pursuit, the
concept from the rules deck's Behavior Tree tutorial, slides 36-40), -1 means
flying directly away, 0 means perpendicular/orbiting. Unlike ata (which
measures nose/boresight alignment, i.e. aim), this measures where the
aircraft's flight path is actually taking it, and is recomputed every step
against the target's live position, so it automatically re-targets as the
target maneuvers. It rewards "turning back toward the target" immediately and
continuously, rather than only after distance has already started shrinking.
Gated by the same safe_altitude check as the rest of the module. Positions
are cached module-level (_prev_position) the same way _prev_distance/_prev_ata
already are, reset on episode boundaries. Bounded per-step magnitude
(scale * [-1, 1]) means even a full-episode maximum (perfectly aimed at the
target the whole time) stays well under crash_penalty in total, so it cannot
by itself create a dive-to-target incentive the way unclipped delta rewards
once did (see the 2026-08-06 fix note above).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex, position_ned


MY_REWARD_CONFIG = {
    "step_penalty": -0.003,
    "survival_bonus": 0.0,
    "too_close_m": 460.0,
    "ideal_range_min_m": 450.0,
    "ideal_range_max_m": 1100.0,
    "range_scale": 0.8,
    "overshoot_penalty": 4.0,
    "overshoot_quadratic_scale": 1.5,
    "inside_min_range_penalty": -3.0,
    "ata_scale": 0.12,
    "aa_scale": 0.03,
    "wez_bonus": 0.5,
    "damage_scale": 20.0,
    "altitude_soft_floor_m": 3800.0,
    "altitude_hard_floor_m": 1800.0,
    "low_altitude_penalty": 1.76,
    "very_low_altitude_penalty": 5.49,
    "altitude_bonus_high_min_m": 2100.0,
    "altitude_bonus_high_max_m": 9000.0,
    "altitude_bonus_mid_min_m": 1500.0,
    "altitude_bonus_high": 0.72,
    "altitude_bonus_mid": 0.43,
    "nose_down_altitude_m": 3200.0,
    "nose_down_pitch_deg": -6.0,
    "nose_down_penalty": -0.9,
    "roll_limit_deg": 80.0,
    "roll_limit_penalty": 0.15,
    "pitch_down_limit_deg": -12.0,
    "pitch_down_penalty": 1.2,
    "pitch_up_limit_deg": 35.0,
    "pitch_up_penalty": 0.25,
    "attack_range_min_m": 250.0,
    "attack_range_max_m": 1500.0,
    "attack_range_bonus": 0.2,
    "far_range_penalty_start_m": 3500.0,
    "far_range_penalty": 0.25,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "crash_penalty": -260.0,
    "range_delta_scale": 0.0,
    "ata_delta_scale": 0.0,
    "range_delta_clip_m": 60.0,
    "ata_delta_clip_deg": 10.0,
    "ata_recovery_scale": 0.0,
    "ata_recovery_threshold_deg": 90.0,
    # New: dense range-x-angle WEZ proximity shaping. 0.0 by default (opt-in
    # via YAML) so this module is a strict superset of
    # my_reward_delta_v1_turnback.py until explicitly set.
    "wez_proximity_scale": 0.0,
    # 2026-08-10: optionally fold aa (aspect angle, i.e. is the target's tail
    # facing us) into the joint proximity term as a third multiplicative
    # factor, using the same aa/180 normalization as the existing static "aa"
    # reward component (see components["aa"] above). False by default so the
    # module keeps its original range x ata behavior unless explicitly opted
    # in -- this only changes anything when wez_proximity_scale != 0.0.
    "wez_proximity_use_aa": False,
    # 2026-08-10: duration-aware hold/loiter shaping -- see module docstring.
    # Both scales 0.0 by default (opt-in via YAML).
    "wez_hold_scale": 0.0,
    "wez_hold_quality_threshold": 0.35,
    "wez_hold_growth_per_step": 0.08,
    "wez_hold_growth_cap_steps": 15,
    "wez_hold_break_penalty_scale": 0.0,
    "wez_hold_break_min_streak": 8,
    # 2026-08-11: directional pursuit-heading shaping -- see module docstring.
    # 0.0 by default (opt-in via YAML).
    "pursuit_heading_scale": 0.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# Module-level previous-step cache. Safe only for a single sequential env
# instance (this project always runs num-env-runners=1 / num-envs-per-env-runner=1
# for these pilots) -- reset whenever the previous call ended the episode, so a
# fresh episode's first step never sees a stale delta from the last one.
_prev_distance: float | None = None
_prev_ata: float | None = None
_hold_streak: int = 0
_prev_position: np.ndarray | None = None


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
    global _prev_distance, _prev_ata, _hold_streak, _prev_position

    cfg = {**MY_REWARD_CONFIG, **reward_config}
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    aa = abs(float(geo_info._get_aspect_angle(ownship_state, target_state, False)))
    altitude = float(ownship_state[StateIndex.ALT])
    roll = float(ownship_state[StateIndex.ROLL])
    pitch = float(ownship_state[StateIndex.PITCH])

    components: dict[str, float] = {
        "step": float(cfg["step_penalty"]),
        "survival": float(cfg.get("survival_bonus", 0.0)),
    }

    ideal_min = float(cfg["ideal_range_min_m"])
    ideal_max = float(cfg["ideal_range_max_m"])
    ideal_mid = 0.5 * (ideal_min + ideal_max)
    ideal_half_width = max(1.0, 0.5 * (ideal_max - ideal_min))
    range_score = 1.0 - abs(distance - ideal_mid) / ideal_half_width
    components["range"] = float(cfg["range_scale"]) * max(-4.0, range_score)

    if distance < float(cfg["too_close_m"]):
        ratio = _clamp01(1.0 - distance / max(1.0, float(cfg["too_close_m"])))
        components["overshoot"] = -float(cfg["overshoot_penalty"]) * (
            ratio + float(cfg["overshoot_quadratic_scale"]) * ratio * ratio
        )
    else:
        components["overshoot"] = 0.0
    components["inside_min_range"] = (
        float(cfg["inside_min_range_penalty"])
        if distance < float(wez_config["min_range_m"])
        else 0.0
    )

    components["ata"] = float(cfg["ata_scale"]) * max(-1.0, 1.0 - ata / 90.0)
    components["aa"] = float(cfg["aa_scale"]) * max(-1.0, 1.0 - aa / 180.0)
    in_wez = (
        float(wez_config["min_range_m"]) <= distance <= float(wez_config["max_range_m"])
        and ata <= float(wez_config["angle_deg"]) / 2.0
    )
    components["wez"] = float(cfg["wez_bonus"]) if in_wez else 0.0
    components["damage"] = float(cfg["damage_scale"]) * (
        float(target_damage) - float(ownship_damage)
    )

    if float(cfg["altitude_bonus_high_min_m"]) <= altitude <= float(
        cfg["altitude_bonus_high_max_m"]
    ):
        components["altitude"] = float(cfg["altitude_bonus_high"])
    elif float(cfg["altitude_bonus_mid_min_m"]) <= altitude < float(
        cfg["altitude_bonus_high_min_m"]
    ):
        components["altitude"] = float(cfg["altitude_bonus_mid"])
    else:
        components["altitude"] = 0.0

    components["nose_down"] = (
        float(cfg["nose_down_penalty"])
        if altitude < float(cfg["nose_down_altitude_m"])
        and pitch < float(cfg["nose_down_pitch_deg"])
        else 0.0
    )

    safety = 0.0
    if altitude < float(cfg["altitude_soft_floor_m"]):
        safety -= float(cfg["low_altitude_penalty"]) * _clamp01(
            (float(cfg["altitude_soft_floor_m"]) - altitude)
            / max(
                1.0,
                float(cfg["altitude_soft_floor_m"])
                - float(cfg["altitude_hard_floor_m"]),
            )
        )
    if altitude < float(cfg["altitude_hard_floor_m"]):
        safety -= float(cfg["very_low_altitude_penalty"])
    components["safety"] = safety

    control = 0.0
    if abs(roll) > float(cfg["roll_limit_deg"]):
        control -= float(cfg["roll_limit_penalty"]) * _clamp01(
            (abs(roll) - float(cfg["roll_limit_deg"]))
            / max(1.0, 180.0 - float(cfg["roll_limit_deg"]))
        )
    if pitch < float(cfg["pitch_down_limit_deg"]):
        control -= float(cfg["pitch_down_penalty"]) * _clamp01(
            (float(cfg["pitch_down_limit_deg"]) - pitch)
            / max(1.0, 90.0 + float(cfg["pitch_down_limit_deg"]))
        )
    if pitch > float(cfg["pitch_up_limit_deg"]):
        control -= float(cfg["pitch_up_penalty"]) * _clamp01(
            (pitch - float(cfg["pitch_up_limit_deg"]))
            / max(1.0, 90.0 - float(cfg["pitch_up_limit_deg"]))
        )
    components["control_stability"] = control

    attack_min = float(cfg["attack_range_min_m"])
    attack_max = float(cfg["attack_range_max_m"])
    components["attack_range"] = (
        float(cfg["attack_range_bonus"])
        if attack_min <= distance <= attack_max
        else 0.0
    )
    far_start = float(cfg["far_range_penalty_start_m"])
    far_span = float(cfg.get("far_range_penalty_span_m", far_start * 6.0))
    components["far_range"] = (
        -float(cfg["far_range_penalty"])
        * _clamp01((distance - far_start) / max(1.0, far_span))
        if distance > far_start
        else 0.0
    )

    # Delta shaping: reward closing distance / reducing ata step-over-step,
    # independent of the absolute band. Clipped so a single-step teleport
    # (episode reset, geometry-guard snap) can't dominate the reward.
    #
    # Safety gate (2026-08-06 fix): stage6d_delta_reward_400iter collapsed into
    # repeated nose-down crashes at ~285m alt after ~iter100 -- accumulated
    # range_delta reward over many steps (up to range_delta_scale*range_delta_clip_m
    # per step) exceeded the one-time crash_penalty, making "dive fast toward
    # the target, eat the crash" net-positive. Gating delta reward off below
    # altitude_soft_floor_m removes that incentive: closing distance only pays
    # while altitude is still in the safe band, so diving into the danger zone
    # can no longer be paid for by delta reward, only by the (much smaller,
    # already-existing) safety/altitude penalties.
    range_delta_reward = 0.0
    ata_delta_reward = 0.0
    ata_recovery_reward = 0.0
    wez_proximity_reward = 0.0
    wez_hold_reward = 0.0
    wez_hold_break_reward = 0.0
    safe_altitude = altitude >= float(cfg["altitude_soft_floor_m"])
    ata_delta = 0.0
    if _prev_distance is not None and safe_altitude:
        range_delta = _prev_distance - distance  # positive = closed distance
        range_delta = max(
            -float(cfg["range_delta_clip_m"]),
            min(float(cfg["range_delta_clip_m"]), range_delta),
        )
        range_delta_reward = float(cfg["range_delta_scale"]) * range_delta
    if _prev_ata is not None and safe_altitude:
        ata_delta = _prev_ata - ata  # positive = pointed more at target
        ata_delta = max(
            -float(cfg["ata_delta_clip_deg"]),
            min(float(cfg["ata_delta_clip_deg"]), ata_delta),
        )
        ata_delta_reward = float(cfg["ata_delta_scale"]) * ata_delta
        # Recovery bonus: only while starting from a bad (behind/beside) angle
        # and actively closing it -- does not touch normal in-band holding.
        if (
            _prev_ata > float(cfg["ata_recovery_threshold_deg"])
            and ata_delta > 0.0
        ):
            ata_recovery_reward = float(cfg["ata_recovery_scale"]) * ata_delta
    wez_hold_active = (
        float(cfg["wez_hold_scale"]) != 0.0
        or float(cfg["wez_hold_break_penalty_scale"]) != 0.0
    )
    if safe_altitude and (float(cfg["wez_proximity_scale"]) != 0.0 or wez_hold_active):
        wez_min = float(wez_config["min_range_m"])
        wez_max = float(wez_config["max_range_m"])
        wez_mid = 0.5 * (wez_min + wez_max)
        wez_half_width = max(1.0, 0.5 * (wez_max - wez_min))
        range_closeness = _clamp01(1.0 - abs(distance - wez_mid) / wez_half_width)
        half_wez_angle = float(wez_config["angle_deg"]) / 2.0
        ata_alignment = (
            _clamp01(1.0 - ata / half_wez_angle) if half_wez_angle > 0.0 else 0.0
        )
        aa_alignment = (
            _clamp01(1.0 - aa / 180.0)
            if bool(cfg.get("wez_proximity_use_aa", False))
            else 1.0
        )
        quality = range_closeness * ata_alignment * aa_alignment

        if float(cfg["wez_proximity_scale"]) != 0.0:
            wez_proximity_reward = float(cfg["wez_proximity_scale"]) * quality

        if wez_hold_active:
            growth_cap = int(cfg["wez_hold_growth_cap_steps"])
            if quality >= float(cfg["wez_hold_quality_threshold"]):
                _hold_streak += 1
                streak_factor = min(_hold_streak, growth_cap)
                wez_hold_reward = (
                    float(cfg["wez_hold_scale"])
                    * quality
                    * (1.0 + float(cfg["wez_hold_growth_per_step"]) * streak_factor)
                )
            else:
                if _hold_streak >= int(cfg["wez_hold_break_min_streak"]):
                    broken_streak = min(_hold_streak, growth_cap)
                    wez_hold_break_reward = (
                        -float(cfg["wez_hold_break_penalty_scale"]) * broken_streak
                    )
                _hold_streak = 0
    else:
        _hold_streak = 0

    # Directional pursuit-heading shaping (2026-08-11): rewards this step's
    # actual flight-path direction for pointing at the target's current
    # position, independent of whether range has started shrinking yet --
    # see module docstring for full rationale.
    pursuit_heading_reward = 0.0
    current_position = np.asarray(position_ned(ownship_state), dtype=np.float64)
    if (
        safe_altitude
        and float(cfg["pursuit_heading_scale"]) != 0.0
        and _prev_position is not None
    ):
        displacement = current_position - _prev_position
        displacement_norm = float(np.linalg.norm(displacement))
        target_position = np.asarray(position_ned(target_state), dtype=np.float64)
        los_vector = target_position - current_position
        los_norm = float(np.linalg.norm(los_vector))
        if displacement_norm > 1e-3 and los_norm > 1e-3:
            motion_dir = displacement / displacement_norm
            los_dir = los_vector / los_norm
            closure_alignment = float(np.clip(np.dot(motion_dir, los_dir), -1.0, 1.0))
            pursuit_heading_reward = float(cfg["pursuit_heading_scale"]) * closure_alignment

    components["range_delta"] = range_delta_reward
    components["ata_delta"] = ata_delta_reward
    components["ata_recovery"] = ata_recovery_reward
    components["wez_proximity"] = wez_proximity_reward
    components["wez_hold"] = wez_hold_reward
    components["wez_hold_break"] = wez_hold_break_reward
    components["pursuit_heading"] = pursuit_heading_reward

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

    if terminated or truncated:
        _prev_distance = None
        _prev_ata = None
        _hold_streak = 0
        _prev_position = None
    else:
        _prev_distance = distance
        _prev_ata = ata
        _prev_position = current_position

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
