"""Residual-RL training environment around the frozen W56 rule controller.

W56 (not W53) is the training base per RL_TRAINING_ADDENDUM_W56_KO (2026-08-23):
W53's terminal_vertical_unload released target bank from ~70deg to ~11deg during
close-range alignment, reopening ATA. W56 disables that and widens/enables
prelock terminal tracking instead (see dogfight.ai.rule_profiles.build_w56_config).

Per the addendum, residual RL controls three axes (roll/pitch/throttle, yaw
stays 0 and W56-owned) and is only allowed to act in a specific, threat-aware
window -- NOT a continuous multi-region scale table like the earlier W53
pilot used. Rationale (addendum section 4, live W57 finding): a tight ATA
solution achieved during a mutual head-on (low threat_ata = the opponent is
also nose-on to us) is a bad trade -- W57 hit ATA 0.76deg at 706m but traded
~30% damage dealt for ~70% damage taken. So residual authority is withheld
whenever the encounter is not sufficiently asymmetric in our favor, even if
range/ATA alone look favorable.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionResult, clip_action
from dogfight.ai.rule_profiles import build_w56_config, build_w56_controller
from dogfight.ai.integrated_bfm_controller import IntegratedBFMController
from dogfight.envs.observation import build_observation
from dogfight.envs.single_agent_env import DogFightEnv
from dogfight.sim.state_schema import StateIndex


RESIDUAL_OBSERVATION_SIZE = 36

DEFAULT_RESIDUAL_W56_CONFIG = {
    # Per-axis residual caps (RL_TRAINING_ADDENDUM_W56_KO section 4).
    "max_roll_scale": 0.15,
    "max_pitch_scale": 0.20,
    "max_throttle_scale": 0.15,
    # Activation gate (addendum section 4 "권장 활성/차단 조건").
    "gate_ata_deg": 20.0,
    "gate_range_m": 3000.0,
    "gate_min_threat_ata_deg": 40.0,
    # Addendum section 7, priority 7: small penalty on residual magnitude and
    # step-over-step change. Applied here (not in the reward module) because
    # compute_reward(...)'s fixed signature has no access to the RL action.
    "residual_magnitude_penalty_scale": 0.01,
    "residual_delta_penalty_scale": 0.01,
    # Optional overrides applied on top of the frozen W56 profile, so the
    # 20-iteration A/B pilot (addendum section 5) can vary a single tuned
    # field (e.g. terminal_pitch_los_rate_gain 1.0 vs 1.5) without touching
    # the shared frozen build_w56_config() literal.
    "w56_overrides": {},
}


def compose_w56_residual_action(
    rule_action: np.ndarray,
    raw_residual: np.ndarray,
    active: bool,
    max_roll_scale: float,
    max_pitch_scale: float,
    max_throttle_scale: float,
) -> np.ndarray:
    """Add bounded roll/pitch/throttle residuals while preserving W56 yaw.

    ``raw_residual`` is ``[roll, pitch, throttle]`` in ``[-1, 1]``. When
    ``active`` is False the residual is fully suppressed (scale 0), matching
    the addendum's binary gate rather than a continuous scale.
    """
    rule = np.asarray(rule_action, dtype=np.float32)
    residual = np.asarray(raw_residual, dtype=np.float32)
    if rule.shape != (4,):
        raise ValueError(f"rule_action must have shape (4,), got {rule.shape}")
    if residual.shape != (3,):
        raise ValueError(f"raw_residual must have shape (3,), got {residual.shape}")
    combined = rule.copy()
    if active:
        bounded = np.clip(residual, -1.0, 1.0)
        combined[0] += bounded[0] * max(0.0, float(max_roll_scale))
        combined[1] += bounded[1] * max(0.0, float(max_pitch_scale))
        combined[3] += bounded[2] * max(0.0, float(max_throttle_scale))
    return clip_action(combined)


class ResidualW56Env(DogFightEnv):
    """Train three residual axes while frozen W56 retains full-flight control.

    RL actions are ``[residual_roll, residual_pitch, residual_throttle]`` in
    ``[-1, 1]``. W56 owns yaw always, and owns roll/pitch/throttle whenever
    the activation gate is closed. The combined simulator-format action is
    passed through the existing rate limiter and safety override unchanged.
    """

    def __init__(self, env_config: dict | None = None, **kwargs: Any):
        cfg = dict(env_config or {})
        residual_cfg = {
            **DEFAULT_RESIDUAL_W56_CONFIG,
            **dict(cfg.get("residual_w56", {}) or {}),
        }
        self._residual_cfg = residual_cfg
        overrides = dict(residual_cfg.get("w56_overrides", {}) or {})
        w56_config = build_w56_config()
        for key, value in overrides.items():
            if not hasattr(w56_config, key):
                raise ValueError(f"Unknown W56 config field override: {key!r}")
            setattr(w56_config, key, value)
        self._w56 = IntegratedBFMController(w56_config)
        self._rule_cache_time: float | None = None
        self._rule_cache_result: ActionResult | None = None
        self._active_rule_result: ActionResult | None = None
        self._raw_residual = np.zeros(3, dtype=np.float32)
        self._previous_raw_residual = np.zeros(3, dtype=np.float32)
        self._last_gate_active = False
        self._last_rule_action = np.zeros(4, dtype=np.float32)
        self._last_combined_action = np.zeros(4, dtype=np.float32)
        self._gate_prev_altitude: float | None = None
        self._ep_residual_sq_sum = 0.0
        self._ep_residual_delta_sum = 0.0
        self._ep_residual_saturation_steps = 0
        self._ep_gate_active_steps = 0
        super().__init__(env_config=cfg, **kwargs)
        self.action_space = gym.spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(RESIDUAL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.num_observation = RESIDUAL_OBSERVATION_SIZE
        self.pre_obs = np.zeros(RESIDUAL_OBSERVATION_SIZE, dtype=np.float32)
        self.config["observation_mode"] = "residual_w56_35"
        self.config["observation_summary"] = {
            "mode": "residual_w56_35",
            "size": RESIDUAL_OBSERVATION_SIZE,
            "description": (
                "tactical19 + frozen W56 action + measured rates/closure + "
                "W56 state flags + threat ATA + gate flag + previous 3-axis residual"
            ),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._w56.reset()
        self._invalidate_rule_cache()
        self._active_rule_result = None
        self._raw_residual = np.zeros(3, dtype=np.float32)
        self._previous_raw_residual = np.zeros(3, dtype=np.float32)
        self._last_gate_active = False
        self._last_rule_action = np.zeros(4, dtype=np.float32)
        self._last_combined_action = np.zeros(4, dtype=np.float32)
        self._gate_prev_altitude = None
        self._ep_residual_sq_sum = 0.0
        self._ep_residual_delta_sum = 0.0
        self._ep_residual_saturation_steps = 0
        self._ep_gate_active_steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (3,):
            raise ValueError(f"ResidualW56Env action must have shape (3,), got {raw.shape}")
        raw = np.clip(raw, -1.0, 1.0)
        delta = raw - self._previous_raw_residual
        self._raw_residual = raw
        self._previous_raw_residual = raw.copy()
        self._ep_residual_sq_sum += float(np.dot(raw, raw))
        self._ep_residual_delta_sum += float(np.linalg.norm(delta))
        if np.any(np.abs(raw) >= 0.999):
            self._ep_residual_saturation_steps += 1

        # DogFightEnv's episode action metrics remain four-axis. These values
        # describe the raw residual only; actual combined actions are exposed
        # separately in info below.
        metric_action = np.array([raw[0], raw[1], 0.0, raw[2]], dtype=np.float32)
        # W56 is a 10 Hz policy in the live path. Compute it once per outer
        # environment decision and hold it across all JSBSim step_ratio ticks,
        # exactly as the RL residual is held.
        self._active_rule_result = self._get_rule_result()
        try:
            obs, reward, terminated, truncated, info = super().step(metric_action)
        finally:
            self._active_rule_result = None

        # Addendum section 7 priority 7: small residual magnitude/delta
        # penalty, applied only while the gate actually let residual affect
        # the flight (compute_reward(...)'s fixed signature has no access to
        # the RL action, so this cannot live in the reward module itself).
        residual_penalty = 0.0
        if self._last_gate_active:
            cfg = self._residual_cfg
            residual_penalty = -(
                float(cfg["residual_magnitude_penalty_scale"]) * float(np.dot(raw, raw))
                + float(cfg["residual_delta_penalty_scale"]) * float(np.linalg.norm(delta))
            )
            reward = float(reward) + residual_penalty

        if self._last_gate_active:
            self._ep_gate_active_steps += 1
        steps = max(1, int(info.get("ep_step_count", 1)))
        info.update({
            "residual_raw": raw.copy(),
            "gate_active": bool(self._last_gate_active),
            "rule_action": self._last_rule_action.copy(),
            "combined_action": self._last_combined_action.copy(),
            "residual_penalty": residual_penalty,
            "ep_residual_rms": float(np.sqrt(self._ep_residual_sq_sum / (3.0 * steps))),
            "ep_residual_delta_mean": float(self._ep_residual_delta_sum / steps),
            "ep_residual_saturation_rate": float(self._ep_residual_saturation_steps / steps),
            "ep_gate_active_rate": float(self._ep_gate_active_steps / steps),
        })
        self.info.update(info)
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        base = build_observation(
            "tactical19",
            self._ownship_state,
            self._target_state,
            self._geo_info,
            self._wez,
        )
        result = self._get_rule_result()
        info = result.info
        obs = np.zeros(RESIDUAL_OBSERVATION_SIZE, dtype=np.float32)
        obs[:19] = base
        obs[19:22] = np.clip(result.action[:3], -1.0, 1.0)
        obs[22] = np.clip(2.0 * float(result.action[3]) - 1.0, -1.0, 1.0)
        obs[23] = self._norm(info.get("measured_roll_rate", 0.0), 90.0)
        obs[24] = self._norm(info.get("measured_pitch_rate", 0.0), 45.0)
        obs[25] = self._norm(info.get("own_yaw_rate", 0.0), 45.0)
        obs[26] = self._norm(info.get("target_yaw_rate", 0.0), 45.0)
        obs[27] = self._norm(info.get("los_rate", 0.0), 45.0)
        obs[28] = self._norm(info.get("closure_rate", 0.0), 600.0)
        obs[29] = 1.0 if info.get("terminal_track_active", False) else -1.0
        obs[30] = 1.0 if info.get("turn_match_active", False) else -1.0
        obs[31] = self._norm(info.get("threat_ata", 180.0), 180.0)
        obs[32] = 1.0 if self._gate_active(info) else -1.0
        obs[33:36] = self._previous_raw_residual
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _step_controlled_aircraft(self, action: np.ndarray) -> None:
        result = self._active_rule_result or self._get_rule_result()
        active = self._gate_active(result.info)
        cfg = self._residual_cfg
        combined = compose_w56_residual_action(
            result.action,
            self._raw_residual,
            active,
            float(cfg["max_roll_scale"]),
            float(cfg["max_pitch_scale"]),
            float(cfg["max_throttle_scale"]),
        )
        self._last_gate_active = active
        self._last_rule_action = np.asarray(result.action, dtype=np.float32).copy()
        self._last_combined_action = combined.copy()
        limited = self._apply_action_rate_limit(combined)
        self._sim.step(self._apply_safety_override(limited))

    def _get_rule_result(self) -> ActionResult:
        now = float(self._ownship_state[StateIndex.SIM_TIME])
        if self._rule_cache_result is not None and self._rule_cache_time == now:
            return self._rule_cache_result
        context = ActionContext(
            sim=self._sim,
            opponent_sim=self._target_sim,
            ownship_state=np.array(self._ownship_state, copy=True),
            target_state=np.array(self._target_state, copy=True),
            observation=None,
            info={"timestep": self.current_timestep},
        )
        result = self._w56.compute_action(context)
        self._rule_cache_time = now
        self._rule_cache_result = result
        return result

    def _gate_active(self, info: dict) -> bool:
        """Addendum section 4 activation gate: distance/ATA window, not a
        mutual head-on threat, no active/imminent safety recovery."""
        cfg = self._residual_cfg
        ata = abs(float(info.get("ata", 180.0)))
        distance = float(info.get("distance", float("inf")))
        threat_ata = abs(float(info.get("threat_ata", 0.0)))
        if ata > float(cfg["gate_ata_deg"]):
            return False
        if distance > float(cfg["gate_range_m"]):
            return False
        if threat_ata < float(cfg["gate_min_threat_ata_deg"]):
            return False
        if self._danger_altitude_active():
            return False
        return True

    def _danger_altitude_active(self) -> bool:
        """Read-only mirror of DogFightEnv._apply_safety_override's trigger
        conditions, used only to decide whether residual authority should be
        withheld -- the real enforcement still runs unconditionally on the
        final composed action via self._apply_safety_override(). Tracks its
        own previous-altitude sample so it never disturbs the base class's
        own _safety_override_prev_altitude bookkeeping/step counter."""
        altitude = float(self._ownship_state[StateIndex.ALT])
        pitch = float(self._ownship_state[StateIndex.PITCH])
        vertical_rate = None
        if self._gate_prev_altitude is not None:
            vertical_rate = (altitude - self._gate_prev_altitude) / self._delta_t
        self._gate_prev_altitude = altitude

        if altitude < self._safety_override_hard_floor_m:
            return True
        if (
            altitude < self._safety_override_altitude_m
            and pitch <= self._safety_override_pitch_deg
        ):
            return True
        if vertical_rate is not None and vertical_rate < 0.0:
            time_to_impact = (altitude - self._min_altitude) / max(1.0, -vertical_rate)
            if time_to_impact < self._safety_override_time_horizon_s:
                return True
        return False

    def _invalidate_rule_cache(self) -> None:
        self._rule_cache_time = None
        self._rule_cache_result = None

    @staticmethod
    def _norm(value: float, magnitude: float) -> float:
        return float(np.clip(float(value) / magnitude, -1.0, 1.0))


__all__ = [
    "DEFAULT_RESIDUAL_W56_CONFIG",
    "RESIDUAL_OBSERVATION_SIZE",
    "ResidualW56Env",
    "compose_w56_residual_action",
]
