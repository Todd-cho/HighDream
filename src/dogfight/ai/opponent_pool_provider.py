"""Diverse, per-episode-sampled opponent behaviors for residual RL training.

Both RL_TRAINING_HANDOFF_KO.md and RL_TRAINING_ADDENDUM_W56_KO.txt warn that
training against a single opponent behavior lets the policy memorize that
opponent's trajectory instead of learning a general correction. Before this
module, the only genuinely active opponent available was
``ScriptedPursuitActionProvider`` (always-on pursuit), and it could not be
mixed with other behaviors episode-to-episode: ``DogFightEnv._step_target_aircraft``
uses a single ``target_action_provider`` object for the entire run whenever
one is set, ignoring any per-episode ``initial_scenario`` scenario_pool
``target_mode`` value (see ``single_agent_env.py`` -- the action_provider
branch always short-circuits the string-dispatched modes).

``OpponentPoolActionProvider`` fixes this at the right layer: it IS the
single ``target_action_provider`` for the run, but internally samples (in
training mode) or deterministically cycles through (in eval mode) one of
several named behaviors -- and their randomized parameters -- every episode
reset, then delegates every ``compute_action`` call to whichever behavior
was selected for that episode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.scripted_pursuit_provider import ScriptedPursuitActionProvider
from dogfight.sim.state_schema import StateIndex


class _BankHoldActionProvider(ActionProvider):
    """Shared nested-P-controller structure (heading/bank -> roll,
    altitude/pitch -> pitch, speed -> throttle). Reused, not reinvented, from
    ScriptedPursuitActionProvider's validated structure (2026-08-11: a
    single-shot error-to-command law and JSBSim's own step_autopilot() both
    diverged/crashed; this two-stage structure is what stayed stable)."""

    def __init__(
        self,
        cruise_altitude_m: float,
        cruise_speed_kcas: float,
        max_bank_deg: float,
        bank_rate_gain: float,
        max_pitch_deg: float,
        altitude_to_pitch_gain: float,
        bank_pitch_compensation_gain: float,
        pitch_rate_gain: float,
        speed_gain: float,
        base_throttle: float,
    ):
        self.cruise_altitude_m = cruise_altitude_m
        self.cruise_speed_kcas = cruise_speed_kcas
        self.max_bank_deg = max_bank_deg
        self.bank_rate_gain = bank_rate_gain
        self.max_pitch_deg = max_pitch_deg
        self.altitude_to_pitch_gain = altitude_to_pitch_gain
        self.bank_pitch_compensation_gain = bank_pitch_compensation_gain
        self.pitch_rate_gain = pitch_rate_gain
        self.speed_gain = speed_gain
        self.base_throttle = base_throttle
        self._start_time: float | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self._start_time = None

    @staticmethod
    def _wrap180(deg: float) -> float:
        return (deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _desired_bank_deg(self, context: ActionContext, elapsed: float, my_roll: float, my_yaw: float) -> float:
        raise NotImplementedError

    def _desired_altitude_m(self, context: ActionContext, elapsed: float) -> float:
        return self.cruise_altitude_m

    def compute_action(self, context: ActionContext) -> ActionResult:
        my = context.ownship_state
        now = float(my[StateIndex.SIM_TIME])
        if self._start_time is None:
            self._start_time = now
        elapsed = now - self._start_time

        my_alt = float(my[StateIndex.ALT])
        my_roll = float(my[StateIndex.ROLL])
        my_pitch = float(my[StateIndex.PITCH])
        my_yaw = float(my[StateIndex.YAW])
        my_kcas = float(my[StateIndex.KCAS])

        desired_bank = self._clip(
            self._desired_bank_deg(context, elapsed, my_roll, my_yaw),
            -self.max_bank_deg,
            self.max_bank_deg,
        )
        bank_error = self._wrap180(desired_bank - my_roll)
        roll_action = self._clip(bank_error * self.bank_rate_gain, -1.0, 1.0)

        desired_altitude = self._desired_altitude_m(context, elapsed)
        altitude_error = desired_altitude - my_alt
        desired_pitch = altitude_error * self.altitude_to_pitch_gain
        desired_pitch += self.bank_pitch_compensation_gain * abs(my_roll)
        desired_pitch = self._clip(desired_pitch, -self.max_pitch_deg, self.max_pitch_deg)
        pitch_error = desired_pitch - my_pitch
        # action[1]=+1 pitches DOWN (verified 2026-08-11), so a positive
        # pitch_error (need more nose-up) requires a *negative* action.
        pitch_action = self._clip(-pitch_error * self.pitch_rate_gain, -1.0, 1.0)

        speed_error = self.cruise_speed_kcas - my_kcas
        throttle_action = self._clip(self.base_throttle + speed_error * self.speed_gain, 0.0, 1.0)

        action = clip_action([roll_action, pitch_action, 0.0, throttle_action])
        return ActionResult(action=action, source=self.__class__.__name__, confidence=0.9)


class StraightLevelActionProvider(_BankHoldActionProvider):
    """Straight and (near-)level: holds the heading measured at reset."""

    def __init__(self, heading_to_bank_gain: float = 1.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.heading_to_bank_gain = heading_to_bank_gain
        self._initial_heading: float | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        super().reset(context)
        self._initial_heading = None

    def _desired_bank_deg(self, context, elapsed, my_roll, my_yaw):
        if self._initial_heading is None:
            self._initial_heading = my_yaw
        heading_error = self._wrap180(self._initial_heading - my_yaw)
        return heading_error * self.heading_to_bank_gain


class SteadyTurnActionProvider(_BankHoldActionProvider):
    """Sustained coordinated turn at a fixed bank angle and direction."""

    def __init__(self, turn_bank_deg: float = 30.0, turn_sign: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.turn_bank_deg = turn_bank_deg
        self.turn_sign = turn_sign

    def _desired_bank_deg(self, context, elapsed, my_roll, my_yaw):
        return self.turn_sign * self.turn_bank_deg


class VerticalWeaveActionProvider(_BankHoldActionProvider):
    """Straight heading with a sinusoidal altitude weave -- the addendum's
    "vertical jink/weave" opponent type."""

    def __init__(
        self,
        weave_amplitude_m: float = 300.0,
        weave_period_s: float = 12.0,
        weave_phase_s: float = 0.0,
        heading_to_bank_gain: float = 1.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.weave_amplitude_m = weave_amplitude_m
        self.weave_period_s = max(1.0, weave_period_s)
        self.weave_phase_s = weave_phase_s
        self.heading_to_bank_gain = heading_to_bank_gain
        self._initial_heading: float | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        super().reset(context)
        self._initial_heading = None

    def _desired_bank_deg(self, context, elapsed, my_roll, my_yaw):
        if self._initial_heading is None:
            self._initial_heading = my_yaw
        heading_error = self._wrap180(self._initial_heading - my_yaw)
        return heading_error * self.heading_to_bank_gain

    def _desired_altitude_m(self, context, elapsed):
        phase = 2.0 * math.pi * (elapsed + self.weave_phase_s) / self.weave_period_s
        return self.cruise_altitude_m + self.weave_amplitude_m * math.sin(phase)


@dataclass
class OpponentPoolEntry:
    name: str
    weight: float
    factory: Callable[[np.random.Generator, dict], tuple[ActionProvider, dict]]


def _sample_range(rng: np.random.Generator, low: float, high: float) -> float:
    if high <= low:
        return float(low)
    return float(rng.uniform(low, high))


_COMMON_HOLD_DEFAULTS = dict(
    cruise_altitude_m=7000.0,
    cruise_speed_kcas=260.0,
    max_bank_deg=70.0,
    bank_rate_gain=0.035,
    max_pitch_deg=25.0,
    altitude_to_pitch_gain=0.02,
    bank_pitch_compensation_gain=0.15,
    pitch_rate_gain=0.05,
    speed_gain=0.01,
    base_throttle=0.65,
)


def _straight_level_factory(cfg: dict) -> Callable[[np.random.Generator, dict], tuple[ActionProvider, dict]]:
    ranges = cfg.get("ranges", {})

    def factory(rng: np.random.Generator, common: dict) -> tuple[ActionProvider, dict]:
        speed = _sample_range(rng, *ranges.get("cruise_speed_kcas", [common["cruise_speed_kcas"]] * 2))
        params = {"cruise_speed_kcas": speed}
        provider = StraightLevelActionProvider(
            **{**_COMMON_HOLD_DEFAULTS, **common, "cruise_speed_kcas": speed}
        )
        return provider, params

    return factory


def _steady_turn_factory(cfg: dict) -> Callable[[np.random.Generator, dict], tuple[ActionProvider, dict]]:
    ranges = cfg.get("ranges", {})

    def factory(rng: np.random.Generator, common: dict) -> tuple[ActionProvider, dict]:
        bank = _sample_range(rng, *ranges.get("turn_bank_deg", [25.0, 45.0]))
        sign = float(rng.choice([-1.0, 1.0]))
        params = {"turn_bank_deg": bank, "turn_sign": sign}
        provider = SteadyTurnActionProvider(
            turn_bank_deg=bank,
            turn_sign=sign,
            **{**_COMMON_HOLD_DEFAULTS, **common},
        )
        return provider, params

    return factory


def _scripted_pursuit_factory(cfg: dict) -> Callable[[np.random.Generator, dict], tuple[ActionProvider, dict]]:
    ranges = cfg.get("ranges", {})

    def factory(rng: np.random.Generator, common: dict) -> tuple[ActionProvider, dict]:
        speed = _sample_range(rng, *ranges.get("cruise_speed_kcas", [common["cruise_speed_kcas"]] * 2))
        params = {"cruise_speed_kcas": speed}
        provider = ScriptedPursuitActionProvider(
            cruise_altitude_m=common["cruise_altitude_m"],
            cruise_speed_kcas=speed,
            max_bank_deg=common["max_bank_deg"],
            max_pitch_deg=common["max_pitch_deg"],
        )
        return provider, params

    return factory


def _vertical_weave_factory(cfg: dict) -> Callable[[np.random.Generator, dict], tuple[ActionProvider, dict]]:
    ranges = cfg.get("ranges", {})

    def factory(rng: np.random.Generator, common: dict) -> tuple[ActionProvider, dict]:
        amplitude = _sample_range(rng, *ranges.get("weave_amplitude_m", [200.0, 500.0]))
        period = _sample_range(rng, *ranges.get("weave_period_s", [8.0, 18.0]))
        phase = _sample_range(rng, 0.0, period)
        params = {
            "weave_amplitude_m": amplitude,
            "weave_period_s": period,
            "weave_phase_s": phase,
        }
        provider = VerticalWeaveActionProvider(
            weave_amplitude_m=amplitude,
            weave_period_s=period,
            weave_phase_s=phase,
            **{**_COMMON_HOLD_DEFAULTS, **common},
        )
        return provider, params

    return factory


_FACTORY_BUILDERS: dict[str, Callable[[dict], Callable]] = {
    "straight_level": _straight_level_factory,
    "steady_turn": _steady_turn_factory,
    "scripted_pursuit": _scripted_pursuit_factory,
    "vertical_weave": _vertical_weave_factory,
}

DEFAULT_OPPONENT_POOL_CONFIG = {
    "common": {
        "cruise_altitude_m": 7000.0,
        "cruise_speed_kcas": 260.0,
        "max_bank_deg": 70.0,
        "max_pitch_deg": 25.0,
    },
    "behaviors": [
        {"type": "straight_level", "weight": 1.0},
        {"type": "steady_turn", "weight": 1.0},
        {"type": "scripted_pursuit", "weight": 1.0},
        {"type": "vertical_weave", "weight": 1.0},
    ],
}


def _build_entries(pool_config: dict) -> tuple[list[OpponentPoolEntry], dict]:
    common = dict(DEFAULT_OPPONENT_POOL_CONFIG["common"])
    common.update(pool_config.get("common", {}) or {})
    behaviors = pool_config.get("behaviors") or DEFAULT_OPPONENT_POOL_CONFIG["behaviors"]
    entries: list[OpponentPoolEntry] = []
    for spec in behaviors:
        behavior_type = str(spec["type"])
        if behavior_type not in _FACTORY_BUILDERS:
            raise ValueError(
                f"Unknown opponent pool behavior type {behavior_type!r}; "
                f"known types: {sorted(_FACTORY_BUILDERS)}"
            )
        weight = float(spec.get("weight", 1.0))
        factory = _FACTORY_BUILDERS[behavior_type](spec)
        name = str(spec.get("name", behavior_type))
        entries.append(OpponentPoolEntry(name=name, weight=weight, factory=factory))
    return entries, common


class OpponentPoolActionProvider(ActionProvider):
    """The single ``target_action_provider`` for a run. Internally samples
    (training) or deterministically cycles (eval, via ``forced_sequence``)
    one opponent behavior + its randomized parameters every episode reset.
    """

    def __init__(
        self,
        pool_config: dict | None = None,
        seed: int | None = None,
        forced_sequence: list[dict] | None = None,
    ):
        self._entries, self._common = _build_entries(pool_config or DEFAULT_OPPONENT_POOL_CONFIG)
        total_weight = sum(entry.weight for entry in self._entries)
        if total_weight <= 0.0:
            raise ValueError("opponent pool requires at least one positive-weight behavior")
        self._probabilities = [entry.weight / total_weight for entry in self._entries]
        self._rng = np.random.default_rng(seed)
        self._forced_sequence = list(forced_sequence) if forced_sequence else None
        self._forced_index = 0
        self._active_name: str | None = None
        self._active_params: dict = {}
        self._active_provider: ActionProvider | None = None
        self.reset_count = 0

    def reset(self, context: ActionContext | None = None) -> None:
        self.reset_count += 1
        if self._forced_sequence:
            spec = self._forced_sequence[self._forced_index % len(self._forced_sequence)]
            self._forced_index += 1
            behavior_type = str(spec["type"])
            factory = _FACTORY_BUILDERS[behavior_type]({"ranges": {}})
            forced_rng = np.random.default_rng(int(spec.get("param_seed", 0)))
            provider, params = factory(forced_rng, self._common)
            for key, value in spec.get("params", {}).items():
                setattr(provider, key, value)
                params[key] = value
            name = str(spec.get("name", behavior_type))
        else:
            index = int(self._rng.choice(len(self._entries), p=self._probabilities))
            entry = self._entries[index]
            provider, params = entry.factory(self._rng, self._common)
            name = entry.name

        self._active_name = name
        self._active_params = params
        self._active_provider = provider
        self._active_provider.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        assert self._active_provider is not None, "OpponentPoolActionProvider.reset() was never called"
        result = self._active_provider.compute_action(context)
        info = dict(result.info)
        info["opponent_name"] = self._active_name
        info["opponent_params"] = dict(self._active_params)
        return ActionResult(action=result.action, source=result.source, confidence=result.confidence, info=info)

    @property
    def active_name(self) -> str | None:
        return self._active_name

    @property
    def active_params(self) -> dict:
        return dict(self._active_params)


__all__ = [
    "DEFAULT_OPPONENT_POOL_CONFIG",
    "OpponentPoolActionProvider",
    "StraightLevelActionProvider",
    "SteadyTurnActionProvider",
    "VerticalWeaveActionProvider",
]


