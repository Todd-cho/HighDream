"""Validation and deterministic sampling for initial aircraft scenarios."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


AIRCRAFT_STATE_SIZE = 7
SUPPORTED_TARGET_MODES = {
    "fixed",
    "loiter",
    "autopilot",
    "behavior_tree",
}
_RANDOMIZATION_KEYS = {
    "radius",
    "n_m",
    "e_m",
    "d_m",
    "r_roll",
    "r_pitch",
    "r_heading",
    "speed_mps",
}
_SHARED_RANDOMIZATION_KEYS = {
    "radius",
    "n_m",
    "e_m",
    "d_m",
    "speed_mps",
}


def validate_scenario_pool(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize a ``scenario_pool`` configuration."""
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ValueError("initial_scenario.scenarios must be a non-empty list")
    if not scenarios:
        raise ValueError("initial_scenario.scenarios must not be empty")

    for key in (
        "ownship_randomization",
        "target_randomization",
    ):
        _validate_randomization(config.get(key, {}), key, _RANDOMIZATION_KEYS)
    _validate_randomization(
        config.get("shared_randomization", {}),
        "shared_randomization",
        _SHARED_RANDOMIZATION_KEYS,
    )

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    total_weight = 0.0
    for index, raw_scenario in enumerate(scenarios):
        if not isinstance(raw_scenario, Mapping):
            raise ValueError(f"scenario_pool.scenarios[{index}] must be a mapping")
        scenario = copy.deepcopy(dict(raw_scenario))
        name = str(scenario.get("name", f"scenario_{index}")).strip()
        if not name:
            raise ValueError(f"scenario_pool.scenarios[{index}].name must not be empty")
        if name in names:
            raise ValueError(f"scenario_pool scenario name is duplicated: {name!r}")
        names.add(name)

        weight = _finite_float(scenario.get("weight", 1.0), f"{name}.weight")
        if weight < 0.0:
            raise ValueError(f"{name}.weight must be non-negative")
        total_weight += weight

        scenario["name"] = name
        scenario["weight"] = weight
        scenario["ownship"] = _aircraft_state(scenario.get("ownship"), name, "ownship")
        scenario["target"] = _aircraft_state(scenario.get("target"), name, "target")

        target_mode = str(scenario.get("target_mode", "")).strip()
        if target_mode and target_mode not in SUPPORTED_TARGET_MODES:
            choices = ", ".join(sorted(SUPPORTED_TARGET_MODES))
            raise ValueError(
                f"{name}.target_mode must be one of {choices}; got {target_mode!r}"
            )
        if target_mode:
            scenario["target_mode"] = target_mode

        for key in ("ownship_randomization", "target_randomization"):
            _validate_randomization(scenario.get(key, {}), f"{name}.{key}", _RANDOMIZATION_KEYS)
        _validate_randomization(
            scenario.get("shared_randomization", {}),
            f"{name}.shared_randomization",
            _SHARED_RANDOMIZATION_KEYS,
        )
        normalized.append(scenario)

    if total_weight <= 0.0:
        raise ValueError("scenario_pool requires at least one positive scenario weight")
    return normalized


def sample_scenario_pool(
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], int]:
    """Sample and randomize one scenario using the environment RNG."""
    scenarios = validate_scenario_pool(config)
    weights = np.asarray([scenario["weight"] for scenario in scenarios], dtype=np.float64)
    probabilities = weights / float(weights.sum())
    index = int(rng.choice(len(scenarios), p=probabilities))
    sampled = copy.deepcopy(scenarios[index])

    ownship_randomization = _merged_randomization(
        config.get("ownship_randomization", {}),
        sampled.get("ownship_randomization", {}),
    )
    target_randomization = _merged_randomization(
        config.get("target_randomization", {}),
        sampled.get("target_randomization", {}),
    )
    shared_randomization = _merged_randomization(
        config.get("shared_randomization", {}),
        sampled.get("shared_randomization", {}),
    )

    ownship = randomize_aircraft_state(sampled["ownship"], ownship_randomization, rng)
    target = randomize_aircraft_state(sampled["target"], target_randomization, rng)
    ownship, target = apply_shared_randomization(
        ownship,
        target,
        shared_randomization,
        rng,
    )
    sampled["ownship"] = ownship
    sampled["target"] = target
    return sampled, index


def randomize_aircraft_state(
    state: Sequence[float],
    config: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> list[float]:
    """Return an independently randomized aircraft state vector."""
    result = [float(value) for value in state]
    randomization = dict(config or {})
    if not randomization.get("enabled", False):
        return result

    radius = float(randomization.get("radius", 0.0))
    result[0] += _symmetric_offset(rng, randomization.get("n_m", radius))
    result[1] += _symmetric_offset(rng, randomization.get("e_m", radius))
    result[2] += _symmetric_offset(rng, randomization.get("d_m", radius))
    result[3] = _wrap_signed_angle(
        result[3] + _symmetric_offset(rng, randomization.get("r_roll", 0.0))
    )
    result[4] = _wrap_signed_angle(
        result[4] + _symmetric_offset(rng, randomization.get("r_pitch", 0.0))
    )
    result[5] = float(
        (result[5] + _symmetric_offset(rng, randomization.get("r_heading", 0.0)))
        % 360.0
    )
    result[6] = max(
        0.0,
        result[6] + _symmetric_offset(rng, randomization.get("speed_mps", 0.0)),
    )
    return result


def apply_shared_randomization(
    ownship: Sequence[float],
    target: Sequence[float],
    config: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """Apply the same translation and speed offset to both aircraft."""
    own = [float(value) for value in ownship]
    tgt = [float(value) for value in target]
    randomization = dict(config or {})
    if not randomization.get("enabled", False):
        return own, tgt

    radius = float(randomization.get("radius", 0.0))
    offsets = (
        _symmetric_offset(rng, randomization.get("n_m", radius)),
        _symmetric_offset(rng, randomization.get("e_m", radius)),
        _symmetric_offset(rng, randomization.get("d_m", radius)),
    )
    speed_offset = _symmetric_offset(rng, randomization.get("speed_mps", 0.0))
    for state in (own, tgt):
        state[0] += offsets[0]
        state[1] += offsets[1]
        state[2] += offsets[2]
        state[6] = max(0.0, state[6] + speed_offset)
    return own, tgt


def describe_initial_scenario(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a compact, reproducible description for logs and bundle metadata."""
    scenario_config = copy.deepcopy(dict(config or {}))
    mode = str(scenario_config.get("mode", "default"))
    canonical = json.dumps(
        scenario_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    description: dict[str, Any] = {
        "mode": mode,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    if mode == "scenario_pool":
        scenarios = validate_scenario_pool(scenario_config)
        description.update(
            {
                "scenario_count": len(scenarios),
                "scenario_names": [scenario["name"] for scenario in scenarios],
                "scenario_weights": [scenario["weight"] for scenario in scenarios],
            }
        )
    return description


def _aircraft_state(value: Any, scenario_name: str, side: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{scenario_name}.{side} must be a {AIRCRAFT_STATE_SIZE}-value list")
    if len(value) != AIRCRAFT_STATE_SIZE:
        raise ValueError(
            f"{scenario_name}.{side} must contain [N, E, D, roll, pitch, heading, speed]"
        )
    state = [
        _finite_float(item, f"{scenario_name}.{side}[{index}]")
        for index, item in enumerate(value)
    ]
    if state[6] < 0.0:
        raise ValueError(f"{scenario_name}.{side} speed must be non-negative")
    return state


def _validate_randomization(
    value: Any,
    label: str,
    allowed_keys: set[str],
) -> None:
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - allowed_keys - {"enabled"}
    if unknown:
        raise ValueError(f"{label} has unsupported keys: {sorted(unknown)}")
    for key in allowed_keys:
        if key not in value:
            continue
        number = _finite_float(value[key], f"{label}.{key}")
        if number < 0.0:
            raise ValueError(f"{label}.{key} must be non-negative")


def _merged_randomization(
    base: Any,
    overrides: Any,
) -> dict[str, Any]:
    merged = dict(base) if isinstance(base, Mapping) else {}
    if isinstance(overrides, Mapping):
        merged.update(overrides)
    return merged


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _symmetric_offset(rng: np.random.Generator, limit: Any) -> float:
    magnitude = float(limit or 0.0)
    if magnitude <= 0.0:
        return 0.0
    return float(rng.uniform(-magnitude, magnitude))


def _wrap_signed_angle(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


__all__ = [
    "AIRCRAFT_STATE_SIZE",
    "SUPPORTED_TARGET_MODES",
    "apply_shared_randomization",
    "describe_initial_scenario",
    "randomize_aircraft_state",
    "sample_scenario_pool",
    "validate_scenario_pool",
]
