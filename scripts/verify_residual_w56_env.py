"""Lightweight unit checks for W56 residual action composition and the
addendum's binary activation gate (RL_TRAINING_ADDENDUM_W56_KO section 4)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.envs.residual_w56_env import (
    DEFAULT_RESIDUAL_W56_CONFIG,
    ResidualW56Env,
    compose_w56_residual_action,
)


def gate() -> ResidualW56Env:
    """Construct only the state needed by the pure gate/composition logic."""
    env = object.__new__(ResidualW56Env)
    env._residual_cfg = dict(DEFAULT_RESIDUAL_W56_CONFIG)
    return env


def assert_gate(env: ResidualW56Env, ata: float, distance: float, threat_ata: float, expected: bool, label: str) -> None:
    actual = env._gate_active({"ata": ata, "distance": distance, "threat_ata": threat_ata})
    if actual != expected:
        raise AssertionError(f"{label}: expected gate_active={expected}, got {actual}")
    print(f"PASS: {label:<40} gate_active={actual}")


def main() -> int:
    env = gate()
    env._danger_altitude_active = lambda: False  # isolate the ata/distance/threat_ata window

    assert_gate(env, 10.0, 1500.0, 90.0, True, "in-window, favorable threat_ata")
    assert_gate(env, 20.0, 3000.0, 40.0, True, "exact boundary, all thresholds inclusive")
    assert_gate(env, 20.01, 1500.0, 90.0, False, "ATA above gate window")
    assert_gate(env, 10.0, 3000.01, 90.0, False, "range above gate window")
    assert_gate(env, 0.76, 706.0, 39.0, False, "W57-style mutual head-on (addendum finding)")

    env._danger_altitude_active = lambda: True
    assert_gate(env, 10.0, 1500.0, 90.0, False, "danger altitude overrides an otherwise-good window")

    rule = np.array([0.25, -0.30, 0.20, 0.73], dtype=np.float32)
    raw = np.array([1.0, -1.0, 1.0], dtype=np.float32)

    inactive = compose_w56_residual_action(rule, raw, False, 0.15, 0.20, 0.15)
    np.testing.assert_array_equal(inactive, rule)
    print("PASS: gate inactive -> combined action equals rule action exactly")

    active = compose_w56_residual_action(rule, raw, True, 0.15, 0.20, 0.15)
    np.testing.assert_allclose(active, [0.40, -0.50, 0.20, 0.88], rtol=0.0, atol=1e-7)
    if active[2] != rule[2]:
        raise AssertionError("residual must never control yaw")
    print(f"PASS: gate active -> combined action {active.tolist()}")

    saturated = compose_w56_residual_action(
        np.array([0.95, -0.95, -0.4, 0.90], dtype=np.float32),
        raw,
        True,
        0.15,
        0.20,
        0.15,
    )
    np.testing.assert_allclose(saturated, [1.0, -1.0, -0.4, 1.0], rtol=0.0, atol=1e-7)
    print("PASS: roll/pitch/throttle clip with yaw untouched")

    zero_residual = compose_w56_residual_action(rule, np.zeros(3, dtype=np.float32), True, 0.15, 0.20, 0.15)
    np.testing.assert_array_equal(zero_residual, rule)
    print("PASS: zero residual == rule action even when gate is active")

    print("PASS: all W56 residual gate and composition checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
