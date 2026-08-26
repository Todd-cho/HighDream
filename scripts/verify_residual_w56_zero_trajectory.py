"""Compare rule-only W56 and zero-residual W56 trajectories in JSBSim.

Both controllers update once per outer control step and hold their command
across ``step_ratio`` simulator ticks. This matches the live ~10 Hz command
path and isolates whether ResidualW56Env changes the frozen W56 baseline --
the "residual=0 equivalence test" both handoff documents require before any
training run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.rule_profiles import build_w56_controller
from dogfight.envs.residual_w56_env import ResidualW56Env


class ControlStepHoldProvider(ActionProvider):
    """Evaluate an inner provider once per outer env timestep."""

    def __init__(self, provider: ActionProvider):
        self.provider = provider
        self._timestep: int | None = None
        self._result: ActionResult | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self.provider.reset(context)
        self._timestep = None
        self._result = None

    def compute_action(self, context: ActionContext) -> ActionResult:
        timestep = int(context.info.get("timestep", -1))
        if self._result is None or timestep != self._timestep:
            self._result = self.provider.compute_action(context)
            self._timestep = timestep
        return self._result

    @property
    def last_result(self) -> ActionResult | None:
        return self._result


def _assert_close(name: str, left, right, step: int, atol: float) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"step {step}: {name} shape differs: {a.shape} != {b.shape}")
    delta = np.abs(a - b)
    max_delta = float(np.max(delta)) if delta.size else 0.0
    if not np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True):
        index = tuple(int(i) for i in np.unravel_index(int(np.nanargmax(delta)), delta.shape))
        raise AssertionError(
            f"step {step}: {name} differs at {index}: "
            f"rule={a[index]!r}, residual={b[index]!r}, "
            f"delta={delta[index]:.12g}, atol={atol}"
        )
    return max_delta


def run(seed: int, steps: int, atol: float) -> None:
    config = {
        "target_mode": "fixed",
        "episode_step_limit": steps + 1,
        "action_rate_limit": None,
        "safety_override_enabled": False,
    }
    held_w56 = ControlStepHoldProvider(build_w56_controller())
    rule_env = DogFightWrapper(
        env_config=config,
        ownship_action_provider=held_w56,
    )
    residual_env = ResidualW56Env(env_config=config)

    maxima = {"ownship_state": 0.0, "target_state": 0.0, "applied_action": 0.0}
    completed = 0
    try:
        rule_env.reset(seed=seed)
        residual_env.reset(seed=seed)
        maxima["ownship_state"] = max(
            maxima["ownship_state"],
            _assert_close(
                "initial ownship state",
                rule_env.get_ownship_state(),
                residual_env.get_ownship_state(),
                0,
                atol,
            ),
        )
        maxima["target_state"] = max(
            maxima["target_state"],
            _assert_close(
                "initial target state",
                rule_env.get_target_state(),
                residual_env.get_target_state(),
                0,
                atol,
            ),
        )

        for step in range(1, steps + 1):
            _, rule_reward, rule_term, rule_trunc, rule_info = rule_env.step(
                np.zeros(4, dtype=np.float32)
            )
            _, residual_reward, residual_term, residual_trunc, residual_info = residual_env.step(
                np.zeros(3, dtype=np.float32)
            )

            maxima["ownship_state"] = max(
                maxima["ownship_state"],
                _assert_close(
                    "ownship state",
                    rule_env.get_ownship_state(),
                    residual_env.get_ownship_state(),
                    step,
                    atol,
                ),
            )
            maxima["target_state"] = max(
                maxima["target_state"],
                _assert_close(
                    "target state",
                    rule_env.get_target_state(),
                    residual_env.get_target_state(),
                    step,
                    atol,
                ),
            )
            maxima["applied_action"] = max(
                maxima["applied_action"],
                _assert_close(
                    "applied action",
                    rule_env.get_ownship_action(),
                    residual_env.get_ownship_action(),
                    step,
                    atol,
                ),
            )
            if held_w56.last_result is None:
                raise AssertionError(f"step {step}: rule-only provider produced no action")
            _assert_close(
                "W56 command",
                held_w56.last_result.action,
                residual_info["rule_action"],
                step,
                atol,
            )
            _assert_close(
                "zero-residual combined action",
                residual_info["rule_action"],
                residual_info["combined_action"],
                step,
                atol,
            )
            # Reward comparison uses the base rule-only reward, not the
            # residual env's post-penalty reward: at zero residual the
            # magnitude/delta penalty is exactly 0 (see
            # ResidualW56Env.step()), so the two must still match exactly.
            if abs(float(rule_reward) - float(residual_reward)) > atol:
                raise AssertionError(
                    f"step {step}: reward differs: {rule_reward} != {residual_reward}"
                )
            if (rule_term, rule_trunc) != (residual_term, residual_trunc):
                raise AssertionError(
                    f"step {step}: termination differs: "
                    f"rule={(rule_term, rule_trunc)}, residual={(residual_term, residual_trunc)}"
                )
            for key in ("end_condition", "outcome"):
                if rule_info.get(key) != residual_info.get(key):
                    raise AssertionError(
                        f"step {step}: {key} differs: "
                        f"{rule_info.get(key)!r} != {residual_info.get(key)!r}"
                    )
            completed = step
            if rule_term or rule_trunc:
                break
    finally:
        rule_env.close()
        residual_env.close()

    print("PASS: zero-residual W56 trajectory matches rule-only W56")
    print(f"  seed: {seed}")
    print(f"  compared control steps: {completed}")
    print(f"  ownship state max |delta|: {maxima['ownship_state']:.12g}")
    print(f"  target state max |delta|:  {maxima['target_state']:.12g}")
    print(f"  applied action max |delta|:{maxima['applied_action']:.12g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()
    run(args.seed, args.steps, args.atol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
