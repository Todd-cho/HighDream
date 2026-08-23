"""Fixed-command pulse-test ActionProvider for live system identification
(2026-08-21, step 3 of the post-pursuit_controller pivot plan).

Every control law tuned tonight (tactical_wrapper.py W1-W14, pursuit_controller.py
P0-P7) was designed against ASSUMED response characteristics (gain, latency,
rate limits) inferred from JSBSim behavior or from re-reading live logs after
the fact. This provider instead sends a scripted, open-loop sequence of KNOWN
fixed commands (no RL, no geometry, no feedback of any kind) straight to the
live Unreal/DogFightViewer server, so the resulting --log-csv trajectory (which
already records own_roll_deg/own_pitch_deg/own_yaw_deg every frame via
ProviderCommandPolicy) is a clean step-response measurement of the REAL live
plant -- not a JSBSim approximation of it. That measured gain/latency/rate is
what should size a from-scratch minimal controller's gains, instead of
guessing and re-tuning against symptoms after the fact.

Default sequence (roll step-response, ~25s total):
  0-5s   neutral (baseline / trim)
  5-10s  roll_cmd=+0.5 (step up)
  10-15s roll_cmd=0.0 (step back to neutral -- decay/settle)
  15-20s roll_cmd=-0.5 (step down -- symmetry check)
  20-25s roll_cmd=0.0 (final settle)
Pass --pulse-sequence to test pitch/yaw/throttle instead, or a custom timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex

# (duration_s, roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd)
DEFAULT_ROLL_PULSE_SEQUENCE: list[tuple[float, float, float, float, float]] = [
    (5.0, 0.0, 0.0, 0.0, 0.7),
    (5.0, 0.5, 0.0, 0.0, 0.7),
    (5.0, 0.0, 0.0, 0.0, 0.7),
    (5.0, -0.5, 0.0, 0.0, 0.7),
    (5.0, 0.0, 0.0, 0.0, 0.7),
]

PITCH_PULSE_SEQUENCE: list[tuple[float, float, float, float, float]] = [
    (5.0, 0.0, 0.0, 0.0, 0.7),
    (5.0, 0.0, 0.5, 0.0, 0.7),
    (5.0, 0.0, 0.0, 0.0, 0.7),
    (5.0, 0.0, -0.5, 0.0, 0.7),
    (5.0, 0.0, 0.0, 0.0, 0.7),
]

# C. Yaw/rudder authority -- W16 showed that increasing turn pull preserved
# altitude but barely changed yaw rate while bank was already 70+ degrees.
# Keep these pulses deliberately small and short: the purpose is to measure
# direct yaw authority and roll coupling, not to fly a combat trajectory.
YAW_PULSE_SEQUENCE: list[tuple[float, float, float, float, float]] = [
    (3.0, 0.0, 0.0, 0.0, 0.7),
    (2.0, 0.0, 0.0, 0.2, 0.7),
    (3.0, 0.0, 0.0, 0.0, 0.7),
    (2.0, 0.0, 0.0, -0.2, 0.7),
    (3.0, 0.0, 0.0, 0.0, 0.7),
]

# Added 2026-08-21 (second pass over run0044/run0045, user critique): the
# 5s +-0.5 pulses above were strong enough to be useful for a first rough
# rate/bias measurement, but too strong for clean follow-up analysis --
# run0044's roll pulse alone put the aircraft through a full inverted
# rotation and left ~50s of "settling" data contaminated by that
# disturbance. These two sequences are deliberately short (1-3s legs) and
# small-magnitude, aimed at two specific follow-up questions the first
# pass couldn't answer:

# A. Pitch trim sweep -- pitch_cmd=0/throttle=0.7 was shown to NOT be level
# (own_pitch drifted -6.9deg over the first quiet 5s in both prior logs).
# Step through small nose-up commands and look for the one where vertical
# speed crosses zero -- that's the actual level_pitch_trim value (W1Config's
# level_pitch_trim=-0.10 is only a first estimate from the drift rate, not
# measured this way yet).
PITCH_TRIM_SWEEP_SEQUENCE: list[tuple[float, float, float, float, float]] = [
    (3.0, 0.0, 0.00, 0.0, 0.7),
    (3.0, 0.0, -0.05, 0.0, 0.7),
    (3.0, 0.0, -0.10, 0.0, 0.7),
    (3.0, 0.0, -0.15, 0.0, 0.7),
    (3.0, 0.0, -0.20, 0.0, 0.7),
    (3.0, 0.0, 0.00, 0.0, 0.7),
]

# B. Roll inertia/braking -- a short step in one direction, then an
# opposite-sign brake pulse held long enough to see the roll rate actually
# arrest (not just start reversing), repeated for the other direction. Read
# off: input delay (frames between command change and rate change), peak
# roll rate reached from just a 1s step (vs. the ~85deg/s seen from a full
# 5s pulse), how long the brake takes to zero the rate, and what bank angle
# it stops at.
ROLL_INERTIA_SEQUENCE: list[tuple[float, float, float, float, float]] = [
    (2.0, 0.0, 0.0, 0.0, 0.7),
    (1.0, 0.3, 0.0, 0.0, 0.7),
    (3.0, -0.3, 0.0, 0.0, 0.7),
    (2.0, 0.0, 0.0, 0.0, 0.7),
    (1.0, -0.3, 0.0, 0.0, 0.7),
    (3.0, 0.3, 0.0, 0.0, 0.7),
    (2.0, 0.0, 0.0, 0.0, 0.7),
]


@dataclass
class PulseTestConfig:
    sequence: list[tuple[float, float, float, float, float]] = field(
        default_factory=lambda: list(DEFAULT_ROLL_PULSE_SEQUENCE)
    )
    loop: bool = False  # repeat the sequence forever instead of holding the last step


class PulseTestActionProvider(ActionProvider):
    """Open-loop scripted command sequence. No feedback, no RL, no geometry --
    pure system identification signal generator."""

    def __init__(self, config: PulseTestConfig | None = None):
        self.cfg = config or PulseTestConfig()
        self._start_sim_time: float | None = None
        self._cumulative_durations = np.cumsum([step[0] for step in self.cfg.sequence])
        self._total_duration = float(self._cumulative_durations[-1]) if len(self.cfg.sequence) else 0.0

    def reset(self, context: ActionContext | None = None) -> None:
        self._start_sim_time = None

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = context.ownship_state
        now = float(own[StateIndex.SIM_TIME]) if own is not None else 0.0
        if self._start_sim_time is None:
            self._start_sim_time = now
        elapsed = now - self._start_sim_time

        if self.cfg.loop and self._total_duration > 0:
            elapsed = elapsed % self._total_duration

        step_index = int(np.searchsorted(self._cumulative_durations, elapsed, side="right"))
        step_index = min(step_index, len(self.cfg.sequence) - 1)
        _, roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = self.cfg.sequence[step_index]

        action = np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)
        return ActionResult(
            action=action,
            source=f"pulse_test[step{step_index}]",
            confidence=1.0,
            info={"elapsed_s": elapsed, "step_index": step_index},
        )
