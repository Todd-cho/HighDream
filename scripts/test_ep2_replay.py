"""Replay EP2 on a live EP1 log and verify its isolated bank ownership."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from test_w108_lift_vector_replay import _provider, _state
from dogfight.ai.action_provider import ActionContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    provider = _provider("ep2")
    baseline = _provider("w111")
    override_frames = 0
    commit_flips = 0
    commit_flip_times: list[float] = []
    previous_sign = 0
    max_abs_action = 0.0
    roll_deltas: list[float] = []
    with args.log.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            context = ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=_state(row, "own"),
                target_state=_state(row, "enemy"),
            )
            result = provider.compute_action(context)
            baseline_result = baseline.compute_action(context)
            action = np.asarray(result.action)
            assert np.all(np.isfinite(action))
            max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))
            if result.info.get("explicit_bank_override_active", False):
                override_frames += 1
                roll_deltas.append(abs(float(action[0] - baseline_result.action[0])))
            sign = int(result.info.get("explicit_commit_sign", 0))
            if sign and previous_sign and sign != previous_sign:
                commit_flips += 1
                commit_flip_times.append(float(row["sim_time_s"]))
            if sign:
                previous_sign = sign
    assert override_frames > 0, "EP2 never owned reacquire bank"
    assert max_abs_action <= 1.00001, "EP2 emitted an out-of-bounds command"
    assert commit_flips <= 5, f"EP2 commit chattered: {commit_flips} flips"
    changed = sum(delta > 0.05 for delta in roll_deltas)
    print(
        f"EP2 override_frames={override_frames} commit_flips={commit_flips} "
        f"max_action={max_abs_action:.3f} mean_roll_delta={np.mean(roll_deltas):.3f} "
        f"changed_gt_005={changed}({100.0 * changed / len(roll_deltas):.1f}%) "
        f"flip_times={[round(value, 2) for value in commit_flip_times]}"
    )


if __name__ == "__main__":
    main()
