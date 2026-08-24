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
    override_frames = 0
    commit_flips = 0
    previous_sign = 0
    max_abs_action = 0.0
    with args.log.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result = provider.compute_action(ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=_state(row, "own"),
                target_state=_state(row, "enemy"),
            ))
            action = np.asarray(result.action)
            assert np.all(np.isfinite(action))
            max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))
            if result.info.get("explicit_bank_override_active", False):
                override_frames += 1
            sign = int(result.info.get("explicit_commit_sign", 0))
            if sign and previous_sign and sign != previous_sign:
                commit_flips += 1
            if sign:
                previous_sign = sign
    assert override_frames > 0, "EP2 never owned reacquire bank"
    assert max_abs_action <= 1.00001, "EP2 emitted an out-of-bounds command"
    assert commit_flips <= 2, f"EP2 commit chattered: {commit_flips} flips"
    print(
        f"EP2 override_frames={override_frames} commit_flips={commit_flips} "
        f"max_action={max_abs_action:.3f}"
    )


if __name__ == "__main__":
    main()
