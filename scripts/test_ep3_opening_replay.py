"""Verify EP3's opening pull is active, bounded, and softer than W112."""

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
    ep3 = _provider("ep3")
    w111 = _provider("w111")
    active = changed = 0
    max_added_pull = 0.0
    with args.log.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            context = ActionContext(
                sim=None, opponent_sim=None,
                ownship_state=_state(row, "own"), target_state=_state(row, "enemy"),
            )
            result = ep3.compute_action(context)
            baseline = w111.compute_action(context)
            assert np.all(np.isfinite(result.action))
            assert np.max(np.abs(result.action)) <= 1.00001
            if result.info.get("explicit_opening_pull_active", False):
                active += 1
                delta = float(baseline.action[1] - result.action[1])
                if delta > 1e-6:
                    changed += 1
                    max_added_pull = max(max_added_pull, delta)
    assert active > 0, "EP3 opening pull never activated"
    assert changed > 0, "EP3 never added opening pull"
    assert max_added_pull < 0.5, "EP3 added a W112-like excessive pull step"
    print(f"EP3 active={active} changed={changed} max_added_pull={max_added_pull:.3f}")


if __name__ == "__main__":
    main()
