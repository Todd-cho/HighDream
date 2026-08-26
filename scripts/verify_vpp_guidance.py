"""Deterministic checks for the constant-cost W101 VPP/VPG/PNG equations."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dogfight.ai.vpp_guidance import compute_vpp_guidance


def main() -> None:
    own = np.array([0.0, 0.0, -3048.0])
    target = np.array([2000.0, 0.0, -3048.0])
    own_v = np.array([200.0, 0.0, 0.0])
    target_v = np.array([0.0, 200.0, 0.0])
    result = compute_vpp_guidance(
        own, target, own_v, target_v,
        ata_deg=5.0, threat_ata_deg=70.0, distance_m=2000.0,
        previous_blend=0.5, dt=0.1,
    )
    assert np.all(np.isfinite(result.point))
    assert np.isfinite(result.turn_rate_degps)
    assert -16.0 <= result.turn_rate_degps <= 16.0
    assert -24.0 <= result.gamma_deg <= 24.0

    mutual = compute_vpp_guidance(
        own, target, own_v, -own_v,
        ata_deg=5.0, threat_ata_deg=2.0, distance_m=1000.0,
        previous_blend=0.5, dt=0.1,
    )
    assert mutual.mode == "lag"
    assert mutual.blend < 0.5
    assert not np.allclose(mutual.point, target)
    print("PASS VPP/VPG/PNG guidance finite, bounded, and mutual-pass lag selected")


if __name__ == "__main__":
    main()
