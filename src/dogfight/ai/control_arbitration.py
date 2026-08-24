"""Control-ownership diagnostics and the contract for the explicit-phase line.

No command is modified in EP1.  The proposal/ownership data exposes where the
legacy monolith has competing active features, so EP2 can move one axis at a
time behind this common arbitration contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ControlProposal:
    owner: str
    priority: int
    authority: float


def _active(info: Mapping[str, Any], names: tuple[str, ...]) -> list[str]:
    return [name for name in names if bool(info.get(name, False))]


def diagnose_control_ownership(info: Mapping[str, Any], action) -> dict[str, Any]:
    bank_features = _active(info, (
        "headon_deconflict_active", "defensive_escape_active",
        "formula_zem_defense_active", "planner3d_active",
        "predictive_override_active", "integrated_rate_bank_active",
        "formula_rate_bank_active", "turn_match_active",
        "terminal_track_active", "rear_rate_match_active",
        "rate_deficit_overbank_active", "spiral_recovery_active",
        "lift_vector_active",
    ))
    pitch_features = _active(info, (
        "opening_pull_boost_active", "terminal_vertical_unload_active",
        "vertical_alignment_active", "vertical_follow_active",
        "energy_gamma_guard_active", "horizontal_energy_hold_active",
        "physical_pull_limited", "pitch_soft_limited",
    ))
    throttle_features = _active(info, (
        "energy_recovery_active", "overshoot_control_active",
        "high_bank_speed_control_active", "lag_energy_preserve_active",
    ))

    if bool(info.get("lift_vector_active", False)):
        bank_owner = "lift_vector_overlay"
    elif bool(info.get("formula_zem_defense_active", False)):
        bank_owner = "zem_defense_rate"
    elif bool(info.get("planner3d_active", False)):
        bank_owner = "planner3d_rate"
    elif bool(info.get("predictive_override_active", False)):
        bank_owner = "predictive_rate"
    elif bool(info.get("terminal_track_active", False)):
        bank_owner = "terminal_rate"
    elif bool(info.get("turn_match_active", False)):
        bank_owner = "turn_match_rate"
    elif bool(info.get("integrated_rate_bank_active", False)):
        bank_owner = "integrated_rate"
    elif bool(info.get("headon_deconflict_active", False)):
        bank_owner = "headon_deconflict"
    else:
        bank_owner = "course_geometry"

    pitch_owner = pitch_features[-1] if pitch_features else "gamma_rate_loop"
    throttle_owner = throttle_features[-1] if throttle_features else "energy_schedule"
    conflict_count = max(len(bank_features), len(pitch_features), len(throttle_features))
    saturated = any(abs(float(action[index])) >= 0.995 for index in (0, 1))
    return {
        "bank_owner": bank_owner,
        "pitch_owner": pitch_owner,
        "throttle_owner": throttle_owner,
        "bank_authority": 1.0,
        "pitch_authority": 1.0,
        "throttle_authority": 1.0,
        "bank_proposal_count": len(bank_features) or 1,
        "pitch_proposal_count": len(pitch_features) or 1,
        "throttle_proposal_count": len(throttle_features) or 1,
        "proposal_conflict": conflict_count > 1,
        "proposal_saturated": saturated,
        "active_bank_proposals": "|".join(bank_features),
        "active_pitch_proposals": "|".join(pitch_features),
        "active_throttle_proposals": "|".join(throttle_features),
    }
