"""Explicit engagement-phase classification for the post-W113 controller line.

EP1 is deliberately observational: it classifies the legacy W111 result without
changing its command.  This gives the next controller revision an evidence-based
phase boundary instead of adding another implicit flag to the monolith.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class EngagementPhase(str, Enum):
    OPENING_MERGE = "opening_merge"
    HEADON_NEUTRAL = "headon_neutral"
    DEFENSIVE = "defensive"
    REACQUIRE = "reacquire"
    TRACKING_OFFENSE = "tracking_offense"
    TERMINAL_ATTACK = "terminal_attack"
    SAFETY = "safety"


def classify_phase(info: Mapping[str, Any]) -> EngagementPhase:
    """Map the legacy flag set to one mutually-exclusive engagement phase."""
    state = str(info.get("state", ""))
    first_merge_passed = bool(info.get("first_merge_passed", False))

    if bool(info.get("safety_override_active", False)):
        return EngagementPhase.SAFETY
    if state == "defensive" or bool(info.get("defensive_escape_active", False)):
        return EngagementPhase.DEFENSIVE
    if not first_merge_passed:
        return EngagementPhase.OPENING_MERGE
    if bool(info.get("terminal_track_active", False)):
        return EngagementPhase.TERMINAL_ATTACK
    if state in {"weapons_track", "offensive_track"}:
        return EngagementPhase.TRACKING_OFFENSE
    if bool(info.get("post_defense_conversion_active", False)) or state == "reacquire":
        return EngagementPhase.REACQUIRE
    return EngagementPhase.HEADON_NEUTRAL
