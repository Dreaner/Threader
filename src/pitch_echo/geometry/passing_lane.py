"""
Project: PitchEcho
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: passing_lane.py
Description: 
    Passing lane obstruction detection.
    Determines whether defenders are blocking the passing lane between
    a passer and a potential receiver, and quantifies the degree of blocking.
"""

from __future__ import annotations

from pitch_echo.geometry.distance import (
    point_to_segment_distance,
    projection_parameter,
)
from pitch_echo.core.models import Player


def is_in_passing_lane(
    defender: Player,
    passer: Player,
    receiver: Player,
    *,
    t_min: float = 0.1,
    t_max: float = 0.9,
) -> bool:
    """Check whether a defender is within the effective passing lane.

    We use t ∈ [0.1, 0.9] to avoid noise near the passer (who naturally
    has nearby teammates/opponents) and near the receiver.
    """
    t = projection_parameter(
        defender.x, defender.y,
        passer.x, passer.y,
        receiver.x, receiver.y,
    )
    return t_min < t < t_max


def passing_lane_blocking(
    passer: Player,
    receiver: Player,
    defenders: list[Player],
) -> float:
    """Calculate total blocking factor from defenders in the passing lane.

    Returns a value in [0, 0.8] representing cumulative obstruction.
    Higher = more blocked.

    Blocking thresholds (from CLAUDE.md):
      < 1.5m  →  0.4 (severe)
      < 3.0m  →  0.2 (moderate)
      < 5.0m  →  0.1 (minor)
    """
    blocking = 0.0

    for defender in defenders:
        if not is_in_passing_lane(defender, passer, receiver):
            continue

        dist = point_to_segment_distance(
            defender.x, defender.y,
            passer.x, passer.y,
            receiver.x, receiver.y,
        )

        if dist < 1.5:
            blocking += 0.4
        elif dist < 3.0:
            blocking += 0.2
        elif dist < 5.0:
            blocking += 0.1

    return min(0.8, blocking)
