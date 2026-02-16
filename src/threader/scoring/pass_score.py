"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: pass_score.py
Description: 
    Pass Score — the master scoring formula combining all 5 dimensions.
    Pass Score = (
        completion_probability × zone_value +    # Expected value
        penetration_score × 0.20 +               # Penetration bonus
        space_available × 0.001 -                # Space bonus
        (receiving_pressure / 10) × 0.15         # Pressure penalty
    ) × 100
    Range: 0–100
"""

from __future__ import annotations

from threader.models import PassOption, Player
from threader.scoring.completion import completion_probability
from threader.scoring.penetration import penetration_score
from threader.scoring.pressure import receiving_pressure
from threader.scoring.space import space_available
from threader.scoring.zone_value import zone_value


def score_pass_option(
    passer: Player,
    target: Player,
    defenders: list[Player],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> PassOption:
    """Score a single pass option from passer to target.

    Returns a PassOption with all 5 dimension scores and the
    final Pass Score (0–100).
    """
    comp = completion_probability(passer, target, defenders)
    zone = zone_value(target.x, target.y, pitch_length, pitch_width)
    pressure = receiving_pressure(target, defenders)
    space = space_available(target, defenders)
    penetration = penetration_score(passer, target, defenders)

    raw_score = (
        comp * zone
        + penetration * 0.20
        + space * 0.001
        - (pressure / 10.0) * 0.15
    )

    # Scale to 0–100, clamp to non-negative
    final_score = max(0.0, raw_score * 100.0)

    return PassOption(
        target=target,
        pass_score=round(final_score, 1),
        completion_probability=round(comp, 3),
        zone_value=round(zone, 4),
        receiving_pressure=round(pressure, 1),
        space_available=round(space, 1),
        penetration_score=round(penetration, 2),
    )
