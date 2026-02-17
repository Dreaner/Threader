"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: pass_score.py
Description: 
    Pass Score — the master scoring formula combining all 5 dimensions.
    Pass Score = (
        completion_probability × zone_value × 1.5 +  # Expected value (amplified)
        penetration_score × 0.20 +                   # Penetration bonus/drag
        space_available × 0.001                      # Space bonus (capped at 15m)
    ) × (1.0 − pressure/10 × 0.20)                  # Pressure as multiplier
      × 100
    Range: 0–100
    
    Key changes (v1.1):
    - zone_value amplified ×1.5 to widen gap between attacking/defensive positions
    - Pressure switched from additive penalty to multiplicative scaling
    - Penetration now returns negative for backward passes (mild drag)
    - Space capped at 15m upstream
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
    attack_direction: float = 1.0,
) -> PassOption:
    """Score a single pass option from passer to target.

    Args:
        attack_direction: +1.0 if the passer's team attacks towards
            positive-x, -1.0 if towards negative-x.  Used to orient
            zone_value and penetration_score correctly.

    Returns a PassOption with all 5 dimension scores and the
    final Pass Score (0–100).
    """
    comp = completion_probability(passer, target, defenders)
    zone = zone_value(
        target.x, target.y, pitch_length, pitch_width,
        attack_direction=attack_direction,
    )
    pressure = receiving_pressure(target, defenders)
    space = space_available(target, defenders)
    penetration = penetration_score(
        passer, target, defenders,
        attack_direction=attack_direction,
    )

    # Combine expected value + bonus terms
    base_score = (
        comp * zone * 1.5           # Expected value (amplified)
        + penetration * 0.20        # Penetration bonus/drag
        + space * 0.001             # Space bonus
    )

    # Pressure as a multiplicative dampener (max 20% reduction)
    pressure_factor = 1.0 - (pressure / 10.0) * 0.20
    raw_score = base_score * pressure_factor

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
