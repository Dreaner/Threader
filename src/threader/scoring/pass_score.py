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

from dataclasses import dataclass

from threader.models import PassOption, Player
from threader.scoring.completion import completion_probability
from threader.scoring.penetration import penetration_score
from threader.scoring.pressure import receiving_pressure
from threader.scoring.space import space_available
from threader.scoring.zone_value import zone_value


@dataclass(frozen=True)
class ScoringWeights:
    """Adjustable weights for the Pass Score formula.

    Used by sensitivity analysis to sweep parameters.
    Default values match the production formula (v1.1).
    """

    zone_amplifier: float = 1.5
    penetration_weight: float = 0.20
    space_weight: float = 0.001
    pressure_scaling: float = 0.20


# Default weights — used when no overrides are supplied.
DEFAULT_WEIGHTS = ScoringWeights()


def compute_pass_score(
    comp: float,
    zone: float,
    pressure: float,
    space: float,
    penetration: float,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> float:
    """Compute the final Pass Score (0–100) from pre-calculated dimensions.

    This is the pure formula, separated for reuse in sensitivity analysis
    without re-computing the individual dimension scores.
    """
    base_score = (
        comp * zone * weights.zone_amplifier
        + penetration * weights.penetration_weight
        + space * weights.space_weight
    )
    pressure_factor = 1.0 - (pressure / 10.0) * weights.pressure_scaling
    return max(0.0, base_score * pressure_factor * 100.0)


def score_pass_option(
    passer: Player,
    target: Player,
    defenders: list[Player],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    attack_direction: float = 1.0,
    weights: ScoringWeights | None = None,
) -> PassOption:
    """Score a single pass option from passer to target.

    Args:
        attack_direction: +1.0 if the passer's team attacks towards
            positive-x, -1.0 if towards negative-x.  Used to orient
            zone_value and penetration_score correctly.
        weights: Optional ScoringWeights override for sensitivity analysis.
            When None, uses DEFAULT_WEIGHTS.

    Returns a PassOption with all 5 dimension scores and the
    final Pass Score (0–100).
    """
    w = weights or DEFAULT_WEIGHTS

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

    final_score = compute_pass_score(comp, zone, pressure, space, penetration, w)

    return PassOption(
        target=target,
        pass_score=round(final_score, 1),
        completion_probability=round(comp, 3),
        zone_value=round(zone, 4),
        receiving_pressure=round(pressure, 1),
        space_available=round(space, 1),
        penetration_score=round(penetration, 2),
    )
