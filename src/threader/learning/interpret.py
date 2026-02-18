"""
Interpret learned models → extract recommended ScoringWeights.

Maps linear regression coefficients back to the Pass Score formula
structure to produce concrete weight updates.
"""

from __future__ import annotations

import numpy as np

from threader.scoring.pass_score import ScoringWeights

from .models import ModelResult


def interpret_linear_coefficients(
    result: ModelResult,
) -> dict:
    """Interpret a linear model's coefficients relative to Pass Score formula.

    The current formula is:
      score = (comp × zone × zone_amp + pen × pen_w + space × space_w) × (1 - pres/10 × pres_k) × 100

    A linear regression learns:
      y = a₀ + a₁·comp + a₂·zone + a₃·pressure + a₄·space + a₅·penetration

    We map the learned coefficients to formula-compatible weights.
    """
    if result.coefficients is None:
        return {"error": "no coefficients available"}

    coeffs = result.coefficients
    c = {k: coeffs.get(k, 0.0) for k in ["completion", "zone_value", "pressure", "space", "penetration"]}
    intercept = coeffs.get("intercept", 0.0)

    # Relative importance (absolute value of coefficients, normalized)
    abs_sum = sum(abs(v) for v in c.values())
    if abs_sum == 0:
        return {"error": "all coefficients are zero"}

    rel_importance = {k: round(abs(v) / abs_sum, 4) for k, v in c.items()}

    # Sign analysis
    signs = {k: ("+" if v > 0 else "-") for k, v in c.items()}

    return {
        "raw_coefficients": {k: round(v, 6) for k, v in c.items()},
        "intercept": round(intercept, 6),
        "relative_importance": rel_importance,
        "signs": signs,
        "model_name": result.name,
        "test_r2": result.test_r2,
    }


def interpret_interactions(result: ModelResult) -> dict:
    """Interpret interaction model to find key synergies."""
    if result.coefficients is None:
        return {"error": "no coefficients available"}

    coeffs = result.coefficients
    # Separate main effects from interactions
    main_effects = {}
    interactions = {}
    for name, val in coeffs.items():
        if name == "intercept":
            continue
        if " " in name:
            interactions[name] = round(val, 6)
        else:
            main_effects[name] = round(val, 6)

    # Sort interactions by absolute magnitude
    sorted_interactions = dict(
        sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        "main_effects": main_effects,
        "top_interactions": dict(list(sorted_interactions.items())[:5]),
        "model_name": result.name,
        "test_r2": result.test_r2,
    }


def interpret_xgboost(result: ModelResult) -> dict:
    """Interpret XGBoost feature importances."""
    if result.feature_importance is None:
        return {"error": "no feature importance available"}

    imp = result.feature_importance
    if "error" in imp:
        return imp

    sorted_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

    return {
        "feature_importance": {k: round(v, 4) for k, v in sorted_imp.items()},
        "model_name": result.name,
        "test_r2": result.test_r2,
    }


def suggest_weights(
    linear_result: ModelResult,
    interaction_result: ModelResult,
    xgb_result: ModelResult,
) -> dict:
    """Synthesize interpretations into a concrete ScoringWeights recommendation.

    Strategy:
    1. Use linear coefficients for direction & magnitude
    2. Use interaction model to validate comp × zone synergy
    3. Use XGBoost importance for overall ranking

    Returns dict with suggested weights and rationale.
    """
    linear = interpret_linear_coefficients(linear_result)
    interactions = interpret_interactions(interaction_result)
    xgb = interpret_xgboost(xgb_result)

    if "error" in linear:
        return {"error": "cannot suggest weights — linear model failed"}

    coeffs = linear["raw_coefficients"]
    rel_imp = linear["relative_importance"]

    # Current defaults for reference
    current = ScoringWeights()

    # ── Build suggestion ─────────────────────────────────────────────────
    # The formula has specific structural constraints:
    #   zone_amplifier: scales comp × zone product (amplifies zone value)
    #   penetration_weight: additive weight for penetration
    #   space_weight: additive weight for space
    #   pressure_scaling: multiplicative penalty [0, 1]

    # Zone amplifier: if zone importance is high relative to others, increase
    zone_imp = rel_imp.get("zone_value", 0)
    comp_imp = rel_imp.get("completion", 0)
    pen_imp = rel_imp.get("penetration", 0)
    space_imp = rel_imp.get("space", 0)

    # Scale penetration_weight based on learned importance relative to zone
    # Current: pen=0.20, which means penetration contributes 0.20 × pen_score
    # vs zone contribution = comp × zone × 1.5
    # Average comp ≈ 0.7, zone ≈ 0.1 → zone_contrib ≈ 0.105
    # pen contributes ≈ 0.20 × 0.5 = 0.10 (similar magnitude)
    # So the ratio pen_imp/zone_imp maps roughly to pen_w/zone_amp

    if zone_imp > 0:
        pen_ratio = pen_imp / zone_imp
        suggested_pen_w = round(current.zone_amplifier * pen_ratio * 0.15, 4)
        suggested_pen_w = max(0.05, min(0.50, suggested_pen_w))
    else:
        suggested_pen_w = current.penetration_weight

    if zone_imp > 0:
        space_ratio = space_imp / zone_imp
        suggested_space_w = round(current.space_weight * max(0.2, min(5.0, space_ratio)), 6)
    else:
        suggested_space_w = current.space_weight

    # Zone amplifier: if zone is the dominant signal, bump it up
    if zone_imp > comp_imp and zone_imp > pen_imp:
        suggested_zone_amp = round(current.zone_amplifier * 1.2, 2)
    elif zone_imp < comp_imp * 0.5:
        suggested_zone_amp = round(current.zone_amplifier * 0.8, 2)
    else:
        suggested_zone_amp = current.zone_amplifier

    # Pressure: if negative coefficient (more pressure → lower offensive value), keep as penalty
    pressure_sign = coeffs.get("pressure", 0)
    if pressure_sign < 0:
        # Higher pressure hurts → keep/increase scaling
        pres_imp = rel_imp.get("pressure", 0)
        if pres_imp > 0.2:
            suggested_pres_k = round(current.pressure_scaling * 1.3, 3)
        else:
            suggested_pres_k = current.pressure_scaling
    else:
        # Surprising: higher pressure → higher offensive value
        # This might mean risky passes into pressure yield more xT
        # Reduce pressure penalty
        suggested_pres_k = round(current.pressure_scaling * 0.5, 3)

    suggested_pres_k = max(0.05, min(0.40, suggested_pres_k))

    suggested = ScoringWeights(
        zone_amplifier=suggested_zone_amp,
        penetration_weight=suggested_pen_w,
        space_weight=suggested_space_w,
        pressure_scaling=suggested_pres_k,
    )

    return {
        "current_weights": {
            "zone_amplifier": current.zone_amplifier,
            "penetration_weight": current.penetration_weight,
            "space_weight": current.space_weight,
            "pressure_scaling": current.pressure_scaling,
        },
        "suggested_weights": {
            "zone_amplifier": suggested.zone_amplifier,
            "penetration_weight": suggested.penetration_weight,
            "space_weight": suggested.space_weight,
            "pressure_scaling": suggested.pressure_scaling,
        },
        "rationale": {
            "linear_importance": rel_imp,
            "linear_signs": linear["signs"],
            "xgb_importance": xgb.get("feature_importance", {}),
            "top_interactions": interactions.get("top_interactions", {}),
        },
        "model_metrics": {
            "linear_test_r2": linear.get("test_r2"),
            "interaction_test_r2": interactions.get("test_r2"),
            "xgb_test_r2": xgb.get("test_r2"),
        },
        "suggested_scoring_weights": suggested,
    }
