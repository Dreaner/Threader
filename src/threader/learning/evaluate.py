"""
Evaluate learned models by re-scoring all passes and comparing to
the current Pass Score formula.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from threader.scoring.pass_score import DEFAULT_WEIGHTS, ScoringWeights, compute_pass_score

if TYPE_CHECKING:
    from threader.validation.collector import ValidatedPass


def rescore_with_weights(
    records: list[ValidatedPass],
    weights: ScoringWeights,
) -> np.ndarray:
    """Re-score all passes with given weights. Returns array of scores."""
    scores = np.empty(len(records), dtype=np.float64)
    for i, r in enumerate(records):
        scores[i] = compute_pass_score(
            comp=r.actual_target_completion,
            zone=r.actual_target_zone,
            pressure=r.actual_target_pressure,
            space=r.actual_target_space,
            penetration=r.actual_target_penetration,
            weights=weights,
        )
    return scores


def evaluate_weights(
    records: list[ValidatedPass],
    weights: ScoringWeights,
    label: str = "candidate",
) -> dict:
    """Evaluate a set of weights against offensive-value ground truth.

    Returns dict with Spearman ρ for ΔxT, AUC for lines-broken,
    and comparison with DEFAULT_WEIGHTS.
    """
    scores = rescore_with_weights(records, weights)
    delta_xts = np.array([r.delta_xt for r in records])

    # ΔxT Spearman
    rho, rho_p = stats.spearmanr(scores, delta_xts)

    # Lines-broken AUC
    has_lb = np.array([1.0 if r.pff_lines_broken_count >= 1 else 0.0 for r in records])
    from sklearn.metrics import roc_auc_score
    try:
        lb_auc = roc_auc_score(has_lb, scores)
    except ValueError:
        lb_auc = 0.5

    # Compare with default weights
    default_scores = rescore_with_weights(records, DEFAULT_WEIGHTS)
    rho_def, _ = stats.spearmanr(default_scores, delta_xts)
    try:
        lb_auc_def = roc_auc_score(has_lb, default_scores)
    except ValueError:
        lb_auc_def = 0.5

    return {
        "label": label,
        "weights": {
            "zone_amplifier": weights.zone_amplifier,
            "penetration_weight": weights.penetration_weight,
            "space_weight": weights.space_weight,
            "pressure_scaling": weights.pressure_scaling,
        },
        "delta_xt_spearman": round(float(rho), 4),
        "delta_xt_p_value": float(rho_p),
        "lines_broken_auc": round(lb_auc, 4),
        "vs_default": {
            "delta_xt_spearman_default": round(float(rho_def), 4),
            "delta_xt_spearman_improvement": round(float(rho - rho_def), 4),
            "lines_broken_auc_default": round(lb_auc_def, 4),
            "lines_broken_auc_improvement": round(float(lb_auc - lb_auc_def), 4),
        },
    }
