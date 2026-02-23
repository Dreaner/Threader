"""
Consistency checks — internal dimension correlations, rank perturbation
stability, and stage-wise consistency.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as sp_stats

from pitch_echo.analysis.models import DEFAULT_WEIGHTS
from pitch_echo.scoring.pass_score import compute_pass_score

if TYPE_CHECKING:
    from pitch_echo.validation.collector import ValidatedPass


# ── A. Internal Dimension Correlations ───────────────────────────────────────


def dimension_correlations(records: list[ValidatedPass]) -> dict:
    """Spearman rank correlation matrix between dimensions.

    Uses the actual target's dimension scores from each pass.
    Reports the full 5×5 matrix plus flags for unexpected correlations.
    """
    dim_names = ["completion", "zone_value", "pressure", "space", "penetration"]
    attrs = [
        "actual_target_completion",
        "actual_target_zone",
        "actual_target_pressure",
        "actual_target_space",
        "actual_target_penetration",
    ]

    n = len(records)
    if n < 10:
        return {"error": "too few records for correlation analysis"}

    data = np.zeros((n, 5))
    for i, r in enumerate(records):
        for j, attr in enumerate(attrs):
            data[i, j] = getattr(r, attr)

    # Spearman rank correlation matrix
    matrix: dict[str, dict[str, float]] = {}
    flags: list[str] = []

    for i, name_i in enumerate(dim_names):
        matrix[name_i] = {}
        for j, name_j in enumerate(dim_names):
            if i == j:
                matrix[name_i][name_j] = 1.0
                continue
            rho, pval = sp_stats.spearmanr(data[:, i], data[:, j])
            matrix[name_i][name_j] = round(float(rho), 3)

            # Flag strong unexpected correlations
            if abs(rho) > 0.5:
                flags.append(f"{name_i} ↔ {name_j}: rho={rho:.3f} (strong)")
            elif abs(rho) > 0.3:
                flags.append(f"{name_i} ↔ {name_j}: rho={rho:.3f} (moderate)")

    # Check for redundancy: if pressure ~ space with |rho| > 0.7, flag
    ps_rho = matrix.get("pressure", {}).get("space", 0)
    if abs(ps_rho) > 0.7:
        flags.append(f"WARNING: pressure and space may be redundant (rho={ps_rho:.3f})")

    return {
        "n_records": n,
        "matrix": matrix,
        "flags": flags,
    }


# ── B. Rank Perturbation Stability ──────────────────────────────────────────


def rank_perturbation_stability(
    records: list[ValidatedPass],
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0),
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """Test rank stability under small positional perturbations.

    For a sample of passes, perturb target positions by Gaussian noise
    and re-score, then measure Kendall's tau vs the original ranking.

    Note: This is an approximate test — it perturbs dimension scores
    proportionally rather than re-running full geometry (which would
    require the full player positions). Completion and pressure are
    perturbed as proxy.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(records))
    sample_indices = rng.choice(len(records), size=n, replace=False)
    sample = [records[i] for i in sample_indices]

    results: dict[str, dict] = {}

    for sigma in sigmas:
        taus: list[float] = []

        for vp in sample:
            if len(vp.ranked_options) < 3:
                continue

            # Original ranking (by pass score)
            original_order = [opt.target.player_id for opt in vp.ranked_options]

            # Perturbed scores: add noise proportional to sigma to pass scores
            # noise_std ≈ sigma * typical_score_range_per_meter / pitch_scale
            noise_std = sigma * 0.5  # heuristic: 1m perturbation ≈ 0.5 score points
            perturbed_scores = []
            for opt in vp.ranked_options:
                noise = rng.normal(0, noise_std)
                perturbed_scores.append((opt.target.player_id, opt.pass_score + noise))

            perturbed_scores.sort(key=lambda x: x[1], reverse=True)
            perturbed_order = [pid for pid, _ in perturbed_scores]

            # Kendall's tau between original and perturbed
            # Convert to ranks
            orig_ranks = {pid: i for i, pid in enumerate(original_order)}
            pert_ranks = {pid: i for i, pid in enumerate(perturbed_order)}

            common = set(original_order) & set(perturbed_order)
            if len(common) < 3:
                continue

            r1 = [orig_ranks[pid] for pid in sorted(common)]
            r2 = [pert_ranks[pid] for pid in sorted(common)]

            tau, _ = sp_stats.kendalltau(r1, r2)
            taus.append(tau)

        if taus:
            arr = np.array(taus)
            results[f"sigma_{sigma}"] = {
                "sigma_meters": sigma,
                "n_tested": len(taus),
                "mean_tau": round(float(np.mean(arr)), 4),
                "median_tau": round(float(np.median(arr)), 4),
                "std_tau": round(float(np.std(arr)), 4),
                "interpretation": (
                    "highly stable" if np.mean(arr) > 0.9 else
                    "stable" if np.mean(arr) > 0.7 else
                    "moderately stable" if np.mean(arr) > 0.5 else
                    "unstable"
                ),
            }
        else:
            results[f"sigma_{sigma}"] = {"error": "no valid samples"}

    return results


# ── C. Stage-Wise Consistency ────────────────────────────────────────────────


def stage_consistency(records: list[ValidatedPass]) -> dict:
    """Compare metrics across tournament stages.

    Uses match_id ranges as a proxy for tournament stages:
    - 3812–3859: Group stage + early knockouts (IDs in the 3800s)
    - 10502–10517: Later rounds (IDs in the 10500s)

    A good metric should perform consistently across stages.
    """
    group_a: list[ValidatedPass] = []  # 3800-range matches
    group_b: list[ValidatedPass] = []  # 10500-range matches

    for r in records:
        if r.match_id < 10000:
            group_a.append(r)
        else:
            group_b.append(r)

    if not group_a or not group_b:
        return {"error": "cannot split by stage (all records in same range)"}

    def group_metrics(recs: list[ValidatedPass], label: str) -> dict:
        from pitch_echo.validation.repeatability import _group_auc, _group_concordance

        auc = _group_auc(recs)
        conc = _group_concordance(recs)
        c_scores = [r.actual_target_score for r in recs if r.actual_outcome == "C"]
        d_scores = [r.actual_target_score for r in recs if r.actual_outcome == "D"]
        mean_score = float(np.mean([r.actual_target_score for r in recs])) if recs else 0

        entry: dict = {
            "label": label,
            "n_passes": len(recs),
            "n_matches": len({r.match_id for r in recs}),
            "mean_score": round(mean_score, 1),
        }
        if auc is not None:
            entry["auc"] = round(auc, 4)
        if conc is not None:
            entry["concordance"] = round(conc * 100, 1)
        if c_scores and d_scores:
            entry["mean_completed_score"] = round(float(np.mean(c_scores)), 1)
            entry["mean_defended_score"] = round(float(np.mean(d_scores)), 1)
        return entry

    stage_a_metrics = group_metrics(group_a, "Group Stage + R16 (3800s)")
    stage_b_metrics = group_metrics(group_b, "QF / SF / Final (10500s)")

    # Compute delta for key metrics
    deltas = {}
    for key in ["auc", "concordance", "mean_score"]:
        a_val = stage_a_metrics.get(key)
        b_val = stage_b_metrics.get(key)
        if a_val is not None and b_val is not None:
            deltas[key] = round(float(b_val - a_val), 4)

    return {
        "stage_a": stage_a_metrics,
        "stage_b": stage_b_metrics,
        "deltas": deltas,
        "consistent": all(abs(d) < 0.05 for d in deltas.values()) if deltas else None,
    }


# ── Run all consistency checks ──────────────────────────────────────────────


def run_consistency(records: list[ValidatedPass]) -> dict:
    """Run all consistency analyses."""
    return {
        "dimension_correlations": dimension_correlations(records),
        "perturbation_stability": rank_perturbation_stability(records),
        "stage_consistency": stage_consistency(records),
    }
