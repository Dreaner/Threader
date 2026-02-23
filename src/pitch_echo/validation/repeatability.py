"""
Repeatability assessment — cross-match consistency, split-half reliability,
and bootstrap confidence intervals.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from threader.validation.collector import ValidatedPass


# ── Helpers (per-group metric computation) ───────────────────────────────────


def _group_auc(records: list[ValidatedPass]) -> float | None:
    """Compute AUC-ROC for completion prediction within a group."""
    c_scores = [r.actual_target_score for r in records if r.actual_outcome == "C"]
    d_scores = [r.actual_target_score for r in records if r.actual_outcome == "D"]
    if len(c_scores) < 5 or len(d_scores) < 5:
        return None
    try:
        u, _ = sp_stats.mannwhitneyu(c_scores, d_scores, alternative="greater")
        return u / (len(c_scores) * len(d_scores))
    except ValueError:
        return None


def _group_concordance(records: list[ValidatedPass]) -> float | None:
    """Compute betterOption concordance rate within a group."""
    annotated = [r for r in records if r.pff_better_option_player_id and r.better_option_rank > 0]
    if len(annotated) < 5:
        return None
    hits = sum(1 for r in annotated if r.better_option_rank < r.actual_target_rank)
    return hits / len(annotated)


def _group_mean_score(records: list[ValidatedPass]) -> float:
    """Mean Pass Score in a group."""
    if not records:
        return 0.0
    return float(np.mean([r.actual_target_score for r in records]))


# ── A. Cross-Match Consistency ───────────────────────────────────────────────


def cross_match_consistency(records: list[ValidatedPass]) -> dict:
    """Compute per-match metrics and assess consistency.

    Reports: mean, std, CV, IQR, outlier matches.
    """
    by_match: dict[int, list[ValidatedPass]] = defaultdict(list)
    for r in records:
        by_match[r.match_id].append(r)

    match_metrics: list[dict] = []
    aucs: list[float] = []
    concordances: list[float] = []
    mean_scores: list[float] = []

    for mid, mrecs in sorted(by_match.items()):
        auc = _group_auc(mrecs)
        conc = _group_concordance(mrecs)
        ms = _group_mean_score(mrecs)

        entry = {"match_id": mid, "n_passes": len(mrecs), "mean_score": round(ms, 1)}
        if auc is not None:
            entry["auc"] = round(auc, 4)
            aucs.append(auc)
        if conc is not None:
            entry["concordance"] = round(conc * 100, 1)
            concordances.append(conc)
        mean_scores.append(ms)
        match_metrics.append(entry)

    def distribution_stats(values: list[float], label: str) -> dict:
        arr = np.array(values)
        if len(arr) < 2:
            return {"label": label, "n": len(arr), "error": "too few values"}
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        cv = std / mean if mean != 0 else float("inf")
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        outliers = int(np.sum((arr < lo_fence) | (arr > hi_fence)))
        return {
            "label": label,
            "n": len(arr),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv": round(cv, 4),
            "q1": round(q1, 4),
            "median": round(float(np.median(arr)), 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "outliers": outliers,
        }

    return {
        "n_matches": len(by_match),
        "auc_dist": distribution_stats(aucs, "AUC-ROC"),
        "concordance_dist": distribution_stats(concordances, "Concordance"),
        "mean_score_dist": distribution_stats(mean_scores, "Mean Score"),
        "per_match": match_metrics,
    }


# ── B. Split-Half Reliability ───────────────────────────────────────────────


def split_half_reliability(
    records: list[ValidatedPass],
    n_splits: int = 100,
    seed: int = 42,
) -> dict:
    """Randomly split matches into two halves repeatedly and measure correlation.

    For each split:
    - Draw two independent random samples of matches (with replacement)
    - Compute AUC-ROC and concordance on each sample
    - Correlate across splits to assess reliability
    """
    match_ids = sorted({r.match_id for r in records})
    n_matches = len(match_ids)
    if n_matches < 6:
        return {"error": f"only {n_matches} matches, need ≥6 for split-half"}

    by_match: dict[int, list[ValidatedPass]] = defaultdict(list)
    for r in records:
        by_match[r.match_id].append(r)

    rng = np.random.default_rng(seed)
    half_size = n_matches // 2

    auc_half1: list[float] = []
    auc_half2: list[float] = []
    conc_half1: list[float] = []
    conc_half2: list[float] = []

    match_id_arr = np.array(match_ids)

    for _ in range(n_splits):
        # Two independent random samples (with replacement)
        h1_ids = set(rng.choice(match_id_arr, size=half_size, replace=True))
        h2_ids = set(rng.choice(match_id_arr, size=half_size, replace=True))

        h1_records = [r for r in records if r.match_id in h1_ids]
        h2_records = [r for r in records if r.match_id in h2_ids]

        a1 = _group_auc(h1_records)
        a2 = _group_auc(h2_records)
        c1 = _group_concordance(h1_records)
        c2 = _group_concordance(h2_records)

        # Only keep splits where both halves have valid metrics
        if a1 is not None and a2 is not None:
            auc_half1.append(a1)
            auc_half2.append(a2)
        if c1 is not None and c2 is not None:
            conc_half1.append(c1)
            conc_half2.append(c2)

    results: dict = {"n_splits": n_splits, "n_matches": n_matches}

    if len(auc_half1) >= 10:
        r_auc, p_auc = sp_stats.pearsonr(auc_half1, auc_half2)
        results["auc_correlation"] = {
            "pearson_r": round(r_auc, 4),
            "p_value": float(p_auc),
            "valid_splits": len(auc_half1),
            "interpretation": (
                "excellent" if r_auc > 0.8 else
                "good" if r_auc > 0.6 else
                "moderate" if r_auc > 0.4 else
                "poor"
            ),
        }
    else:
        results["auc_correlation"] = {"error": "insufficient valid splits"}

    if len(conc_half1) >= 10:
        r_conc, p_conc = sp_stats.pearsonr(conc_half1, conc_half2)
        results["concordance_correlation"] = {
            "pearson_r": round(r_conc, 4),
            "p_value": float(p_conc),
            "valid_splits": len(conc_half1),
            "interpretation": (
                "excellent" if r_conc > 0.8 else
                "good" if r_conc > 0.6 else
                "moderate" if r_conc > 0.4 else
                "poor"
            ),
        }
    else:
        results["concordance_correlation"] = {"error": "insufficient valid splits"}

    return results


# ── C. Bootstrap Confidence Intervals ────────────────────────────────────────


def bootstrap_ci(
    records: list[ValidatedPass],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap CIs for headline metrics.

    Resamples pass events (with replacement) and computes:
    - AUC-ROC 95% CI
    - Concordance rate 95% CI
    - Mean score difference (completed vs defended) 95% CI
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    arr = np.array(records, dtype=object)

    boot_aucs: list[float] = []
    boot_concs: list[float] = []
    boot_diffs: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [arr[i] for i in idx]

        auc = _group_auc(sample)
        if auc is not None:
            boot_aucs.append(auc)

        conc = _group_concordance(sample)
        if conc is not None:
            boot_concs.append(conc)

        c_scores = [r.actual_target_score for r in sample if r.actual_outcome == "C"]
        d_scores = [r.actual_target_score for r in sample if r.actual_outcome == "D"]
        if c_scores and d_scores:
            boot_diffs.append(float(np.mean(c_scores) - np.mean(d_scores)))

    def ci_stats(values: list[float], label: str) -> dict:
        if len(values) < 20:
            return {"label": label, "error": "insufficient bootstrap samples"}
        arr_v = np.array(values)
        return {
            "label": label,
            "mean": round(float(np.mean(arr_v)), 4),
            "median": round(float(np.median(arr_v)), 4),
            "ci_95_lower": round(float(np.percentile(arr_v, 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(arr_v, 97.5)), 4),
            "n_valid": len(values),
        }

    return {
        "auc_roc": ci_stats(boot_aucs, "AUC-ROC"),
        "concordance": ci_stats(boot_concs, "Concordance"),
        "score_diff": ci_stats(boot_diffs, "Score Diff (C-D)"),
    }


# ── Run all repeatability checks ────────────────────────────────────────────


def run_repeatability(records: list[ValidatedPass]) -> dict:
    """Run all repeatability analyses."""
    return {
        "cross_match": cross_match_consistency(records),
        "split_half": split_half_reliability(records),
        "bootstrap": bootstrap_ci(records),
    }
