"""
Statistical significance tests for the Pass Score formula.

Three categories:
  A. Score vs Outcome — does Pass Score predict pass completion?
  B. BetterOption Concordance — does Threader agree with PFF analysts?
  C. Pressure Validation — does computed pressure match PFF annotation?
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from threader.validation.collector import ValidatedPass


# ── Helpers ──────────────────────────────────────────────────────────────────


def _bootstrap_auc(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap AUC-ROC with 95% CI.

    Returns (auc, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5, 0.5

    aucs: list[float] = []
    for _ in range(n_boot):
        idx_p = rng.integers(0, n_pos, size=n_pos)
        idx_n = rng.integers(0, n_neg, size=n_neg)
        p_sample = pos_scores[idx_p]
        n_sample = neg_scores[idx_n]
        try:
            u, _ = stats.mannwhitneyu(p_sample, n_sample, alternative="greater")
            aucs.append(u / (n_pos * n_neg))
        except ValueError:
            aucs.append(0.5)

    aucs_arr = np.array(aucs)
    return float(np.median(aucs_arr)), float(np.percentile(aucs_arr, 2.5)), float(np.percentile(aucs_arr, 97.5))


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


# ── A. Score vs Outcome ─────────────────────────────────────────────────────


def score_vs_outcome(records: list[ValidatedPass]) -> dict:
    """Test whether Pass Score predicts completion.

    Splits passes by outcome (C vs D) and tests:
    - Mann-Whitney U (are completed-pass scores significantly higher?)
    - Cohen's d effect size
    - AUC-ROC with bootstrap 95% CI
    - Per-dimension breakdown
    """
    completed = [r for r in records if r.actual_outcome == "C"]
    defended = [r for r in records if r.actual_outcome == "D"]

    results: dict = {"n_completed": len(completed), "n_defended": len(defended)}

    # Overall Pass Score
    c_scores = np.array([r.actual_target_score for r in completed])
    d_scores = np.array([r.actual_target_score for r in defended])

    if len(c_scores) > 0 and len(d_scores) > 0:
        u_stat, p_value = stats.mannwhitneyu(c_scores, d_scores, alternative="greater")
        n1, n2 = len(c_scores), len(d_scores)
        auc = u_stat / (n1 * n2)
        auc_med, ci_lo, ci_hi = _bootstrap_auc(c_scores, d_scores)
        d = _cohens_d(c_scores, d_scores)

        results["pass_score"] = {
            "mean_completed": round(float(np.mean(c_scores)), 2),
            "mean_defended": round(float(np.mean(d_scores)), 2),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_value),
            "auc_roc": round(auc, 4),
            "auc_bootstrap_median": round(auc_med, 4),
            "auc_95ci": (round(ci_lo, 4), round(ci_hi, 4)),
            "cohens_d": round(d, 3),
        }
    else:
        results["pass_score"] = {"error": "insufficient data"}

    # Per-dimension breakdown
    dimensions = {
        "completion": ("actual_target_completion", ),
        "zone_value": ("actual_target_zone", ),
        "pressure": ("actual_target_pressure", ),
        "space": ("actual_target_space", ),
        "penetration": ("actual_target_penetration", ),
    }

    dim_results = {}
    for dim_name, (attr, ) in dimensions.items():
        c_vals = np.array([getattr(r, attr) for r in completed])
        d_vals = np.array([getattr(r, attr) for r in defended])

        if len(c_vals) > 0 and len(d_vals) > 0:
            # For pressure, lower is "better" for the receiver, so flip alternative
            alt = "less" if dim_name == "pressure" else "greater"
            u_stat, p_val = stats.mannwhitneyu(c_vals, d_vals, alternative=alt)
            n1, n2 = len(c_vals), len(d_vals)
            auc_raw = u_stat / (n1 * n2)
            # Normalize so higher AUC = more predictive regardless of direction
            auc_dim = auc_raw if dim_name != "pressure" else 1.0 - auc_raw
            d_val = _cohens_d(c_vals, d_vals)

            dim_results[dim_name] = {
                "mean_completed": round(float(np.mean(c_vals)), 4),
                "mean_defended": round(float(np.mean(d_vals)), 4),
                "p_value": float(p_val),
                "auc_roc": round(auc_dim, 4),
                "cohens_d": round(d_val, 3),
            }
        else:
            dim_results[dim_name] = {"error": "insufficient data"}

    results["dimensions"] = dim_results
    return results


# ── B. BetterOption Concordance ──────────────────────────────────────────────


def better_option_concordance(records: list[ValidatedPass]) -> dict:
    """Test whether Threader ranks PFF's betterOption above the actual target.

    - Concordance rate (% where betterOption ranked higher)
    - Binomial test vs chance (1/N_teammates)
    - Wilcoxon signed-rank on score differences
    """
    # Filter to passes with betterOption annotation where the player was found
    annotated = [r for r in records if r.pff_better_option_player_id and r.better_option_rank > 0]

    results: dict = {"n_annotated": len(annotated)}
    if not annotated:
        results["error"] = "no betterOption annotations found in data"
        return results

    # Concordance: betterOption ranked higher (lower rank number) than actual target
    concordant = sum(1 for r in annotated if r.better_option_rank < r.actual_target_rank)
    concordance_rate = concordant / len(annotated)

    # Chance baseline: average probability of ranking one specific player above another
    # In uniform ranking of N options, P(A ranked above B) = 0.5
    # But chance of a specific player being ranked above another random one = ~1/N
    # We use 0.5 as the null (any two players equally likely to be ranked either way)
    binom_result = stats.binomtest(concordant, len(annotated), p=0.5, alternative="greater")

    # Wilcoxon signed-rank: is betterOption score systematically > actual target score?
    score_diffs = np.array([r.better_option_score - r.actual_target_score for r in annotated])
    nonzero_diffs = score_diffs[score_diffs != 0]

    wilcoxon_result: dict = {}
    if len(nonzero_diffs) >= 10:
        w_stat, w_pval = stats.wilcoxon(nonzero_diffs, alternative="greater")
        wilcoxon_result = {
            "statistic": float(w_stat),
            "p_value": float(w_pval),
            "mean_diff": round(float(np.mean(score_diffs)), 2),
            "median_diff": round(float(np.median(score_diffs)), 2),
        }
    else:
        wilcoxon_result = {"error": f"only {len(nonzero_diffs)} non-zero diffs (need ≥ 10)"}

    # MRR of betterOption player
    mrr = float(np.mean([1.0 / r.better_option_rank for r in annotated]))

    results.update({
        "concordance_rate": round(concordance_rate * 100, 1),
        "concordant": concordant,
        "binomial_p_value": float(binom_result.pvalue),
        "wilcoxon": wilcoxon_result,
        "mrr": round(mrr, 4),
        "mean_better_rank": round(float(np.mean([r.better_option_rank for r in annotated])), 1),
        "mean_actual_rank": round(float(np.mean([r.actual_target_rank for r in annotated])), 1),
    })

    return results


# ── C. Pressure Validation ──────────────────────────────────────────────────


def pressure_validation(records: list[ValidatedPass]) -> dict:
    """Validate Threader's pressure metric against PFF's pressureType annotation.

    - Mann-Whitney U on computed pressure: P vs N groups
    - AUC-ROC: can pressure score distinguish pressed from not pressed?
    - Per-score-bucket analysis
    """
    pressed = [r for r in records if r.pff_pressure_type == "P"]
    not_pressed = [r for r in records if r.pff_pressure_type == "N"]

    results: dict = {"n_pressed": len(pressed), "n_not_pressed": len(not_pressed)}

    if not pressed or not not_pressed:
        results["error"] = "insufficient pressure annotations"
        return results

    p_pressures = np.array([r.actual_target_pressure for r in pressed])
    n_pressures = np.array([r.actual_target_pressure for r in not_pressed])

    # Mann-Whitney: are pressures higher when PFF says "pressed"?
    u_stat, p_value = stats.mannwhitneyu(p_pressures, n_pressures, alternative="greater")
    n1, n2 = len(p_pressures), len(n_pressures)
    auc = u_stat / (n1 * n2)
    auc_med, ci_lo, ci_hi = _bootstrap_auc(p_pressures, n_pressures)
    d = _cohens_d(p_pressures, n_pressures)

    results.update({
        "mean_pressure_when_pressed": round(float(np.mean(p_pressures)), 2),
        "mean_pressure_when_not": round(float(np.mean(n_pressures)), 2),
        "mann_whitney_u": float(u_stat),
        "p_value": float(p_value),
        "auc_roc": round(auc, 4),
        "auc_bootstrap_median": round(auc_med, 4),
        "auc_95ci": (round(ci_lo, 4), round(ci_hi, 4)),
        "cohens_d": round(d, 3),
    })

    # Bucket analysis: split pressure score into buckets and show P/N ratio
    bucket_edges = [0, 1, 2, 3, 5, 7, 10.01]
    buckets: list[dict] = []

    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        bp = sum(1 for p in p_pressures if lo <= p < hi)
        bn = sum(1 for p in n_pressures if lo <= p < hi)
        total = bp + bn
        buckets.append({
            "range": f"{lo:.0f}-{hi:.0f}",
            "pressed": bp,
            "not_pressed": bn,
            "total": total,
            "pressed_pct": round(bp / total * 100, 1) if total > 0 else 0.0,
        })

    results["buckets"] = buckets
    return results


# ── Run all significance tests ───────────────────────────────────────────────


def run_significance_tests(records: list[ValidatedPass]) -> dict:
    """Run all significance tests and return combined results."""
    return {
        "score_vs_outcome": score_vs_outcome(records),
        "better_option": better_option_concordance(records),
        "pressure": pressure_validation(records),
    }
