#!/usr/bin/env python3
"""
Threader Scoring Validation — CLI entry point.

Runs the full validation framework:
  1. Collect all passes from match data → ValidatedPass records
  2. Baseline comparison (random, distance-only, zone-only, etc.)
  3. Significance tests (Mann-Whitney, AUC-ROC, binomial, Wilcoxon)
  4. Sensitivity analysis (weight sweep, ablation, pressure deep-dive)
  5. Repeatability (split-half, bootstrap CIs, cross-match)
  6. Consistency (dimension correlations, perturbation, stage-wise)

Usage:
    python scripts/validate_scoring.py                  # full 64-match run
    python scripts/validate_scoring.py --matches 5      # quick test (5 matches)
    python scripts/validate_scoring.py --skip-sensitivity  # skip slow sweep
    python scripts/validate_scoring.py --cache-dir .cache  # custom cache path
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path so imports work when run as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ── Pretty-printing helpers ──────────────────────────────────────────────────


def _header(title: str) -> str:
    width = 60
    return f"\n{'═' * width}\n  {title}\n{'═' * width}"


def _subheader(title: str) -> str:
    return f"\n  ── {title} {'─' * max(1, 50 - len(title))}"


def _kv(key: str, value, indent: int = 4) -> str:
    pad = " " * indent
    return f"{pad}{key}: {value}"


def _pval_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ── Report sections ──────────────────────────────────────────────────────────


def print_data_summary(summary: dict) -> None:
    print(_header("DATA SUMMARY"))
    print(_kv("Matches analyzed", summary.get("matches", "?")))
    print(_kv("Total passes", summary.get("total", 0)))
    n = summary.get("total", 1)
    c = summary.get("completed", 0)
    print(_kv("Completed", f"{c} ({summary.get('completed_pct', 0)}%)"))
    print(_kv("Defended", summary.get("defended", 0)))
    bo = summary.get("with_better_option", 0)
    print(_kv("With betterOption annotation", f"{bo} ({summary.get('with_better_option_pct', 0)}%)"))
    wp = summary.get("with_pressure_annotation", 0)
    pct = round(wp / n * 100, 1) if n > 0 else 0
    print(_kv("With pressure annotation", f"{wp} ({pct}%)"))
    print(_kv("  Pressed (P)", summary.get("pressed", 0)))
    print(_kv("  Not pressed (N)", summary.get("not_pressed", 0)))


def print_baselines(results: dict[str, dict]) -> None:
    print(_header("BASELINE COMPARISON"))

    try:
        from tabulate import tabulate
        rows = []
        for name, metrics in results.items():
            rows.append([
                name,
                f"{metrics['auc_roc']:.4f}",
                f"{metrics['concordance']:.1f}%",
                f"{metrics['mrr']:.4f}",
            ])
        print(tabulate(rows, headers=["Model", "AUC-ROC", "PFF Concord.", "MRR"],
                        tablefmt="simple", stralign="right"))
    except ImportError:
        # Fallback without tabulate
        hdr = f"  {'Model':<22} {'AUC-ROC':>8} {'PFF Concord.':>13} {'MRR':>8}"
        print(hdr)
        print(f"  {'─' * 55}")
        for name, metrics in results.items():
            print(f"  {name:<22} {metrics['auc_roc']:>8.4f} {metrics['concordance']:>12.1f}% {metrics['mrr']:>8.4f}")


def print_significance(results: dict) -> None:
    print(_header("SIGNIFICANCE TESTS"))

    # A. Offensive Value (PRIMARY)
    ov = results.get("offensive_value", {})
    print(_subheader("A. Pass Score vs Offensive Value (PRIMARY)"))

    dxt = ov.get("delta_xt_spearman", {})
    if "error" not in dxt:
        rho = dxt.get("rho", 0)
        p = dxt.get("p_value", 1)
        print(_kv("ΔxT Spearman ρ", f"{rho:.4f} {_pval_stars(p)} (p={p:.2e})"))
        print(_kv("Interpretation", dxt.get("interpretation", "?")))

        # Quartile table
        qstats = ov.get("delta_xt_by_quartile", [])
        if qstats:
            try:
                from tabulate import tabulate
                rows = [[q["quartile"], q["n"], f"{q['mean_score']:.2f}", f"{q['mean_delta_xt']:.5f}"]
                        for q in qstats]
                print(f"\n    ΔxT by Pass Score quartile:")
                print(tabulate(rows, headers=["Quartile", "N", "Mean Score", "Mean ΔxT"],
                                tablefmt="simple", stralign="right"))
            except ImportError:
                for q in qstats:
                    print(f"      {q['quartile']}: n={q['n']} score={q['mean_score']:.2f} ΔxT={q['mean_delta_xt']:.5f}")
    else:
        print(_kv("ΔxT Spearman", dxt.get("error")))

    lb = ov.get("lines_broken_auc", {})
    if "error" not in lb:
        print(f"\n    Lines-broken prediction (≥1 line broken):")
        print(_kv("AUC-ROC", lb.get("auc_roc")))
        ci = lb.get("auc_95ci", (0, 0))
        print(_kv("AUC 95% CI", f"[{ci[0]:.4f}, {ci[1]:.4f}]"))
        p = lb.get("p_value", 1)
        print(_kv("Mann-Whitney p-value", f"{p:.2e} {_pval_stars(p)}"))
        print(_kv("Cohen's d", lb.get("cohens_d")))
        print(_kv("Mean score (broke ≥1)", lb.get("mean_score_broken")))
        print(_kv("Mean score (broke 0)", lb.get("mean_score_not_broken")))

        bd = ov.get("lines_broken_breakdown", [])
        if bd:
            try:
                from tabulate import tabulate
                rows = [[b["lines_broken"], b["n"], f"{b['mean_score']:.2f}", f"{b['std_score']:.2f}"]
                        for b in bd]
                print(f"\n    Breakdown by lines broken:")
                print(tabulate(rows, headers=["Lines", "N", "Mean Score", "Std"],
                                tablefmt="simple", stralign="right"))
            except ImportError:
                for b in bd:
                    print(f"      {b['lines_broken']} lines: n={b['n']} mean={b['mean_score']:.2f} std={b['std_score']:.2f}")
    else:
        print(_kv("Lines-broken AUC", lb.get("error")))

    counts = ov.get("lines_broken_counts", {})
    if counts:
        print(_kv("Passes breaking ≥1 line", counts.get("broke_ge1", 0)))
        print(_kv("Passes breaking 0 lines", counts.get("broke_0", 0)))

    # B. BetterOption
    bo = results.get("better_option", {})
    print(_subheader("B. BetterOption Concordance"))
    if "error" not in bo:
        print(_kv("Annotated passes", bo.get("n_annotated")))
        print(_kv("Concordance rate", f"{bo.get('concordance_rate', 0)}%"))
        print(_kv("Concordant", bo.get("concordant")))
        bp = bo.get("binomial_p_value", 1)
        print(_kv("Binomial test p-value", f"{bp:.2e} {_pval_stars(bp)}"))
        print(_kv("MRR", bo.get("mrr")))
        print(_kv("Mean betterOption rank", bo.get("mean_better_rank")))
        print(_kv("Mean actual target rank", bo.get("mean_actual_rank")))

        w = bo.get("wilcoxon", {})
        if "error" not in w:
            wp = w.get("p_value", 1)
            print(_kv("Wilcoxon signed-rank p", f"{wp:.2e} {_pval_stars(wp)}"))
            print(_kv("Mean score diff (better - actual)", w.get("mean_diff")))
        else:
            print(_kv("Wilcoxon", w.get("error")))
    else:
        print(_kv("Error", bo.get("error")))

    # C. Pressure
    pv = results.get("pressure", {})
    print(_subheader("C. Pressure Validation (vs PFF pressureType)"))
    if "error" not in pv:
        print(_kv("Pressed (P) passes", pv.get("n_pressed")))
        print(_kv("Not pressed (N) passes", pv.get("n_not_pressed")))
        print(_kv("Mean pressure when pressed", pv.get("mean_pressure_when_pressed")))
        print(_kv("Mean pressure when not", pv.get("mean_pressure_when_not")))
        pp = pv.get("p_value", 1)
        print(_kv("Mann-Whitney p-value", f"{pp:.2e} {_pval_stars(pp)}"))
        print(_kv("AUC-ROC", pv.get("auc_roc")))
        ci = pv.get("auc_95ci", (0, 0))
        print(_kv("AUC 95% CI", f"[{ci[0]:.4f}, {ci[1]:.4f}]"))
        print(_kv("Cohen's d", pv.get("cohens_d")))

        # Buckets
        buckets = pv.get("buckets", [])
        if buckets:
            print(f"\n    Pressure score distribution by PFF annotation:")
            try:
                from tabulate import tabulate
                rows = [[b["range"], b["pressed"], b["not_pressed"], b["total"], f"{b['pressed_pct']}%"]
                        for b in buckets]
                print(tabulate(rows, headers=["Range", "P", "N", "Total", "P%"],
                                tablefmt="simple", stralign="right"))
            except ImportError:
                for b in buckets:
                    print(f"      [{b['range']}] P={b['pressed']:>4} N={b['not_pressed']:>4} ({b['pressed_pct']}% pressed)")
    else:
        print(_kv("Error", pv.get("error")))

    # D. Score vs Outcome (observational — NOT the primary metric)
    svo = results.get("score_vs_outcome", {})
    ps = svo.get("pass_score", {})
    print(_subheader("D. Pass Score vs Outcome (observational)"))
    print("    NOTE: By design, Pass Score rewards risky attacking passes")
    print("          which may have lower completion rates. Low AUC here")
    print("          is expected and NOT a formula deficiency.")
    if "error" not in ps:
        print(_kv("Mean score (completed)", ps.get("mean_completed")))
        print(_kv("Mean score (defended)", ps.get("mean_defended")))
        p = ps.get("p_value", 1)
        print(_kv("Mann-Whitney p-value", f"{p:.2e} {_pval_stars(p)}"))
        print(_kv("AUC-ROC", ps.get("auc_roc")))
        ci = ps.get("auc_95ci", (0, 0))
        print(_kv("AUC 95% CI (bootstrap)", f"[{ci[0]:.4f}, {ci[1]:.4f}]"))
        print(_kv("Cohen's d", ps.get("cohens_d")))


def print_sensitivity(results: dict) -> None:
    print(_header("SENSITIVITY ANALYSIS"))

    # Ablation
    abl = results.get("ablation", {})
    if abl:
        print(_subheader("Dimension Ablation (vs baseline)"))
        baseline = abl.get("baseline", {})
        print(_kv("Baseline AUC", baseline.get("auc")))
        print(_kv("Baseline concordance", f"{baseline.get('concordance', 0)}%"))
        print()
        try:
            from tabulate import tabulate
            rows = []
            for dim in ["zone_value", "penetration", "space", "pressure"]:
                d = abl.get(dim, {})
                if not d:
                    continue
                rows.append([
                    f"−{dim}",
                    f"{d.get('auc', 0):.4f}",
                    f"{d.get('auc_delta', 0):+.4f}",
                    f"{d.get('concordance', 0):.1f}%",
                    f"{d.get('concordance_delta', 0):+.1f}%",
                    f"{d.get('flip_rate', 0):.1f}%",
                ])
            print(tabulate(rows,
                            headers=["Ablated", "AUC", "ΔAUC", "Concord.", "ΔConc.", "Flip%"],
                            tablefmt="simple", stralign="right"))
        except ImportError:
            for dim in ["zone_value", "penetration", "space", "pressure"]:
                d = abl.get(dim, {})
                if not d:
                    continue
                print(f"    −{dim:<15} AUC={d.get('auc', 0):.4f} (Δ{d.get('auc_delta', 0):+.4f})  "
                      f"Conc={d.get('concordance', 0):.1f}% (Δ{d.get('concordance_delta', 0):+.1f}%)  "
                      f"Flip={d.get('flip_rate', 0):.1f}%")

    # Weight sweep summary
    ws = results.get("weight_sweep", {})
    if ws:
        print(_subheader("Weight Sweep Summary (AUC range)"))
        for wname, rows in ws.items():
            aucs = [r["auc"] for r in rows]
            vals = [r["value"] for r in rows]
            best_idx = max(range(len(aucs)), key=lambda i: aucs[i])
            print(f"    {wname:<22} AUC range: [{min(aucs):.4f}, {max(aucs):.4f}]  "
                  f"best @ {vals[best_idx]:.4f} (AUC={aucs[best_idx]:.4f})")

    # Pressure deep dive
    pdive = results.get("pressure_deep_dive", {})
    if pdive:
        print(_subheader("Pressure Deep-Dive"))
        nopres = pdive.get("no_pressure_baseline", {})
        print(_kv("No-pressure baseline AUC", nopres.get("auc")))
        print(_kv("No-pressure baseline concordance", f"{nopres.get('concordance', 0)}%"))

        best_k = pdive.get("best_k_by_auc", {})
        print(_kv("Best pressure_scaling (k)", best_k.get("k")))
        print(_kv("Best k AUC", best_k.get("auc")))
        print(_kv("Best k AUC Δ vs no-pressure", best_k.get("auc_delta_vs_no_pressure")))

        disc = pdive.get("pressure_discrimination", {})
        if "error" not in disc:
            print(_kv("Pressure metric AUC (P vs N)", disc.get("auc")))
            dp = disc.get("p_value", 1)
            print(_kv("Pressure metric p-value", f"{dp:.2e} {_pval_stars(dp)}"))


def print_repeatability(results: dict) -> None:
    print(_header("REPEATABILITY"))

    # Split-half
    sh = results.get("split_half", {})
    if sh and "error" not in sh:
        print(_subheader("Split-Half Reliability"))
        print(_kv("Splits", sh.get("n_splits")))
        print(_kv("Matches", sh.get("n_matches")))
        ac = sh.get("auc_correlation", {})
        if "error" not in ac:
            print(_kv("AUC Pearson r", f"{ac.get('pearson_r', 0):.4f} ({ac.get('interpretation', '?')})"))
        cc = sh.get("concordance_correlation", {})
        if "error" not in cc:
            print(_kv("Concordance Pearson r", f"{cc.get('pearson_r', 0):.4f} ({cc.get('interpretation', '?')})"))
    elif sh:
        print(_subheader("Split-Half Reliability"))
        print(_kv("Error", sh.get("error")))

    # Bootstrap CIs
    boot = results.get("bootstrap", {})
    if boot:
        print(_subheader("Bootstrap 95% Confidence Intervals"))
        for key in ["auc_roc", "concordance", "score_diff"]:
            b = boot.get(key, {})
            if "error" in b:
                print(_kv(b.get("label", key), b.get("error")))
            else:
                lo = b.get("ci_95_lower", 0)
                hi = b.get("ci_95_upper", 0)
                m = b.get("mean", 0)
                print(_kv(b.get("label", key), f"{m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"))

    # Cross-match
    cm = results.get("cross_match", {})
    if cm:
        print(_subheader("Cross-Match Consistency"))
        print(_kv("Matches", cm.get("n_matches")))
        for dist_key in ["auc_dist", "concordance_dist", "mean_score_dist"]:
            d = cm.get(dist_key, {})
            if "error" in d:
                print(_kv(d.get("label", dist_key), d.get("error")))
            else:
                print(_kv(d.get("label", dist_key),
                          f"mean={d.get('mean', 0):.4f} std={d.get('std', 0):.4f} "
                          f"CV={d.get('cv', 0):.4f} outliers={d.get('outliers', 0)}"))


def print_consistency(results: dict) -> None:
    print(_header("CONSISTENCY"))

    # Dimension correlations
    dc = results.get("dimension_correlations", {})
    if dc and "error" not in dc:
        print(_subheader("Dimension Correlation Matrix (Spearman)"))
        matrix = dc.get("matrix", {})
        dims = list(matrix.keys())
        if dims:
            try:
                from tabulate import tabulate
                rows = []
                for d in dims:
                    row = [d] + [f"{matrix[d].get(d2, 0):.3f}" for d2 in dims]
                    rows.append(row)
                print(tabulate(rows, headers=[""] + dims, tablefmt="simple", stralign="right"))
            except ImportError:
                header = f"    {'':>12}" + "".join(f"{d:>12}" for d in dims)
                print(header)
                for d in dims:
                    row = f"    {d:>12}" + "".join(f"{matrix[d].get(d2, 0):>12.3f}" for d2 in dims)
                    print(row)

        flags = dc.get("flags", [])
        if flags:
            print("\n    Flags:")
            for f in flags:
                print(f"      ⚠ {f}")

    # Perturbation stability
    ps = results.get("perturbation_stability", {})
    if ps:
        print(_subheader("Rank Perturbation Stability"))
        for key, vals in ps.items():
            if "error" in vals:
                print(_kv(key, vals["error"]))
            else:
                print(_kv(f"σ={vals['sigma_meters']}m",
                          f"τ={vals['mean_tau']:.4f} ± {vals['std_tau']:.4f} ({vals['interpretation']})"))

    # Stage consistency
    sc = results.get("stage_consistency", {})
    if sc and "error" not in sc:
        print(_subheader("Stage-Wise Consistency"))
        for stage_key in ["stage_a", "stage_b"]:
            s = sc.get(stage_key, {})
            if s:
                label = s.get("label", stage_key)
                auc = s.get("auc", "?")
                conc = s.get("concordance", "?")
                print(_kv(label, f"AUC={auc}  Concordance={conc}%  (n={s.get('n_passes', 0)})"))
        deltas = sc.get("deltas", {})
        if deltas:
            consistent = sc.get("consistent")
            tag = "✓ CONSISTENT" if consistent else "⚠ DIVERGENT" if consistent is False else "?"
            print(_kv("Cross-stage deltas", deltas))
            print(_kv("Verdict", tag))


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threader Scoring Validation Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/FIFA_World_Cup_2022",
        help="Root data directory (default: data/FIFA_World_Cup_2022)",
    )
    parser.add_argument(
        "--matches", type=int, default=None,
        help="Max matches to process (default: all)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=".validation_cache",
        help="Cache directory for intermediate results",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip cache — always re-collect data",
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="Skip sensitivity analysis (slowest step)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Collect data ──────────────────────────────────────────────────
    from threader.validation.collector import (
        collect_validated_passes,
        data_summary,
        load_cache,
        save_cache,
    )

    cache_dir = Path(args.cache_dir)
    cache_tag = f"validated_passes_n{args.matches or 'all'}.pkl"
    cache_path = cache_dir / cache_tag

    records = None
    if not args.no_cache:
        records = load_cache(cache_path)
        if records:
            print(f"\n  Loaded {len(records)} cached records from {cache_path}", file=sys.stderr)

    if records is None:
        print("\n  Collecting pass data...\n", file=sys.stderr)
        records = collect_validated_passes(
            args.data_dir,
            max_matches=args.matches,
            verbose=True,
        )
        save_cache(records, cache_path)
        print(f"  Cached to {cache_path}\n", file=sys.stderr)

    if not records:
        print("ERROR: No validated passes collected. Check data directory.", file=sys.stderr)
        sys.exit(1)

    summary = data_summary(records)

    # ── Print report ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  THREADER SCORING VALIDATION REPORT")
    print("═" * 60)

    print_data_summary(summary)

    # ── 2. Baselines ─────────────────────────────────────────────────────
    print("\n  Running baselines...", file=sys.stderr, flush=True)
    from threader.validation.baselines import run_all_baselines
    baseline_results = run_all_baselines(records)
    print_baselines(baseline_results)

    # ── 3. Significance ──────────────────────────────────────────────────
    print("  Running significance tests...", file=sys.stderr, flush=True)
    from threader.validation.significance import run_significance_tests
    sig_results = run_significance_tests(records)
    print_significance(sig_results)

    # ── 4. Sensitivity ───────────────────────────────────────────────────
    if not args.skip_sensitivity:
        print("  Running sensitivity analysis...", file=sys.stderr, flush=True)
        from threader.validation.sensitivity import run_sensitivity
        sens_results = run_sensitivity(records)
        print_sensitivity(sens_results)
    else:
        print(_header("SENSITIVITY ANALYSIS"))
        print("    (skipped with --skip-sensitivity)")

    # ── 5. Repeatability ─────────────────────────────────────────────────
    print("  Running repeatability checks...", file=sys.stderr, flush=True)
    from threader.validation.repeatability import run_repeatability
    rep_results = run_repeatability(records)
    print_repeatability(rep_results)

    # ── 6. Consistency ───────────────────────────────────────────────────
    print("  Running consistency checks...", file=sys.stderr, flush=True)
    from threader.validation.consistency import run_consistency
    con_results = run_consistency(records)
    print_consistency(con_results)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'═' * 60}\n")

    # ── Quick verdict ────────────────────────────────────────────────────
    ov = sig_results.get("offensive_value", {})
    dxt = ov.get("delta_xt_spearman", {})
    lb = ov.get("lines_broken_auc", {})
    bo_conc = sig_results.get("better_option", {}).get("concordance_rate", 0)

    print("  QUICK VERDICT:")

    # Primary: offensive value
    rho = dxt.get("rho", 0)
    if abs(rho) > 0.3:
        print(f"    ✓ Pass Score correlates with ΔxT (ρ={rho:.4f}, {dxt.get('interpretation', '?')})")
    elif abs(rho) > 0.1:
        print(f"    ~ Pass Score weakly correlates with ΔxT (ρ={rho:.4f})")
    else:
        print(f"    ✗ Pass Score NOT correlated with ΔxT (ρ={rho:.4f})")

    lb_auc = lb.get("auc_roc", 0.5)
    lb_p = lb.get("p_value", 1.0)
    if lb_auc > 0.55 and lb_p < 0.01:
        print(f"    ✓ Pass Score predicts line-breaking (AUC={lb_auc:.4f}, p={lb_p:.2e})")
    elif lb_auc > 0.50:
        print(f"    ~ Pass Score weakly predicts line-breaking (AUC={lb_auc:.4f})")
    else:
        print(f"    ✗ Pass Score NOT predictive of line-breaking (AUC={lb_auc:.4f})")

    if bo_conc > 50:
        print(f"    ✓ BetterOption concordance above chance ({bo_conc:.1f}%)")
    else:
        print(f"    ✗ BetterOption concordance below 50% ({bo_conc:.1f}%)")

    pres = sig_results.get("pressure", {})
    pres_auc = pres.get("auc_roc", 0.5)
    if pres_auc > 0.55:
        print(f"    ✓ Pressure metric discriminates PFF annotations (AUC={pres_auc:.4f})")
    else:
        print(f"    ✗ Pressure metric WEAK at discriminating PFF annotations (AUC={pres_auc:.4f})")

    # Observational note
    svo = sig_results.get("score_vs_outcome", {}).get("pass_score", {})
    auc = svo.get("auc_roc", 0.5)
    print(f"    ℹ C/D outcome AUC={auc:.4f} (by design — risky passes score higher)")

    print()


if __name__ == "__main__":
    main()
