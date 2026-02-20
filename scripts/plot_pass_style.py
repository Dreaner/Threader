#!/usr/bin/env python3
"""
Pass Style Scatter Plot — Completion Probability vs Pass Score

Maps every player onto a 2D "style space":

  Y (Pass Score, 0-100)
  │  Risk-Taker     │  Elite Playmaker  │
  │  Low comp       │  High comp        │
  │  High score     │  High score       │
  ├─────────────────┼───────────────────┤
  │  Inefficient    │  Safe / Recycler  │
  │  Low comp       │  High comp        │
  │  Low score      │  Low score        │
  └─────────────────┴───────────────────→
                             X (Completion Probability)

X-axis: mean completion_probability of passes actually made
Y-axis: mean Pass Score of passes actually made  (the full formula)

Each dot = one player.  Size ∝ √(number of passes).

Usage:
    uv run python scripts/plot_pass_style.py
    uv run python scripts/plot_pass_style.py --min-passes 30
    uv run python scripts/plot_pass_style.py --save pass_style.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root (so this works from any working directory)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_PATH = _ROOT / ".validation_cache" / "validated_passes_nall.pkl"
PLAYERS_CSV = _ROOT / "data" / "FIFA_World_Cup_2022" / "players.csv"
MIN_PASSES_DEFAULT = 20

# Position → colour (football-flavoured palette)
POSITION_COLORS: dict[str, str] = {
    "GK": "#9E9E9E",  # grey
    "D": "#4472C4",   # blue
    "M": "#70AD47",   # green
    "F": "#FF6B6B",   # red
}
POSITION_LABELS: dict[str, str] = {
    "GK": "Goalkeeper",
    "D": "Defender",
    "M": "Midfielder",
    "F": "Forward",
}

# Map PFF's fine-grained positions to the 4 broad groups
_POSITION_NORMALIZE: dict[str, str] = {
    # Forwards
    "CF": "F", "RW": "F", "LW": "F",
    # Midfielders
    "CM": "M", "DM": "M", "AM": "M",
    # Defenders
    "RB": "D", "LB": "D", "RCB": "D", "LCB": "D", "CB": "D",
    # Broad groups pass through unchanged
    "GK": "GK", "D": "D", "M": "M", "F": "F",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_records(cache_path: Path) -> list:
    if not cache_path.exists():
        sys.exit(
            f"Cache not found: {cache_path}\n"
            "Run `uv run python scripts/validate_scoring.py` first to build it."
        )
    print(f"Loading cache …  ({cache_path.name})", file=sys.stderr)
    with cache_path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301


def _load_players(csv_path: Path) -> dict[int, dict[str, str]]:
    """Return {player_id: {name, position}} from players.csv."""
    df = pd.read_csv(csv_path)
    result: dict[int, dict[str, str]] = {}
    for _, row in df.iterrows():
        pid = int(row["id"])
        nickname = row.get("nickname", "")
        if pd.isna(nickname) or str(nickname).strip() == "":
            name = f"{row['firstName']} {row['lastName']}"
        else:
            name = str(nickname)
        raw_pos = str(row["positionGroupType"]) if pd.notna(row["positionGroupType"]) else "?"
        pos = _POSITION_NORMALIZE.get(raw_pos, raw_pos)
        result[pid] = {"name": name, "position": pos}
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(records: list, min_passes: int) -> pd.DataFrame:
    """One row per player with mean completion, mean Pass Score, pass count."""
    rows = [
        {
            "passer_id": r.passer_id,
            "completion": r.actual_target_completion,
            "pass_score": r.actual_target_score,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)

    agg = (
        df.groupby("passer_id")
        .agg(
            mean_completion=("completion", "mean"),
            mean_score=("pass_score", "mean"),
            pass_count=("completion", "count"),
        )
        .reset_index()
    )
    return agg[agg["pass_count"] >= min_passes].copy()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    agg: pd.DataFrame,
    players: dict[int, dict[str, str]],
    save_path: str | None,
    min_passes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # ── scatter ──────────────────────────────────────────────────────────────
    seen_positions: set[str] = set()

    for _, row in agg.iterrows():
        pid = int(row["passer_id"])
        info = players.get(pid, {"name": str(pid), "position": "?"})
        pos = info["position"]
        color = POSITION_COLORS.get(pos, "#FFA500")
        size = max(25.0, float(row["pass_count"]) ** 0.55 * 12.0)

        ax.scatter(
            row["mean_completion"],
            row["mean_score"],
            s=size,
            c=color,
            alpha=0.80,
            edgecolors="#0f1117",
            linewidths=0.5,
            zorder=3,
        )
        seen_positions.add(pos)

    # ── player labels (top 40 by pass count) ─────────────────────────────────
    label_rows = agg.nlargest(40, "pass_count")
    for _, row in label_rows.iterrows():
        pid = int(row["passer_id"])
        info = players.get(pid, {"name": str(pid), "position": "?"})
        last_name = info["name"].split()[-1]
        ax.annotate(
            last_name,
            (row["mean_completion"], row["mean_score"]),
            color="white",
            fontsize=6.5,
            alpha=0.80,
            xytext=(4, 3),
            textcoords="offset points",
        )

    # ── quadrant dividers (median of each axis) ───────────────────────────────
    x_mid = float(agg["mean_completion"].median())
    y_mid = float(agg["mean_score"].median())

    line_kw = dict(color="#555555", linewidth=0.9, linestyle="--", alpha=0.7, zorder=2)
    ax.axvline(x_mid, **line_kw)
    ax.axhline(y_mid, **line_kw)

    # ── quadrant annotations ──────────────────────────────────────────────────
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    px = (xlim[1] - xlim[0]) * 0.017
    py = (ylim[1] - ylim[0]) * 0.025

    quad_kw = dict(fontsize=8.5, color="#888888", fontstyle="italic", zorder=4)
    ax.text(x_mid - px, ylim[1] - py, "Risk-Taker", ha="right", va="top", **quad_kw)
    ax.text(xlim[1] - px, ylim[1] - py, "Elite Playmaker", ha="right", va="top", **quad_kw)
    ax.text(x_mid - px, ylim[0] + py, "Inefficient", ha="right", va="bottom", **quad_kw)
    ax.text(xlim[1] - px, ylim[0] + py, "Safe / Recycler", ha="right", va="bottom", **quad_kw)

    # ── median crosshair labels ───────────────────────────────────────────────
    ax.text(
        x_mid,
        ylim[0] + py * 0.3,
        f"median comp: {x_mid:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#666666",
        zorder=4,
    )
    ax.text(
        xlim[0] + px * 0.3,
        y_mid,
        f"median score: {y_mid:.1f}",
        ha="left",
        va="center",
        fontsize=7,
        color="#666666",
        rotation=90,
        zorder=4,
    )

    # ── legend ────────────────────────────────────────────────────────────────
    pos_order = ["GK", "D", "M", "F"]
    legend_handles = [
        mpatches.Patch(
            color=POSITION_COLORS[pos],
            label=POSITION_LABELS[pos],
        )
        for pos in pos_order
        if pos in seen_positions
    ]
    # Size reference entries
    for n_ref in [20, 60, 150]:
        s_ref = max(25.0, float(n_ref) ** 0.55 * 12.0)
        legend_handles.append(
            plt.scatter(
                [],
                [],
                s=s_ref,
                c="#888888",
                edgecolors="#0f1117",
                label=f"{n_ref} passes",
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.25,
        labelcolor="white",
        facecolor="#1a1a2e",
        edgecolor="#333333",
    )

    # ── axes styling ──────────────────────────────────────────────────────────
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_xlabel("Mean Completion Probability  →  Safer passes", fontsize=11, color="white", labelpad=8)
    ax.set_ylabel("Mean Pass Score  (0–100)  →  Higher value", fontsize=11, color="white", labelpad=8)
    ax.set_title(
        "Player Pass Style — FIFA World Cup 2022\n"
        "Where does your average pass sit on the Safety ↔ Value spectrum?",
        fontsize=13,
        color="white",
        pad=12,
    )

    # Subtle grid
    ax.grid(visible=True, color="#222222", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Footer
    n_players = len(agg)
    n_passes = int(agg["pass_count"].sum())
    fig.text(
        0.01,
        0.005,
        f"{n_players} players  ·  {n_passes:,} passes  ·  min {min_passes} passes threshold",
        fontsize=7,
        color="#555555",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass Style scatter plot (Completion vs Pass Score)")
    parser.add_argument(
        "--min-passes",
        type=int,
        default=MIN_PASSES_DEFAULT,
        metavar="N",
        help=f"Minimum passes per player to include (default: {MIN_PASSES_DEFAULT})",
    )
    parser.add_argument(
        "--position",
        metavar="POS",
        help="Filter to a specific position group: GK, D, M, F (default: all)",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save plot to FILE (.png / .pdf / .svg) instead of displaying it",
    )
    args = parser.parse_args()

    records = _load_records(CACHE_PATH)
    print(f"  {len(records):,} validated passes loaded", file=sys.stderr)

    players = _load_players(PLAYERS_CSV)
    print(f"  {len(players):,} players loaded", file=sys.stderr)

    agg = _aggregate(records, min_passes=args.min_passes)

    if args.position:
        pos_filter = args.position.upper()
        ids_in_pos = {
            pid for pid, info in players.items()
            if info["position"] == pos_filter
        }
        agg = agg[agg["passer_id"].isin(ids_in_pos)].copy()
        print(f"  Filtered to position '{pos_filter}': {len(agg)} players", file=sys.stderr)
    else:
        print(f"  {len(agg)} players with ≥ {args.min_passes} passes", file=sys.stderr)

    _plot(agg, players, save_path=args.save, min_passes=args.min_passes)


if __name__ == "__main__":
    main()
