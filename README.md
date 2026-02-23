# PitchEcho

**AlphaGo-inspired football pass analysis — find the optimal pass target in any match snapshot.**

Just as AlphaGo evaluates every possible move on the board, Threader freezes a moment in a football match and asks: *who is the best player to pass to?*

```
Scenario: De Bruyne receives the ball at the center circle
  1st choice → Haaland       Pass Score: 78
  2nd choice → Foden          Pass Score: 76
  3rd choice → Rodri          Pass Score: 61
```

Built on **FIFA World Cup 2022** data (64 matches, PFF FC format). Interactive Dash web app with Plotly pitch visualizations and tracking-data animations.

---

## Quick Start

**Requirements:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)

```bash
# Clone & install
git clone https://github.com/your-username/Threader.git
cd Threader
uv sync

# Run the web app
uv run threader
# → http://127.0.0.1:8050
```

### Optional extras

```bash
uv sync --extra dev          # pytest + ruff
uv sync --extra validation   # scipy + tabulate (for validation scripts)
uv sync --extra learning     # scikit-learn + xgboost (for ML weight training)
uv sync --extra deploy       # gunicorn (for production deployment)
```

---

## How It Works

### The AlphaGo Analogy

| AlphaGo | Threader |
|---------|----------|
| Board state (19×19 grid + stones) | Match snapshot (22 player positions) |
| Move selection | Pass target selection |
| Win rate evaluation | **Pass Score** evaluation (0–100) |
| 1st / 2nd / 3rd choice ranking | Pass option ranking |

### Pass Score — The Core Metric

Every potential pass target is evaluated across **5 dimensions**, combined into a single 0–100 score:

```
Pass Score = (
    completion × zone_value × 4.27     # Expected value
  + penetration × 1.00                 # Forward progress bonus
  + space × 0.0001                     # Space tiebreaker
) × (1 − pressure/10 × 0.01)          # Pressure multiplier
  × 100
```

| Dimension | What it measures | Range |
|-----------|------------------|-------|
| **Completion Probability** | Can this pass reach its target? (distance decay + passing lane obstruction) | 0 – 1 |
| **Zone Value (xT)** | How dangerous is the receiver's position? (Expected Threat grid) | 0 – 0.45 |
| **Receiving Pressure** | Will the receiver be under pressure? (3 nearest defenders) | 0 – 10 |
| **Space Available** | How much room does the receiver have? (capped at 15m) | 0 – 15m |
| **Penetration Score** | How much forward progress does this pass achieve? (distance + defenders bypassed) | -0.3 – 1.0 |

Weights were optimized via `scipy.optimize.differential_evolution` maximizing Spearman ρ(Pass Score, ΔxT) across all 64 World Cup matches (ρ = 0.654, AUC = 0.768).

---

## Web App

The interactive Dash application lets you explore every pass in the tournament:

- **Match selector** — browse all 64 FIFA World Cup 2022 matches
- **Pass event browser** — pick any pass event with period, clock, passer → target, and outcome
- **Interactive pitch** — Plotly rendering with players, ball, and ranked pass arrows (gold / silver / bronze)
- **Analysis cards** — ranked pass options with 5-dimension breakdowns
- **Tracking animation** — "Play Pass" replays the actual pass sequence from 25fps tracking data
- **Click-to-highlight** — click any option card to highlight that pass on the pitch

### Production Deployment

```bash
uv sync --extra deploy
uv run gunicorn threader.app:server -b 0.0.0.0:8050
```

---

## Project Structure

```
src/threader/
├── app.py                  # Dash web application
├── models.py               # Core data models (Player, Snapshot, PassOption, etc.)
├── analysis/
│   └── analyzer.py         # Pass analysis engine (analyze_snapshot / analyze_pass_event)
├── scoring/
│   ├── pass_score.py       # Master Pass Score formula + ScoringWeights
│   ├── completion.py       # Completion probability (distance + lane blocking)
│   ├── zone_value.py       # xT Expected Threat grid (12×8)
│   ├── pressure.py         # Receiving pressure (weighted nearest defenders)
│   ├── space.py            # Space available (distance to nearest opponent)
│   └── penetration.py      # Penetration score (forward gain + defenders bypassed)
├── geometry/
│   ├── distance.py         # Euclidean distance, point-to-segment distance
│   └── passing_lane.py     # Passing lane obstruction detection
├── data/
│   ├── events.py           # PFF event data JSON parser
│   ├── metadata.py         # Match metadata + roster loading
│   ├── tracking.py         # Tracking data loader (kloppy, .jsonl.bz2)
│   └── tracking_frames.py  # Fast animation frames (Parquet primary, ~20-50ms)
├── viz/
│   ├── plotly_pitch.py     # Plotly pitch rendering
│   ├── plotly_passes.py    # Plotly pass analysis visualization
│   ├── plotly_animation.py # Plotly tracking-data animation builder
│   ├── pitch.py            # Matplotlib pitch rendering
│   └── passes.py           # Matplotlib pass visualization
├── validation/
│   ├── collector.py        # Batch analysis + ground truth collection
│   ├── baselines.py        # Baseline comparison models
│   ├── significance.py     # Statistical significance tests
│   ├── sensitivity.py      # Weight perturbation + ablation analysis
│   ├── consistency.py      # Internal consistency checks
│   └── repeatability.py    # Cross-match reliability + bootstrap CI
└── learning/
    ├── dataset.py          # Feature matrix preparation
    ├── models.py           # Linear / Ridge / XGBoost model layers
    ├── evaluate.py         # Re-scoring with learned weights
    └── interpret.py        # Coefficient → ScoringWeights mapping

scripts/
├── validate_scoring.py     # Full validation report (CLI)
├── optimize_weights.py     # Weight optimization (differential evolution)
├── train_scoring_model.py  # ML weight learning pipeline
└── convert_to_parquet.py   # Tracking data → Parquet conversion
```

---

## Data

Uses the [PFF FC FIFA World Cup 2022 open dataset](https://github.com/ProFootballFocus/pff-fc-data):

- **64 matches** with event data, metadata, rosters, and tracking data
- **Event Data** — JSON files with pass events + embedded 22-player freeze-frames
- **Tracking Data** — `.jsonl.bz2` files at 25fps (convertible to Parquet for fast loading)
- **Coordinate system** — PFF center-origin: x ∈ [-52.5, 52.5], y ∈ [-34, 34]

### Converting Tracking Data to Parquet (recommended)

```bash
uv run python scripts/convert_to_parquet.py
```

This enables ~20-50ms animation frame queries via PyArrow predicate pushdown, vs streaming `.jsonl.bz2`.

---

## Scripts

### Validate the scoring system

```bash
uv run python scripts/validate_scoring.py
```

Produces a comprehensive report: data summary, baseline comparisons, statistical significance (Spearman ρ, AUC), sensitivity analysis, and cross-match repeatability.

### Optimize scoring weights

```bash
uv run python scripts/optimize_weights.py
```

Uses differential evolution to find weights that maximize Spearman ρ(Pass Score, ΔxT).

### Train ML models

```bash
uv run python scripts/train_scoring_model.py
```

Trains Linear / Ridge / XGBoost models to learn optimal dimension weights from data.

---

## Testing

```bash
uv run pytest
```

Tests cover the Pass Score formula, completion probability, geometry algorithms, and analyzer integration.

---

## Design Philosophy

> **A pure tactical optimization AI — only evaluating the optimal solution at the current instant.**

**What Threader does:**
- Evaluates pass targets (who to pass to)
- Static snapshot analysis (freeze a moment, find the optimal solution)
- Multi-dimensional scoring with explainable breakdowns
- Clear 1st / 2nd / 3rd choice ranking

**What Threader does NOT do (by design):**
- No match context (score, time, leading/trailing)
- No risk preference (conservative vs aggressive)
- No chain reaction prediction (what happens after the pass)
- No pass trajectory analysis (ball physics)

See [CLAUDE.md](CLAUDE.md) for the full design document.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Web framework | Dash + Plotly |
| Data processing | pandas, NumPy, PyArrow |
| Tracking data | kloppy |
| Optimization | SciPy (differential evolution) |
| ML (optional) | scikit-learn, XGBoost |
| Testing | pytest |
| Linting | Ruff |
| Packaging | hatchling + uv |

---

## License

[GPL-3.0-or-later](LICENSE)
