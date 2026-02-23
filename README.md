# PitchEcho

**AlphaGo-inspired football pass analysis — find the optimal pass target in any match snapshot.**

Just as AlphaGo evaluates every possible move on the board, PitchEcho freezes a moment in a football match and asks: *who is the best player to pass to?*

```
Scenario: De Bruyne receives the ball at the center circle
  1st choice → Haaland       Pass Score: 78
  2nd choice → Foden          Pass Score: 76
  3rd choice → Rodri          Pass Score: 61
```

Built on **FIFA World Cup 2022** data (64 matches, PFF FC format).

---

## Installation

```bash
pip install pitch-echo
```

**Optional extras:**

```bash
pip install "pitch-echo[viz]"        # matplotlib + plotly visualizations
pip install "pitch-echo[validation]" # scipy + tabulate (validation scripts)
pip install "pitch-echo[learning]"   # scikit-learn + xgboost (ML weight training)
```

**Requirements:** Python ≥ 3.12

---

## Quick Start

```python
from pitch_echo import (
    analyze_pass_event,
    analyze_snapshot,
    build_pass_network,
    compute_metrics,
    load_pff,
)

# Load pass events from PFF data
events = load_pff("path/to/events.json", metadata, roster)

# Analyze a single pass event
result = analyze_pass_event(events[0])
for option in result.ranked_options:
    print(f"{option.player.name}: Pass Score {option.pass_score:.1f}")

# Build a pass network for one team across the match
network = build_pass_network(events, team_id="home")
metrics = compute_metrics(network)
print(f"Network density: {metrics.density:.3f}")
```

---

## How It Works

### The AlphaGo Analogy

| AlphaGo | PitchEcho |
|---------|-----------|
| Board state (19×19 grid + stones) | Match snapshot (22 player positions) |
| Move selection | Pass target selection |
| Win rate evaluation | **Pass Score** evaluation (0–100) |
| 1st / 2nd / 3rd choice ranking | Pass option ranking |

### Pass Score — The Core Metric

Every potential pass target is evaluated across **5 dimensions**, combined into a single 0–100 score:

```
Pass Score = (
    completion × zone_value × 1.5     # Expected value (amplified)
  + penetration × 0.20                # Forward progress bonus
  + space × 0.001                     # Space tiebreaker (capped at 15m)
) × (1 − pressure/10 × 0.20)         # Pressure as multiplier
  × 100
```

| Dimension | What it measures | Range |
|-----------|-----------------|-------|
| **Completion Probability** | Can this pass reach its target? (distance decay + lane obstruction) | 0 – 1 |
| **Zone Value (xT)** | How dangerous is the receiver's position? (Expected Threat 12×8 grid) | 0 – 0.45 |
| **Receiving Pressure** | Will the receiver be under pressure? (3 nearest defenders, weighted) | 0 – 10 |
| **Space Available** | How much room does the receiver have? (nearest opponent, capped at 15m) | 0 – 15m |
| **Penetration Score** | How much forward progress does this pass achieve? (distance + defenders bypassed) | -0.3 – 1.0 |

Weights were optimized via `scipy.optimize.differential_evolution` maximizing Spearman ρ(Pass Score, ΔxT) across all 64 World Cup matches (ρ = 0.654, AUC = 0.768).

### Pass Network Analysis

Beyond individual pass decisions, PitchEcho can build a **pass network** for a team across a match — mapping the passing structure and identifying key players.

```python
network = build_pass_network(events, team_id="home", completed_only=True)
metrics = compute_metrics(network)

# Graph-theory metrics for each player
for player_id, pm in metrics.player_metrics.items():
    print(f"{player_id}: centrality={pm.degree_centrality:.3f}, pagerank={pm.pagerank:.3f}")
```

| Network Metric | What it measures |
|---------------|-----------------|
| **Density** | How evenly distributed are passing connections? |
| **Degree Centrality** | Who has the most unique passing partners? |
| **Betweenness Centrality** | Who is the passing "middleman"? |
| **PageRank** | Who receives passes from important players? |

---

## Project Structure

```
src/pitch_echo/
├── core/
│   ├── models.py           # Core data models (Player, Snapshot, BallPosition)
│   └── types.py            # Type aliases
├── analysis/
│   ├── analyzer.py         # Pass analysis engine (analyze_snapshot / analyze_pass_event)
│   └── models.py           # AnalysisResult, PassOption, ScoringWeights
├── scoring/
│   ├── pass_score.py       # Master Pass Score formula
│   ├── completion.py       # Completion probability (distance + lane blocking)
│   ├── zone_value.py       # xT Expected Threat grid (12×8)
│   ├── pressure.py         # Receiving pressure (weighted nearest defenders)
│   ├── space.py            # Space available (nearest opponent, capped 15m)
│   └── penetration.py      # Penetration score (forward gain + defenders bypassed)
├── network/
│   ├── builder.py          # Pass network construction
│   ├── metrics.py          # Graph metrics (density, centrality, PageRank)
│   └── models.py           # PassNetwork, PassEdge, PlayerNode, NetworkMetrics
├── geometry/
│   ├── distance.py         # Euclidean + point-to-segment distance
│   └── passing_lane.py     # Passing lane obstruction detection
├── data/
│   ├── base.py             # Base data loading interface
│   └── pff/
│       ├── events.py       # PFF event data JSON parser
│       ├── metadata.py     # Match metadata + roster loading
│       ├── tracking.py     # Tracking data loader (kloppy, .jsonl.bz2)
│       └── tracking_frames.py  # Fast animation frames (Parquet, ~20-50ms)
├── viz/
│   ├── plotly_pitch.py     # Plotly pitch rendering
│   ├── plotly_passes.py    # Plotly pass analysis visualization
│   ├── plotly_network.py   # Plotly pass network visualization
│   ├── plotly_animation.py # Plotly tracking-data animation
│   ├── plotly_animation_3d.py
│   ├── plotly_pitch_3d.py
│   ├── mpl_pitch.py        # Matplotlib pitch rendering
│   └── mpl_passes.py       # Matplotlib pass visualization
├── validation/             # Scoring system validation tools
└── learning/               # ML-based weight learning pipeline

app/                        # Interactive Dash web application (not part of the PyPI package)
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
python scripts/convert_to_parquet.py
```

Enables ~20-50ms animation frame queries via PyArrow predicate pushdown.

---

## Scripts

### Validate the scoring system

```bash
python scripts/validate_scoring.py
```

Produces a comprehensive report: baseline comparisons, Spearman ρ, AUC, sensitivity analysis, and cross-match repeatability.

### Optimize scoring weights

```bash
python scripts/optimize_weights.py
```

Uses differential evolution to find weights maximizing Spearman ρ(Pass Score, ΔxT).

### Train ML models

```bash
python scripts/train_scoring_model.py
```

Trains Linear / Ridge / XGBoost models to learn optimal dimension weights from data.

---

## Web App

An interactive Dash application is included in the `app/` directory (requires cloning the repository):

```bash
git clone https://github.com/Dreaner/PitchEcho.git
cd PitchEcho
uv sync --extra app
uv run python app/app.py
# → http://127.0.0.1:8050
```

Features:
- **Match selector** — browse all 64 FIFA World Cup 2022 matches
- **Pass event browser** — pick any pass event by period, clock, passer, and target
- **Interactive pitch** — Plotly rendering with players, ball, and ranked pass arrows (gold / silver / bronze)
- **Analysis cards** — ranked pass options with 5-dimension breakdowns
- **Tracking animation** — "Play Pass" replays the actual pass sequence from 25fps tracking data
- **Pass network tab** — interactive team pass network with graph metrics

---

## Testing

```bash
uv run pytest
```

Tests cover the Pass Score formula, completion probability, geometry algorithms, and analyzer integration.

---

## Design Philosophy

> **A pure tactical optimization AI — only evaluating the optimal solution at the current instant.**

**What PitchEcho does:**
- Evaluates pass targets (who to pass to, not how)
- Static snapshot analysis — freeze a moment, find the optimal solution
- Multi-dimensional scoring with fully explainable breakdowns
- Clear 1st / 2nd / 3rd choice ranking

**What PitchEcho does NOT do (by design):**
- No match context (score, time remaining, leading/trailing)
- No risk preference (conservative vs aggressive)
- No chain reaction prediction (what happens after the pass)
- No pass trajectory analysis (ball physics, spin, arc)

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
