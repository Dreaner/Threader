# PitchEcho

**A Python toolkit for football spatial data analysis.**

PitchEcho provides tools for loading, analyzing, and visualizing football positional data — built around the PFF FC open dataset (FIFA World Cup 2022, 64 matches).

The library has two distinct layers:

- **Public API** (`pip install pitch-echo`) — stable tools for data loading, pitch visualization, and pass network analysis.
- **Research modules** (`pitch_echo.research.*`) — experimental metrics under active development, not part of the public API.

---

## Installation

```bash
pip install pitch-echo
```

**Optional extras:**

```bash
pip install "pitch-echo[viz]"       # matplotlib + plotly visualizations
pip install "pitch-echo[research]"  # scipy, scikit-learn, xgboost (research scripts)
pip install "pitch-echo[app]"       # Dash interactive web app
```

**Requirements:** Python ≥ 3.12

---

## Quick Start

### Data loading

```python
from pitch_echo import extract_pass_events

events = extract_pass_events("data/FIFA_World_Cup_2022/Event Data/3812.json")
print(f"{len(events)} pass events loaded")
print(events[0])  # PassEvent with snapshot, passer_id, target_id, outcome, …
```

### Pitch visualization

```python
from pitch_echo import Pitch

pitch = Pitch(backend="plotly")
fig = pitch.draw()
fig.show()
```

### Pass network

```python
from pitch_echo import extract_pass_events, build_pass_network, compute_metrics, Pitch

events = extract_pass_events("data/FIFA_World_Cup_2022/Event Data/3812.json")

network = build_pass_network(events, team_id=3812)
metrics = compute_metrics(network)

# Graph metrics per player
for pid, pm in metrics.player_metrics.items():
    print(f"{pid}: degree={pm.degree_centrality:.3f}  pagerank={pm.pagerank:.3f}")

# Render on pitch
pitch = Pitch(backend="plotly")
fig = pitch.pass_network(network, metrics=metrics)
fig.show()
```

---

## Research: Pass Value

The `pitch_echo.research` sub-package contains experimental work that is not yet stable enough for the public API.

> **⚠️ Research modules may change without notice between versions.**

### Pass Score — AlphaGo-inspired pass target evaluation

Inspired by AlphaGo's move evaluation, Pass Score freezes a match snapshot and asks: *who is the best player to pass to?*

```
Scenario: De Bruyne receives the ball at the center circle
  1st choice → Haaland       Pass Score: 78
  2nd choice → Foden          Pass Score: 76
  3rd choice → Rodri          Pass Score: 61
```

```python
from pitch_echo import extract_pass_events
from pitch_echo.research.pass_value import analyze_pass_event

events = extract_pass_events("data/FIFA_World_Cup_2022/Event Data/3812.json")

result = analyze_pass_event(events[42])
print(f"Passer: {result.passer.name}")
for i, opt in enumerate(result.ranked_options[:3], 1):
    print(
        f"  #{i} {opt.target.name:<20} "
        f"Score={opt.pass_score:.1f}  "
        f"xT={opt.zone_value:.3f}  "
        f"Completion={opt.completion_probability:.2f}"
    )
```

### Pass Score formula

Every potential receiver is scored across **5 dimensions**:

```
Pass Score = (
    completion × zone_value × 3.01     # Expected value (xT-amplified)
  + penetration × 0.46                 # Forward progress
  + space × 0.0001                     # Space tiebreaker (capped 15m)
) × (1 − pressure/10 × 0.01)          # Pressure multiplier
  × 100
```

| Dimension | What it measures | Range |
|-----------|-----------------|-------|
| **Completion Probability** | Can this pass reach its target? (distance decay + lane obstruction) | 0 – 1 |
| **Zone Value (xT)** | How dangerous is the receiver's position? (Expected Threat 12×8 grid) | 0 – 0.45 |
| **Receiving Pressure** | Will the receiver be under pressure? (3 nearest defenders, weighted) | 0 – 10 |
| **Space Available** | How much room does the receiver have? (nearest opponent, capped 15m) | 0 – 15m |
| **Penetration Score** | How much forward progress does this pass achieve? (x-gain + defenders bypassed) | -0.3 – 1.0 |

Weights were optimized via `scipy.optimize.differential_evolution` maximizing Spearman ρ(Pass Score, ΔxT) across all 64 World Cup matches (ρ = 0.680, AUC = 0.777).

### Visualize on pitch

```python
from pitch_echo import Pitch, extract_pass_events
from pitch_echo.research.pass_value import analyze_pass_event

events = extract_pass_events("data/FIFA_World_Cup_2022/Event Data/3812.json")
result = analyze_pass_event(events[42])

pitch = Pitch(backend="plotly")
fig = pitch.pass_options(result, top_n=3)
fig.show()
```

---

## Project Structure

```
src/pitch_echo/
│
├── __init__.py             # Public API only
│
├── core/                   # ✅ Public — core data models
│   ├── models.py           #   Player, Snapshot, BallPosition
│   └── types.py
│
├── data/                   # ✅ Public — data loading
│   └── pff/
│       ├── events.py       #   PassEvent, extract_pass_events
│       ├── metadata.py     #   load_match_info
│       ├── tracking.py     #   kloppy-based tracking loader
│       └── tracking_frames.py  # Fast Parquet animation frames
│
├── geometry/               # ✅ Public — geometry utilities
│   ├── distance.py
│   └── passing_lane.py
│
├── network/                # ✅ Public — pass network analysis
│   ├── builder.py          #   build_pass_network
│   ├── metrics.py          #   compute_metrics
│   └── models.py           #   PassNetwork, PassEdge, PlayerNode, NetworkMetrics
│
├── viz/                    # ✅ Public — visualization
│   ├── plotly_pitch.py
│   ├── plotly_passes.py
│   ├── plotly_network.py
│   ├── plotly_animation.py
│   ├── plotly_animation_3d.py
│   ├── plotly_pitch_3d.py
│   ├── mpl_pitch.py
│   └── mpl_passes.py
│
├── pitch.py                # ✅ Public — Pitch high-level API
│
└── research/               # 🔬 Internal research (not public API)
    ├── pass_value/         #   Pass Score metric
    │   ├── analysis/
    │   │   ├── analyzer.py #     analyze_snapshot, analyze_pass_event
    │   │   └── models.py   #     AnalysisResult, PassOption, ScoringWeights
    │   └── scoring/
    │       ├── pass_score.py
    │       ├── completion.py
    │       ├── zone_value.py
    │       ├── pressure.py
    │       ├── space.py
    │       └── penetration.py
    ├── validation/         #   Statistical validation framework
    │   ├── collector.py
    │   ├── significance.py
    │   ├── repeatability.py
    │   ├── baselines.py
    │   ├── consistency.py
    │   └── sensitivity.py
    └── learning/           #   ML-based weight optimisation
        ├── dataset.py
        ├── models.py
        ├── evaluate.py
        └── interpret.py

app/                        # Interactive Dash web application
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

## Research Scripts

> Requires `pip install "pitch-echo[research]"`

### Validate Pass Score

```bash
python scripts/validate_scoring.py
```

Produces a comprehensive report: baseline comparisons, Spearman ρ, lines-broken AUC, sensitivity analysis, and cross-match repeatability.

### Optimize scoring weights

```bash
python scripts/optimize_weights.py
```

Uses `scipy.optimize.differential_evolution` to find weights maximizing Spearman ρ(Pass Score, ΔxT).

### Train ML models

```bash
python scripts/train_scoring_model.py
```

Trains Linear / Ridge / XGBoost models on the 5 scoring dimensions to learn data-driven weights.

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
- **Interactive pitch** — players, ball, and ranked pass arrows (gold / silver / bronze)
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

**What Pass Score evaluates:**
- Pass targets (who to pass to, not how)
- Static snapshot analysis — freeze a moment, find the optimal solution
- Multi-dimensional scoring with fully explainable breakdowns
- Clear 1st / 2nd / 3rd choice ranking

**Out of scope (by design):**
- Match context (score, time remaining)
- Risk preference (conservative vs aggressive style)
- Chain reaction prediction (what happens after the pass)
- Pass trajectory (ball physics, spin, arc)

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
