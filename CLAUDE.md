# CLAUDE.md - Threader Development Design Document

> This document records the core design philosophy, technical decisions, and scoring system of the Threader project, serving as a development guide.

---

## 🎯 Core Vision

### Project Positioning

**Applying AlphaGo's decision evaluation framework to football pass analysis.**

Just as AlphaGo shows during game review "this move was the second choice; the best move would have been at Tengen," Threader aims to do the same for football passing scenarios:

```
Scenario: De Bruyne receives the ball at the center circle
1st choice: Pass behind Haaland (Pass Score: 78)
2nd choice: Pass to Foden on the wing (Pass Score: 76)
3rd choice: Pass back to Rodri (Pass Score: 61)
```

### AlphaGo vs Football Passing — Mapping

| AlphaGo | Threader |
|---------|----------|
| Board state (19×19 grid + piece positions) | Match snapshot (22 player positions) |
| Move selection | Pass target selection |
| Win rate evaluation | Pass Score evaluation |
| 1st/2nd/3rd choice | Pass option ranking |
| Pure optimization (emotion-free) | Pure tactical optimization (ignoring score/time) |


---

## 🧭 Design Principles

### What We Do

✅ **Evaluate pass targets**: Who is the best player to pass to? (not the exact pass trajectory)  
✅ **Static snapshot analysis**: Freeze a moment and find the optimal solution (not predicting the next 3 seconds)  
✅ **Multi-dimensional scoring**: Completion probability, zone value, pressure, penetration, etc.  
✅ **Clear ranking**: Provide a definitive 1st/2nd/3rd choice ranking  

### What We Explicitly Do NOT Do (at least in MVP)

❌ **No match context**: Score, remaining time, leading/trailing  
❌ **No risk preference**: Conservative vs aggressive subjective choices  
❌ **No chain reaction prediction**: What happens after the pass  
❌ **No pass trajectory analysis**: Ball speed, spin, arc, and other physics parameters  

### Core Philosophy

> **"A pure tactical optimization AI — only evaluating the optimal solution at the current instant"**

- Pass target first: The primary question is "who to pass to"
- Explainability: Every score can be decomposed into specific dimensions
- Start simple: Begin with static analysis, evolve in the future

---

## 📊 Core Metric: Pass Score

### Why "Pass Score"?

**Alternatives considered:**
- `Win Rate`: AlphaGo's terminology, but our formula isn't a pure "goal probability"
- `Thread Score`: Matches the project name, but overemphasizes through-balls
- `Decision Score`: Accurate but doesn't convey "passing"
- **`Pass Score`**: ✅ Concise, honest, neutral

**Rationale:**
1. Honesty — We are a composite score, not a pure probability model
2. Flexibility — Weights can be adjusted in the future without being constrained by the name
3. Intuitiveness — Everyone can understand "pass score"

### Pass Score Formula

```python
Pass Score = (
    completion_probability × zone_value × 1.5 +    # Expected value (amplified)
    penetration_score × 0.20 +                     # Penetration bonus/drag
    space_available × 0.001                        # Space bonus (capped at 15m)
) × (1.0 − pressure/10 × 0.20)                    # Pressure as multiplier
  × 100
```

**Range:** 0–100

**Key changes (v1.1):**
- `zone_value` amplified ×1.5 to widen gap between attacking and defensive positions
- Pressure switched from additive penalty to **multiplicative** scaling (max 20% reduction)
- Penetration now returns negative for backward passes (mild drag)
- Space capped at 15m upstream to prevent isolated players (e.g. GK) from getting excessive bonus

---

## 🔧 Five Scoring Dimensions

### 1. Completion Probability

**Goal:** Can this pass reach its target?

**Factors:**
- Distance decay
- Defender blocking (passing lane obstruction)

**Calculation logic:**

```python
# Base probability (distance-based)
if distance < 10m:  base_prob = 0.95
elif distance < 20m: base_prob = 0.85
elif distance < 30m: base_prob = 0.70
else: base_prob = max(0.40, 0.90 - distance × 0.015)

# Passing lane blocking penalty
for defender in passing_lane:
    if dist_to_line < 1.5m:  blocking += 0.4  # Severe blocking
    elif dist_to_line < 3.0m: blocking += 0.2  # Moderate blocking
    elif dist_to_line < 5.0m: blocking += 0.1  # Minor blocking

# Final completion probability
completion = base_prob × (1 - min(0.8, blocking))
```

**Key Algorithm: Point-to-Line-Segment Distance**

```python
def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate the shortest distance from a defender to the passing line"""
    # Vector projection method
    dx, dy = x2 - x1, y2 - y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx² + dy²)
    t = clamp(t, 0, 1)  # Constrain to the line segment
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return sqrt((px - closest_x)² + (py - closest_y)²)
```

---

### 2. Zone Value (xT — Expected Threat)

**Goal:** How dangerous is the receiver's position?

**Implementation:** xT (Expected Threat) grid  
**Data structure:** 12×8 grid mapped to actual pitch (105m × 68m)  
**Value range:** 0 (defensive third) → 0.45 (central penalty area)

**Coordinate conversion (PFF → xT grid):**

```python
# PFF center-origin → pitch coordinates (0,0 at bottom-left)
pitch_x = pff_x + pitch_length / 2   # pitch_length from metadata (typically 105m)
pitch_y = pff_y + pitch_width / 2     # pitch_width from metadata (typically 68m)

# Grid index lookup
col = clamp(int(pitch_x / pitch_length * 12), 0, 11)  # 12 columns
row = clamp(int(pitch_y / pitch_width * 8), 0, 7)       # 8 rows
zone_value = XT_GRID[col][row]
```

**Simplified xT grid (12 columns × 8 rows):**

```python
# Columns map to x-axis (0→105m, left-to-right = defense→attack)
# Rows map to y-axis (0→68m, bottom-to-top)
XT_GRID[col][row]  # col: 0-11, row: 0-7
```

**Future improvements:**
- Use a more granular xT model (e.g., Karun Singh's original version)
- ~~Account for the team's attacking direction~~ ✅ Done (v1.1): Uses `homeTeamStartLeft` from metadata + period to determine `attack_direction` (+1.0 or -1.0). The x coordinate is flipped in `zone_value()` and `penetration_score()` when the team attacks left.

---

### 3. Receiving Pressure

**Goal:** Will the receiver be under pressure?

**Method:** Weighted distance from the 3 nearest defenders  
**Range:** 0 (no pressure) → 10 (extreme pressure)

```python
pressure = 0
nearest_3 = sorted(defenders, key=lambda d: distance(receiver, d))[:3]

for i, defender in enumerate(nearest_3):
    weight = 1.0 / (i + 1)  # 1st nearest: weight 1.0, 2nd: 0.5, 3rd: 0.33
    dist = distance(receiver, defender)
    
    if dist < 2m:  pressure += 5.0 × weight
    elif dist < 5m:  pressure += 3.0 × weight
    elif dist < 10m: pressure += 1.0 × weight
    else: pressure += 0.3 × weight

pressure = min(10.0, pressure)
```

**Design rationale:**
- Only the 3 nearest — avoids noise from distant defenders
- Weighting ensures the closest defender has the greatest impact
- Distance thresholds are based on real-match pressing distances

---

### 4. Space Available

**Goal:** How much room does the receiver have after receiving the ball?

**Calculation:** Distance to the nearest opponent (in meters), capped at 15m

```python
opponents = [p for p in all_players if p.team != receiver.team]
space = min(15.0, min(distance(receiver, opp) for opp in opponents))
```

**Role:**
- More space = more options after receiving the ball
- Small weight in Pass Score (×0.001), acts as a fine-tuning factor
- Capped at 15m to prevent isolated/deep players (GK) from getting excessive bonus

---

### 5. Penetration Score

**Goal:** How much forward progress does this pass achieve?

**Two sub-dimensions:**

**A. Forward distance:**
```python
x_gain = receiver.x - passer.x

if x_gain <= 0:
    # Mild backward drag: max -0.3 at 40m back
    forward_score = max(-0.3, x_gain / 40.0)
else:
    forward_score = min(1.0, x_gain / 20.0)  # 20m forward = max score
```

**B. Defenders bypassed:**
```python
defenders_passed = 0
for defender in defenders:
    if passer.x < defender.x < receiver.x:
        defenders_passed += 1

penetration_bonus = min(0.5, defenders_passed × 0.15)
```

**Final penetration score:**
```python
penetration = forward_score + penetration_bonus
penetration = max(-0.3, min(1.0, penetration))  # Range: [-0.3, 1.0]
```

**Design rationale:**
- Forward passes are inherently more valuable (nature of attacking play)
- Bypassing defenders = disrupting the opponent's defensive shape
- Cap at 1.0 to avoid over-rewarding
- Backward passes receive a mild drag (down to -0.3) to discourage backpasses to GK etc.

---

## 🎯 Key Technical Decisions

### Decision 1: Pass Target vs Pass Trajectory

**Question:** Evaluate "who to pass to" or "how to pass"?

**Choice:** Pass target

**Rationale:**
- ✅ Intuitive — The core purpose of a pass is to find a teammate
- ✅ Low data requirements — Only player positions needed, not ball physics
- ✅ MVP-friendly — Quick idea validation
- ✅ Extensible — Trajectory evaluation can be added later

---

### Decision 2: Static Analysis vs Dynamic Prediction

**Question:** Analyze only the current instant, or predict post-pass developments?

**Choice:** Static analysis

**Rationale:**
- ✅ Technically feasible — No need to train multi-agent RL models
- ✅ Already valuable — xT implicitly captures "future potential of a position"
- ✅ Explainable — Easier to communicate scoring logic to users

**Trade-offs:**
- ❌ Cannot accurately predict "what happens after the pass"
- ❌ Cannot capture the importance of "pass timing"

**Future possibilities:**
- Use historical data to estimate "average outcome in similar situations"
- Train a lightweight "next-step predictor"

---

### Decision 3: Subjective Factors

**Question:** Should we consider score, time, risk preference?

**Choice:** Completely ignored in MVP

**Rationale:**
- Maintain the "pure tactical optimization" positioning
- Avoid introducing hard-to-quantify subjective parameters
- Can be added as an "advanced mode" in the future

**Comparison:**
```python
# Current: pure optimization
best_pass = max(all_options, key=lambda x: x.pass_score)

# Future possibility: context-aware
if leading_by_2_goals and time_left < 5_min:
    # Adjust weights toward safer passes
    best_pass = prioritize_safety(all_options)
```

---

### Decision 4: Score Range

**Question:** Use 0–1 probability values or 0–100 scores?

**Choice:** 0–100

**Rationale:**
- ✅ More intuitive (78 points vs 0.78)
- ✅ Familiar scoring convention (Chess Rating, exams, etc.)
- ✅ Avoids misleading implications (probability values suggest "precise prediction")

---

### Decision 5: Data Source

**Question:** Event Data (pass events) or Tracking Data (player positions)?

**Choice:** Both — PFF FC format (FIFA World Cup 2022)

**Data available:**
- 64 matches from FIFA World Cup 2022 (PFF FC open data)
- Event Data: JSON files with pass events, player snapshots (22 players x/y), ball position
- Tracking Data: `.jsonl.bz2` files with continuous 25fps positional data
- Metadata + Rosters: match info, team/player lookups

**Coordinate system:**
- PFF uses center-origin: x ∈ [-52.5, 52.5], y ∈ [-34, 34]
- Pitch dimensions: 105m × 68m (per-match exact values in metadata)
- For xT lookup: `pitch_x = pff_x + pitch_length/2`, `pitch_y = pff_y + pitch_width/2`

**Data loading strategy:**
- `kloppy` library for tracking data (standardized coordinate system, multi-provider compatible)
- Custom parser for PFF event data (kloppy PFF event support still in progress)
- Each event contains embedded `homePlayers`/`awayPlayers` arrays with x/y — acts as a freeze-frame

**Key PFF-specific fields for Threader:**
- `possessionEventType == "PA"` → pass events
- `passerPlayerId` / `targetPlayerId` / `receiverPlayerId` → pass participants
- `passOutcomeType` (`"C"` complete, `"D"` defended) → ground truth
- `betterOptionType` / `betterOptionPlayerId` → PFF's own "better option" annotation (validation!)
- `linesBrokenType` → defensive lines broken
- `pressureType` → whether passer was under pressure

**Rationale:**
- ✅ Real match data — no need for simulated scenarios
- ✅ Event data embeds player positions — each pass has a complete "board state"
- ✅ PFF's `betterOption` annotation aligns directly with Threader's goal
- ✅ kloppy enables future expansion to other data providers (StatsBomb, Metrica, etc.)

---

## 📐 Geometry Algorithm Details

### Passing Lane Obstruction Detection

**Core question:** Is a defender in the passing lane?

**Steps:**
1. Calculate the shortest distance from the defender to the passing line
2. Determine whether the defender is within the "passing line segment" (not the extension)
3. Calculate the degree of obstruction based on distance

**Projection detection:**
```python
def is_in_passing_lane(defender, passer, receiver):
    """Check whether a defender is within the effective range of the passing lane"""
    dx = receiver.x - passer.x
    dy = receiver.y - passer.y
    
    # Projection parameter t
    t = ((defender.x - passer.x) * dx + (defender.y - passer.y) * dy) / (dx² + dy²)
    
    # t ∈ [0, 1] means within the segment
    # t ∈ [0.1, 0.9] to avoid noise near the endpoints
    return 0.1 < t < 0.9
```

---

## 📊 Example Analysis

### Scenario Setup

```yaml
Time: 23rd minute
Ball carrier: #17 Midfielder (60m, 50m)
Situation: Organizing an attack near the center circle
```

### Analysis Results

**1st choice: #9 Striker (85.0, 40.0) — Pass Score 28**
```
Completion:    42%      ← Long distance (25m) + 1 defender blocking
Zone value:    0.180    ← Attacking third, high-threat position
Pressure:      0.5/10   ← Few nearby defenders
Space:         11.2m    ← Ample space
Penetration:   1.00     ← 25m forward, bypasses 2 defenders
```

**Insight:** Despite only 42% completion probability, the high zone value and maximum penetration score yield the highest expected value.

**2nd choice: #8 Attacking Midfielder (75.0, 35.0) — Pass Score 26**
```
Completion:    63%      ← Moderate distance (18m)
Zone value:    0.120
Pressure:      1.2/10
Space:         5.0m
Penetration:   1.00
```

**Insight:** Higher completion probability but lower zone value, resulting in a slightly lower overall score than the 1st choice.

**3rd choice: #47 Right Winger (70.0, 60.0) — Pass Score 19**
```
Completion:    51%
Zone value:    0.080    ← Lower threat on the wing
Pressure:      1.2/10
Space:         7.1m
Penetration:   0.80     ← Partially lateral movement
```

**Insight:** A lateral pass with lower penetration and zone value.

### Key Takeaway

**Expected value thinking:**
```
1st choice: 0.42 × 0.180 = 0.076  (high reward × medium probability)
2nd choice: 0.63 × 0.120 = 0.076  (medium reward × high probability)
```

Even with similar expected values, the penetration bonus tips the balance in favor of the 1st choice.

---

## 🔮 Future Evolution

### Phase 2: Data & Interaction

**Priorities:**
1. Integrate real Metrica data
2. Timeline feature (scrub to view any moment)
3. Actual choice vs AI recommendation comparison
4. Heatmap mode (pitch value density)

### Phase 3: Model Enhancement

**Potential explorations:**
1. Individual ability modeling (Messi's completion rate > average)
2. Lightweight sequence prediction (statistical approximation of "next step")
3. Match context mode (user-selectable risk preference)

### Long-Term Vision

**Become the AlphaGo teaching tool for football:**
- Coaches reviewing matches: "Why did the AI say we should have passed to someone else here?"
- Player development: "How good is my decision-making?"
- Fan experience: "What score does the AI give this pass?"

---

## 💭 Design Philosophy

### Can AI Understand Football?

**We don't pursue "understanding":**
- AlphaGo doesn't need to understand the aesthetics of Go, yet it plays stunning moves
- Threader doesn't need to understand the art of football, yet it can evaluate decision quality

**We pursue "statistical optimality":**
- Learn from large datasets: which passing choices are statistically superior
- Provide an objective "data perspective"

**But remain cautious:**
- Football is full of "irrational" brilliant moments
- Messi's "unreasonable" passes frequently create miracles
- AI scores are a reference, not the truth

### Human-AI Collaboration

**Ideal scenario:**
```
AI:    Provides data perspective + statistical optimality
Human: Retains artistic judgment + tactical intent
→ Together they produce deeper insights
```

---

## 📌 Outstanding Technical Debt

**Scoring weight optimization:**
- [x] Backward pass penalty added (penetration drag, v1.1)
- [x] Pressure changed from additive to multiplicative (v1.1)
- [x] Space bonus capped at 15m (v1.1)
- [x] Zone value amplified ×1.5 (v1.1)
- [ ] Current weights (0.20, etc.) are empirical — need tuning with real data
- [ ] Different zones (defensive vs attacking) may require different weights

**xT model improvement:**
- [ ] Currently using a simplified 12×8 grid
- [ ] Can be upgraded to a more granular xT model

**Passing lane algorithm:**
- [ ] Does not account for defender movement speed
- [ ] Could add "ball flight time vs defender interception time"

**Performance optimization:**
- [ ] Current single analysis ~0.1s — acceptable
- [ ] Analyzing a full match (1000+ passes) would require optimization

---

*Document version: v1.1*  
*Last updated: 2026-02-17*  
*Project: Threader — AlphaGo-inspired Pass Analysis*
