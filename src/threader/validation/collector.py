"""
Data collector for validation — iterate all matches, analyze all passes,
produce flat ValidatedPass records with ground truth annotations.

Supports pickling cached results to avoid re-analyzing on every run.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

from threader.analysis.analyzer import analyze_pass_event
from threader.data.events import extract_pass_events
from threader.models import AnalysisResult, PassEvent, PassOption


# ---------------------------------------------------------------------------
# ValidatedPass — the flat record used throughout validation
# ---------------------------------------------------------------------------


@dataclass
class ValidatedPass:
    """A single pass with Threader analysis + PFF ground truth."""

    # Identifiers
    match_id: int
    event_id: int
    period: int

    # Passer / actual target
    passer_id: int
    actual_target_id: int
    actual_outcome: str  # "C" or "D"

    # PFF ground truth annotations
    pff_pressure_type: str | None  # "P" or "N"
    pff_better_option_type: str | None
    pff_better_option_player_id: int | None

    # Threader analysis summary
    n_options: int  # total number of teammates evaluated
    threader_top1_id: int  # player_id of Threader's #1 pick

    # Actual target's Threader rank & dimension scores
    actual_target_rank: int  # 1-based rank in Threader's ranking
    actual_target_score: float
    actual_target_completion: float
    actual_target_zone: float
    actual_target_pressure: float
    actual_target_space: float
    actual_target_penetration: float

    # Where PFF's betterOption player ranks in Threader (0 if not annotated / not found)
    better_option_rank: int = 0
    better_option_score: float = 0.0

    # Full ranked options (raw PassOption list) — kept for sensitivity re-scoring
    ranked_options: list[PassOption] = field(default_factory=list)

    # Snapshot-level data needed for re-scoring in sensitivity analysis
    passer_x: float = 0.0
    passer_y: float = 0.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    attack_direction: float = 1.0


def _build_validated_pass(
    pe: PassEvent,
    result: AnalysisResult,
) -> ValidatedPass | None:
    """Convert a PassEvent + AnalysisResult into a ValidatedPass.

    Returns None if the actual target cannot be found in the ranked options.
    """
    ranked = result.ranked_options
    if not ranked:
        return None

    # Find actual target in ranked options
    actual_rank = 0
    actual_option: PassOption | None = None
    for i, opt in enumerate(ranked, 1):
        if opt.target.player_id == pe.target_id:
            actual_rank = i
            actual_option = opt
            break

    if actual_option is None:
        # Actual target not among teammates (rare edge case)
        return None

    # Find PFF betterOption player in ranked options
    better_rank = 0
    better_score = 0.0
    if pe.better_option_player_id:
        for i, opt in enumerate(ranked, 1):
            if opt.target.player_id == pe.better_option_player_id:
                better_rank = i
                better_score = opt.pass_score
                break

    snapshot = pe.snapshot
    attack_dir = snapshot.attack_direction(result.passer)

    return ValidatedPass(
        match_id=pe.game_id,
        event_id=pe.game_event_id,
        period=pe.period,
        passer_id=pe.passer_id,
        actual_target_id=pe.target_id,
        actual_outcome=pe.outcome,
        pff_pressure_type=pe.pressure_type,
        pff_better_option_type=pe.better_option_type,
        pff_better_option_player_id=pe.better_option_player_id,
        n_options=len(ranked),
        threader_top1_id=ranked[0].target.player_id,
        actual_target_rank=actual_rank,
        actual_target_score=actual_option.pass_score,
        actual_target_completion=actual_option.completion_probability,
        actual_target_zone=actual_option.zone_value,
        actual_target_pressure=actual_option.receiving_pressure,
        actual_target_space=actual_option.space_available,
        actual_target_penetration=actual_option.penetration_score,
        better_option_rank=better_rank,
        better_option_score=better_score,
        ranked_options=ranked,
        passer_x=result.passer.x,
        passer_y=result.passer.y,
        pitch_length=snapshot.pitch_length,
        pitch_width=snapshot.pitch_width,
        attack_direction=attack_dir,
    )


# ---------------------------------------------------------------------------
# Collection driver
# ---------------------------------------------------------------------------

def collect_validated_passes(
    data_dir: str | Path,
    max_matches: int | None = None,
    verbose: bool = True,
) -> list[ValidatedPass]:
    """Iterate all matches and collect ValidatedPass records.

    Args:
        data_dir: Root data directory (e.g., "data/FIFA_World_Cup_2022").
        max_matches: If set, only process this many matches (for quick tests).
        verbose: Print progress to stderr.

    Returns:
        List of ValidatedPass records across all processed matches.
    """
    data_dir = Path(data_dir)
    event_dir = data_dir / "Event Data"
    meta_dir = data_dir / "Metadata"

    event_files = sorted(event_dir.glob("*.json"))
    if max_matches:
        event_files = event_files[:max_matches]

    all_records: list[ValidatedPass] = []
    errors = 0

    for idx, event_file in enumerate(event_files, 1):
        match_id = event_file.stem
        meta_file = meta_dir / f"{match_id}.json"
        meta_path = meta_file if meta_file.exists() else None

        if verbose:
            print(
                f"  [{idx}/{len(event_files)}] Match {match_id} ...",
                end="",
                file=sys.stderr,
                flush=True,
            )

        try:
            passes = extract_pass_events(str(event_file), str(meta_path) if meta_path else None)
        except Exception as exc:
            if verbose:
                print(f" LOAD ERROR: {exc}", file=sys.stderr)
            errors += 1
            continue

        match_records = 0
        for pe in passes:
            try:
                result = analyze_pass_event(pe)
                vp = _build_validated_pass(pe, result)
                if vp is not None:
                    all_records.append(vp)
                    match_records += 1
            except Exception:
                errors += 1

        if verbose:
            print(f" {match_records} passes", file=sys.stderr)

    if verbose:
        print(
            f"\n  Total: {len(all_records)} validated passes "
            f"({errors} errors skipped)\n",
            file=sys.stderr,
        )

    return all_records


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def save_cache(records: list[ValidatedPass], path: str | Path) -> None:
    """Pickle validated pass records to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path: str | Path) -> list[ValidatedPass] | None:
    """Load cached records.  Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def data_summary(records: list[ValidatedPass]) -> dict[str, int | float]:
    """Compute high-level summary statistics from the validated passes."""
    n = len(records)
    if n == 0:
        return {"total": 0}

    completed = sum(1 for r in records if r.actual_outcome == "C")
    with_better = sum(1 for r in records if r.pff_better_option_type)
    with_pressure = sum(1 for r in records if r.pff_pressure_type)
    pressed = sum(1 for r in records if r.pff_pressure_type == "P")
    matches = len({r.match_id for r in records})

    return {
        "matches": matches,
        "total": n,
        "completed": completed,
        "completed_pct": round(completed / n * 100, 1),
        "defended": n - completed,
        "with_better_option": with_better,
        "with_better_option_pct": round(with_better / n * 100, 1),
        "with_pressure_annotation": with_pressure,
        "pressed": pressed,
        "not_pressed": with_pressure - pressed,
    }
