"""
Project: PitchEcho
File Created: 2026-02-18
Author: Xingnan Zhu
File Name: tracking_frames.py
Description:
    Fast extraction of animation frames from PFF tracking data.

    Primary backend: **Parquet** files with row-group-level predicate
    pushdown via PyArrow — queries any 5-second window in ~20-50ms
    regardless of match period.

    Fallback backend: streaming .jsonl.bz2 decompression for users
    who haven't run the conversion script yet (slower: 0.1-6s).

    Run `uv run python scripts/convert_to_parquet.py` once to generate
    Parquet files and unlock the fast path.

    LRU cache ensures repeated clicks on the same pass are instant.
"""

from __future__ import annotations

import bz2
import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/FIFA_World_Cup_2022")

# Default animation window: 2s before pass → 3s after pass
DEFAULT_PRE_SECONDS = 2.0
DEFAULT_POST_SECONDS = 3.0

# Target animation fps (downsample from ~30fps source)
TARGET_FPS = 10

# Assumed source fps if unknown
DEFAULT_SOURCE_FPS = 30.0


@dataclass
class AnimationPlayerData:
    """A single player's position in one animation frame."""

    player_id: str
    jersey_num: int | None
    x: float
    y: float
    is_home: bool


@dataclass
class AnimationFrame:
    """A single frame of the pass animation."""

    timestamp: float  # seconds elapsed within period
    relative_time: float  # seconds relative to the pass event (negative = before)
    home_players: list[AnimationPlayerData] = field(default_factory=list)
    away_players: list[AnimationPlayerData] = field(default_factory=list)
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_z: float = 0.0


# ===================================================================
# Parquet backend (fast path)
# ===================================================================

def _read_parquet_window(
    parquet_path: Path,
    period: int,
    t_start: float,
    t_end: float,
    pass_elapsed: float,
) -> list[AnimationFrame] | None:
    """Read a time window from a Parquet tracking file using predicate pushdown.

    PyArrow pushes the period/time filters down to the row-group level,
    only reading the ~1-2 row groups that overlap the 5-second window.
    Typical latency: 20-50ms regardless of match period.
    """
    try:
        table = pq.read_table(
            parquet_path,
            filters=[
                ("period", "=", period),
                ("period_elapsed_time", ">=", t_start),
                ("period_elapsed_time", "<=", t_end),
            ],
            columns=[
                "period_elapsed_time",
                "team",
                "jersey_num",
                "x",
                "y",
                "z",
            ],
        )
    except Exception:
        logger.exception("Failed to read Parquet file: %s", parquet_path)
        return None

    if table.num_rows == 0:
        return None

    # Convert to Python lists for grouping
    elapsed_col = table.column("period_elapsed_time").to_pylist()
    team_col = table.column("team").to_pylist()
    jersey_col = table.column("jersey_num").to_pylist()
    x_col = table.column("x").to_pylist()
    y_col = table.column("y").to_pylist()
    z_col = table.column("z").to_pylist()

    # Group rows by timestamp → build AnimationFrame per unique timestamp.
    # Data is sorted by (period, period_elapsed_time, team, jersey_num)
    # so rows with the same timestamp are contiguous.
    frames: list[AnimationFrame] = []
    current_ts: float | None = None
    current_frame: AnimationFrame | None = None

    for i in range(table.num_rows):
        ts = float(elapsed_col[i])

        if ts != current_ts:
            if current_frame is not None:
                frames.append(current_frame)
            current_ts = ts
            current_frame = AnimationFrame(
                timestamp=ts,
                relative_time=ts - pass_elapsed,
            )

        team = team_col[i]
        jersey = int(jersey_col[i]) if jersey_col[i] is not None else None
        x = float(x_col[i])
        y = float(y_col[i])

        assert current_frame is not None

        if team == "ball":
            current_frame.ball_x = x
            current_frame.ball_y = y
            current_frame.ball_z = float(z_col[i]) if z_col[i] else 0.0
        else:
            player = AnimationPlayerData(
                player_id=str(jersey or ""),
                jersey_num=jersey,
                x=x,
                y=y,
                is_home=(team == "home"),
            )
            if team == "home":
                current_frame.home_players.append(player)
            else:
                current_frame.away_players.append(player)

    # Don't forget the last frame
    if current_frame is not None:
        frames.append(current_frame)

    return frames if frames else None


# ===================================================================
# bz2 fallback backend (slow path)
# ===================================================================

def _parse_players(
    raw_list: list[dict[str, Any]],
    is_home: bool,
) -> list[AnimationPlayerData]:
    """Parse a homePlayersSmoothed / awayPlayersSmoothed array."""
    players = []
    for p in raw_list:
        x = p.get("x")
        y = p.get("y")
        if x is None or y is None:
            continue
        jersey_raw = p.get("jerseyNum")
        jersey_num = int(jersey_raw) if jersey_raw is not None else None
        players.append(
            AnimationPlayerData(
                player_id=str(jersey_num or ""),
                jersey_num=jersey_num,
                x=float(x),
                y=float(y),
                is_home=is_home,
            )
        )
    return players


def _parse_frame(
    record: dict[str, Any],
    pass_elapsed: float,
) -> AnimationFrame:
    """Convert a raw JSONL record to an AnimationFrame."""
    elapsed = float(record.get("periodElapsedTime", 0.0))
    relative_time = elapsed - pass_elapsed

    # Use smoothed data (higher quality) with fallback to raw
    home_raw = record.get("homePlayersSmoothed") or record.get("homePlayers") or []
    away_raw = record.get("awayPlayersSmoothed") or record.get("awayPlayers") or []

    home_players = _parse_players(home_raw, is_home=True)
    away_players = _parse_players(away_raw, is_home=False)

    # Ball — smoothed preferred
    ball_data = record.get("ballsSmoothed") or {}
    if not ball_data:
        balls_list = record.get("balls") or []
        ball_data = balls_list[0] if balls_list else {}

    ball_x = float(ball_data.get("x") or 0.0)
    ball_y = float(ball_data.get("y") or 0.0)
    ball_z = float(ball_data.get("z") or 0.0)

    return AnimationFrame(
        timestamp=elapsed,
        relative_time=relative_time,
        home_players=home_players,
        away_players=away_players,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_z=ball_z,
    )


def _stream_tracking_window(
    tracking_path: Path,
    period: int,
    t_start: float,
    t_end: float,
) -> list[dict[str, Any]]:
    """Stream a .jsonl.bz2 file and collect frames in [t_start, t_end].

    Reads line by line, skips non-matching periods, and breaks early
    once past the window end — avoiding parsing the entire file.

    Optimisation: uses a fast string check to skip JSON parsing for
    lines belonging to other periods (~30-40% faster for period 2).

    Returns raw JSON dicts (not yet converted to AnimationFrame).
    """
    raw_frames: list[dict[str, Any]] = []

    # Build fast string markers for the target period.
    # PFF tracking JSON uses compact format: "period":1 (no space).
    period_marker = f'"period":{period}'
    # Fallback for pretty-printed JSON:
    period_marker_spaced = f'"period": {period}'

    with bz2.open(tracking_path, "rt", encoding="utf-8") as f:
        for line in f:
            # Fast string check: skip lines for other periods without
            # paying the cost of full JSON parsing.
            if period_marker not in line and period_marker_spaced not in line:
                # If we already collected frames and period changed → done
                if raw_frames:
                    break
                continue

            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            elapsed = float(record.get("periodElapsedTime", -1.0))

            if elapsed < t_start:
                continue
            if elapsed > t_end:
                break  # ← Early termination! Skip rest of file

            raw_frames.append(record)

    return raw_frames


# ===================================================================
# Main extraction — auto-selects Parquet or bz2 backend
# ===================================================================

def extract_pass_animation_frames(
    game_id: str,
    pass_event,
    period_start_times: dict[int, float],
    *,
    pre_seconds: float = DEFAULT_PRE_SECONDS,
    post_seconds: float = DEFAULT_POST_SECONDS,
    target_fps: int = TARGET_FPS,
) -> list[AnimationFrame] | None:
    """Extract animation frames around a pass event from tracking data.

    Automatically uses Parquet if available (fast: ~30ms), otherwise
    falls back to streaming .jsonl.bz2 (slower: 0.1-6s).

    Args:
        game_id: Match identifier (e.g., "3835").
        pass_event: The pass event with event_time and period attributes.
        period_start_times: Dict mapping period number to video start time.
        pre_seconds: Seconds before the pass to include.
        post_seconds: Seconds after the pass to include.
        target_fps: Target animation frame rate (downsampled from ~30fps).

    Returns:
        List of AnimationFrame objects, or None if tracking data unavailable.
    """
    # Need event_time for precise alignment
    if pass_event.event_time is None:
        logger.warning("Pass event has no event_time; cannot align with tracking data")
        return None

    period = pass_event.period
    period_start = period_start_times.get(period)
    if period_start is None:
        logger.warning("No period start time for period %d", period)
        return None

    # Calculate period-elapsed time of the pass event
    pass_elapsed = pass_event.event_time - period_start

    if pass_elapsed < 0:
        logger.warning(
            "Negative elapsed time: event_time=%.2f, period_start=%.2f",
            pass_event.event_time,
            period_start,
        )
        return None

    # Define the time window
    t_start = max(0.0, pass_elapsed - pre_seconds)
    t_end = pass_elapsed + post_seconds

    # --- Try Parquet first (fast path) ---
    parquet_path = DATA_ROOT / "Tracking Data Parquet" / f"{game_id}.parquet"
    if parquet_path.exists():
        t0 = time.monotonic()
        frames = _read_parquet_window(
            parquet_path, period, t_start, t_end, pass_elapsed
        )
        read_ms = (time.monotonic() - t0) * 1000

        if frames:
            logger.info(
                "[Parquet] Read %d frames in %.0fms for period %d window [%.1f, %.1f]",
                len(frames),
                read_ms,
                period,
                t_start,
                t_end,
            )

            # Downsample to target FPS
            step = max(1, round(DEFAULT_SOURCE_FPS / target_fps))
            frames = frames[::step]
            logger.info(
                "[Parquet] Downsampled → %d frames (target=%dfps)",
                len(frames),
                target_fps,
            )
            return frames

        logger.warning(
            "[Parquet] No frames found for match %s period %d [%.1f, %.1f]",
            game_id,
            period,
            t_start,
            t_end,
        )
        return None

    # --- Fallback to bz2 streaming (slow path) ---
    bz2_path = DATA_ROOT / "Tracking Data" / f"{game_id}.jsonl.bz2"
    if not bz2_path.exists():
        logger.warning("No tracking data found for match %s", game_id)
        return None

    logger.info(
        "[bz2 fallback] Parquet not found for %s, streaming bz2 "
        "(run 'uv run python scripts/convert_to_parquet.py' for faster loading)",
        game_id,
    )

    t0 = time.monotonic()
    raw_frames = _stream_tracking_window(bz2_path, period, t_start, t_end)
    stream_ms = (time.monotonic() - t0) * 1000

    if not raw_frames:
        logger.warning(
            "[bz2] No tracking frames for match %s, "
            "period %d, window [%.1f, %.1f]",
            game_id,
            period,
            t_start,
            t_end,
        )
        return None

    logger.info(
        "[bz2] Streamed %d frames in %.0fms for period %d window [%.1f, %.1f]",
        len(raw_frames),
        stream_ms,
        period,
        t_start,
        t_end,
    )

    # Downsample to target FPS
    step = max(1, round(DEFAULT_SOURCE_FPS / target_fps))
    downsampled = raw_frames[::step]

    logger.info(
        "[bz2] Downsampled %d → %d frames (step=%d, target=%dfps)",
        len(raw_frames),
        len(downsampled),
        step,
        target_fps,
    )

    # Convert to AnimationFrame objects
    animation_frames = [_parse_frame(rec, pass_elapsed) for rec in downsampled]

    return animation_frames


# ---------------------------------------------------------------------------
# LRU-cached wrapper — instant replay on repeated clicks
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _cached_extract(
    game_id: str,
    period: int,
    event_time: float,
    period_start: float,
    pre_seconds: float,
    post_seconds: float,
    target_fps: int,
) -> list[AnimationFrame] | None:
    """Cache-friendly wrapper with hashable-only arguments."""
    from pitch_echo.data.pff.events import PassEvent as _PE

    # Build a minimal PassEvent just for extraction
    dummy = _PE.__new__(_PE)
    object.__setattr__(dummy, "event_time", event_time)
    object.__setattr__(dummy, "period", period)

    return extract_pass_animation_frames(
        game_id=game_id,
        pass_event=dummy,
        period_start_times={period: period_start},
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        target_fps=target_fps,
    )


def get_animation_frames_cached(
    game_id: str,
    pass_event,
    period_start_times: dict[int, float],
    *,
    pre_seconds: float = DEFAULT_PRE_SECONDS,
    post_seconds: float = DEFAULT_POST_SECONDS,
    target_fps: int = TARGET_FPS,
) -> list[AnimationFrame] | None:
    """Extract animation frames with LRU caching.

    Repeated clicks on the same pass event return instantly from cache.
    The cache key is (game_id, period, event_time, period_start) —
    all hashable.
    """
    if pass_event.event_time is None:
        return None

    period = pass_event.period
    period_start = period_start_times.get(period)
    if period_start is None:
        return None

    return _cached_extract(
        game_id=game_id,
        period=period,
        event_time=pass_event.event_time,
        period_start=period_start,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        target_fps=target_fps,
    )
