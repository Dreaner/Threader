#!/usr/bin/env python3
"""Convert PFF tracking data from .jsonl.bz2 to Parquet format.

One-time conversion script. Run once to generate Parquet files:

    uv run python scripts/convert_to_parquet.py

Each .jsonl.bz2 file (~46-64MB compressed, ~185k frames of nested JSON)
becomes a flat Parquet file (~23 rows per frame) with row-group-level
statistics enabling predicate pushdown for fast time-range queries.

Output: data/FIFA_World_Cup_2022/Tracking Data Parquet/{game_id}.parquet
"""

from __future__ import annotations

import bz2
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/FIFA_World_Cup_2022")
SRC_DIR = DATA_ROOT / "Tracking Data"
DST_DIR = DATA_ROOT / "Tracking Data Parquet"

# ---------------------------------------------------------------------------
# Parquet schema (flat: one row per entity per frame)
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    pa.field("period", pa.uint8()),
    pa.field("period_elapsed_time", pa.float32()),
    pa.field("team", pa.utf8()),            # "home" / "away" / "ball"
    pa.field("jersey_num", pa.uint8()),      # 0 for ball
    pa.field("x", pa.float32()),
    pa.field("y", pa.float32()),
    pa.field("z", pa.float32()),            # only meaningful for ball
])

# Row group target: ~30 seconds of tracking data ≈ 30s × 30fps × 23 rows ≈ 20,700
ROW_GROUP_SIZE = 20_000


def _flatten_frame(record: dict) -> list[tuple]:
    """Convert one JSONL record into flat rows (one per player + ball)."""
    period = record.get("period", 0)
    elapsed = record.get("periodElapsedTime", 0.0)

    rows: list[tuple] = []

    # Home players (prefer smoothed)
    home = record.get("homePlayersSmoothed") or record.get("homePlayers") or []
    for p in home:
        x = p.get("x")
        y = p.get("y")
        if x is None or y is None:
            continue
        jn = p.get("jerseyNum")
        jersey = int(jn) if jn is not None else 0
        rows.append((period, elapsed, "home", jersey, x, y, 0.0))

    # Away players (prefer smoothed)
    away = record.get("awayPlayersSmoothed") or record.get("awayPlayers") or []
    for p in away:
        x = p.get("x")
        y = p.get("y")
        if x is None or y is None:
            continue
        jn = p.get("jerseyNum")
        jersey = int(jn) if jn is not None else 0
        rows.append((period, elapsed, "away", jersey, x, y, 0.0))

    # Ball (prefer smoothed)
    ball = record.get("ballsSmoothed")
    if not ball:
        balls_list = record.get("balls") or []
        ball = balls_list[0] if balls_list else {}
    if ball:
        bx = float(ball.get("x") or 0.0)
        by = float(ball.get("y") or 0.0)
        bz = float(ball.get("z") or 0.0)
        rows.append((period, elapsed, "ball", 0, bx, by, bz))

    return rows


def convert_file(src: Path, dst: Path) -> int:
    """Convert one .jsonl.bz2 to .parquet. Returns row count."""
    # Collect all rows into column-oriented lists for efficiency
    col_period: list[int] = []
    col_elapsed: list[float] = []
    col_team: list[str] = []
    col_jersey: list[int] = []
    col_x: list[float] = []
    col_y: list[float] = []
    col_z: list[float] = []

    with bz2.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for period, elapsed, team, jersey, x, y, z in _flatten_frame(record):
                col_period.append(period)
                col_elapsed.append(elapsed)
                col_team.append(team)
                col_jersey.append(jersey)
                col_x.append(x)
                col_y.append(y)
                col_z.append(z)

    n_rows = len(col_period)
    if n_rows == 0:
        print(f"  WARNING: No rows extracted from {src.name}")
        return 0

    table = pa.table(
        {
            "period": pa.array(col_period, type=pa.uint8()),
            "period_elapsed_time": pa.array(col_elapsed, type=pa.float32()),
            "team": pa.array(col_team, type=pa.utf8()),
            "jersey_num": pa.array(col_jersey, type=pa.uint8()),
            "x": pa.array(col_x, type=pa.float32()),
            "y": pa.array(col_y, type=pa.float32()),
            "z": pa.array(col_z, type=pa.float32()),
        },
        schema=SCHEMA,
    )

    # Sort by (period, period_elapsed_time, team, jersey_num) for
    # optimal row-group statistics and query locality
    table = table.sort_by([
        ("period", "ascending"),
        ("period_elapsed_time", "ascending"),
        ("team", "ascending"),
        ("jersey_num", "ascending"),
    ])

    pq.write_table(
        table,
        dst,
        compression="zstd",
        row_group_size=ROW_GROUP_SIZE,
        write_statistics=True,  # enables predicate pushdown
    )

    return n_rows


def main() -> None:
    """Convert all tracking data files to Parquet format."""
    if not SRC_DIR.exists():
        print(f"ERROR: Source dir not found: {SRC_DIR}")
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    bz2_files = sorted(SRC_DIR.glob("*.jsonl.bz2"))
    if not bz2_files:
        print(f"ERROR: No .jsonl.bz2 files found in {SRC_DIR}")
        sys.exit(1)

    total = len(bz2_files)
    total_rows = 0
    total_t0 = time.monotonic()

    print(f"Converting {total} tracking data files to Parquet...")
    print(f"  Source: {SRC_DIR}")
    print(f"  Output: {DST_DIR}")
    print()

    for i, src in enumerate(bz2_files, 1):
        game_id = src.stem.replace(".jsonl", "")
        dst = DST_DIR / f"{game_id}.parquet"

        t0 = time.monotonic()
        n_rows = convert_file(src, dst)
        elapsed = time.monotonic() - t0

        size_mb = dst.stat().st_size / (1024 * 1024) if dst.exists() else 0
        total_rows += n_rows

        print(
            f"  [{i:2d}/{total}] {game_id}: "
            f"{n_rows:>8,} rows → {size_mb:.1f} MB "
            f"({elapsed:.1f}s)"
        )

    total_elapsed = time.monotonic() - total_t0
    print()
    print(f"Done! {total} files, {total_rows:,} total rows in {total_elapsed:.0f}s")
    print(f"Output: {DST_DIR}")


if __name__ == "__main__":
    main()
