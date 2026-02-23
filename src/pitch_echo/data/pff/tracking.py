"""
Project: Threader
Author: Xingnan Zhu
File Name: tracking.py
Description:
    Tracking data loader using kloppy.
    Loads PFF FC tracking data (.jsonl.bz2) via kloppy for standardized
    coordinate systems and multi-provider compatibility.
"""

from __future__ import annotations

from pathlib import Path

from kloppy import pff


def load_tracking_data(
    tracking_path: str | Path,
    metadata_path: str | Path,
    roster_path: str | Path,
    *,
    coordinates: str = "pff",
    sample_rate: float | None = None,
    limit: int | None = None,
    only_alive: bool = True,
):
    """Load PFF tracking data via kloppy.

    Args:
        tracking_path: Path to .jsonl.bz2 tracking file
        metadata_path: Path to metadata JSON
        roster_path: Path to roster JSON
        coordinates: Coordinate system ("pff" for center-origin)
        sample_rate: Downsample rate (e.g., 1/25 for 1fps from 25fps)
        limit: Maximum number of frames to load
        only_alive: Only include frames when ball is in play

    Returns:
        kloppy TrackingDataset
    """
    return pff.load_tracking(
        meta_data=str(metadata_path),
        roster_meta_data=str(roster_path),
        raw_data=str(tracking_path),
        coordinates=coordinates,
        sample_rate=sample_rate,
        limit=limit,
        only_alive=only_alive,
    )
