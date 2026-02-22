"""Backward-compatibility shim — imports from new locations."""

from threader.data.pff.events import extract_pass_events
from threader.data.pff.metadata import load_match_info, load_players_csv, load_roster
from threader.data.pff.tracking import load_tracking_data

__all__ = [
    "extract_pass_events",
    "load_match_info",
    "load_players_csv",
    "load_roster",
    "load_tracking_data",
]
