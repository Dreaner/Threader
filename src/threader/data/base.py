"""
Data provider protocol for Threader.

Defines the abstract interface that all data providers (PFF, StatsBomb,
Metrica, etc.) must satisfy to supply pass events to Threader metrics.
"""

from __future__ import annotations

from typing import Protocol

from threader.core.models import Snapshot


class PassDataProvider(Protocol):
    """Abstract interface for pass data providers.

    Each provider (PFF, StatsBomb, Metrica, etc.) implements this
    protocol to supply pass events with optional snapshots.
    """

    def load_passes(self, path: str) -> list:
        """Load pass events from a data file."""
        ...

    def get_snapshot(self, pass_event) -> Snapshot | None:
        """Extract the positional snapshot from a pass event, if available."""
        ...
