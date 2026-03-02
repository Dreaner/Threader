"""
PitchEcho — A Python toolkit for football spatial data analysis.

Public API
----------
- **Core models**: ``Player``, ``Snapshot``, ``BallPosition``
- **Data loading**: PFF FC event parser (``extract_pass_events``)
- **Visualization**: ``Pitch`` class with matplotlib / Plotly backends
- **Pass Network**: ``build_pass_network``, ``compute_metrics``

Research modules (pass scoring, validation, ML learning) live under
``pitch_echo.research`` and are **not** part of the public API — they
may change without notice between versions.
"""

from importlib.metadata import version

__version__ = version("pitch-echo")

# ── Core models ──────────────────────────────────────────────────────
from pitch_echo.core.models import BallPosition, Player, Snapshot

# ── Data loading ─────────────────────────────────────────────────────
from pitch_echo.data.pff.events import PassEvent, extract_pass_events

# ── Pass Network ─────────────────────────────────────────────────────
from pitch_echo.network.builder import build_pass_network
from pitch_echo.network.metrics import compute_metrics
from pitch_echo.network.models import (
    NetworkMetrics,
    PassEdge,
    PassNetwork,
    PlayerMetrics,
    PlayerNode,
)

# ── Visualization ────────────────────────────────────────────────────
from pitch_echo.pitch import Pitch

__all__ = [
    # Core
    "BallPosition",
    "Player",
    "Snapshot",
    # Data
    "PassEvent",
    "extract_pass_events",
    # Network
    "build_pass_network",
    "compute_metrics",
    "NetworkMetrics",
    "PassEdge",
    "PassNetwork",
    "PlayerMetrics",
    "PlayerNode",
    # Visualization
    "Pitch",
]
