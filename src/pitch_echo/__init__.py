"""
PitchEcho — A Python library for football pass analysis research.

Usage::

    from pitch_echo import (
        Player, BallPosition, Snapshot,
        PassOption, AnalysisResult, ScoringWeights,
        analyze_snapshot, analyze_pass_event,
        PassNetwork, PassEdge, PlayerNode,
        NetworkMetrics, PlayerMetrics,
        build_pass_network, compute_metrics,
        load_pff, Pitch,
    )
"""

from importlib.metadata import version
__version__ = version("pitch-echo")

# Core models
from pitch_echo.core.models import BallPosition, Player, Snapshot

# Analysis
from pitch_echo.analysis.analyzer import analyze_pass_event, analyze_snapshot
from pitch_echo.analysis.models import AnalysisResult, PassOption, ScoringWeights

# Network
from pitch_echo.network.builder import build_pass_network
from pitch_echo.network.metrics import compute_metrics
from pitch_echo.network.models import (
    NetworkMetrics,
    PassEdge,
    PassNetwork,
    PlayerMetrics,
    PlayerNode,
)

# Visualization
from pitch_echo.pitch import Pitch

# Data loading convenience
from pitch_echo.data.pff.events import extract_pass_events as load_pff

__all__ = [
    # Core
    "BallPosition",
    "Player",
    "Snapshot",
    # Analysis
    "AnalysisResult",
    "PassOption",
    "ScoringWeights",
    "analyze_pass_event",
    "analyze_snapshot",
    # Network
    "NetworkMetrics",
    "PassEdge",
    "PassNetwork",
    "PlayerMetrics",
    "PlayerNode",
    "build_pass_network",
    "compute_metrics",
    # Data
    "load_pff",
    # Viz
    "Pitch",
]
