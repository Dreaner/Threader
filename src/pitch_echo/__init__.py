"""
Threader — A Python library for football pass analysis research.
"""

__version__ = "0.2.0"

# Core models
from threader.core.models import BallPosition, Player, Snapshot

# Primary API — Pass Value metric
from threader.metrics.pass_value import analyze_pass_event, analyze_snapshot
from threader.metrics.pass_value.models import (
    AnalysisResult,
    PassOption,
    ScoringWeights,
)

__all__ = [
    "AnalysisResult",
    "BallPosition",
    "PassOption",
    "Player",
    "ScoringWeights",
    "Snapshot",
    "analyze_pass_event",
    "analyze_snapshot",
]
