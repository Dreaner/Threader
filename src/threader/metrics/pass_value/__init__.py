"""
Pass Value metric — the core Pass Score system.

Evaluates pass targets using 5 dimensions: completion probability,
zone value (xT), receiving pressure, space available, and penetration.
"""

from threader.metrics.pass_value.analyzer import analyze_pass_event, analyze_snapshot
from threader.metrics.pass_value.models import (
    AnalysisResult,
    PassOption,
    ScoringWeights,
)

__all__ = [
    "AnalysisResult",
    "PassOption",
    "ScoringWeights",
    "analyze_pass_event",
    "analyze_snapshot",
]
