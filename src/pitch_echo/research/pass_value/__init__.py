"""
Pass Value — AlphaGo-inspired pass target evaluation.

Scores every potential pass target across five dimensions and produces
a ranked list of pass options with a composite 0–100 Pass Score.

⚠️  Research module — API subject to change.

Quick start (internal use only)::

    from pitch_echo.research.pass_value import analyze_pass_event, analyze_snapshot
"""

from pitch_echo.research.pass_value.analysis.analyzer import (
    analyze_pass_event,
    analyze_snapshot,
)
from pitch_echo.research.pass_value.analysis.models import (
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
