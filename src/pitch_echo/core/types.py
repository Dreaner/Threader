"""
Project: PitchEcho
Author: Xingnan Zhu
File Name: core/types.py
Description:
    Protocols and type aliases for the PitchEcho data abstraction layer.
    Data providers (PFF, StatsBomb, etc.) implement these protocols
    to supply pass data to metrics modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pitch_echo.core.models import Snapshot


AttackDirection = float  # +1.0 or -1.0


@runtime_checkable
class PassRecord(Protocol):
    """Minimal protocol for a pass event from any data provider.

    Provider-specific implementations (PFF, StatsBomb, etc.) must
    satisfy this protocol to work with metrics that need snapshot data.
    """

    @property
    def passer_id(self) -> int: ...

    @property
    def snapshot(self) -> Snapshot | None: ...

    @property
    def period(self) -> int: ...
