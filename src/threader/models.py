"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: models.py
Description: 
    Core data models for Threader pass analysis.
    All coordinates use PFF's center-origin system:
      x ∈ [-52.5, 52.5]  (negative = left / defensive, positive = right / attacking)
      y ∈ [-34, 34]       (negative = bottom, positive = top)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Player:
    """A player on the pitch at a specific moment."""

    player_id: int
    team_id: int
    x: float
    y: float
    jersey_num: int | None = None
    name: str | None = None
    position: str | None = None  # e.g. "GK", "CB", "CF", "RW"


@dataclass(frozen=True)
class BallPosition:
    """Ball position at a specific moment."""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    """A frozen moment on the pitch — the 'board state'.

    Contains positions of all visible players and the ball.
    Attacking direction is determined from metadata:
      - home_attacks_right=True means home team attacks positive-x in P1.
      - In Period 2+ the direction swaps.
    """

    home_players: list[Player]
    away_players: list[Player]
    ball: BallPosition
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    home_attacks_right: bool = True
    period: int = 1

    @property
    def all_players(self) -> list[Player]:
        return self.home_players + self.away_players

    def _is_home(self, player: Player) -> bool:
        """Check if a player belongs to the home team."""
        return bool(
            self.home_players
            and player.team_id == self.home_players[0].team_id
        )

    def attack_direction(self, player: Player) -> float:
        """Return +1.0 if the player's team attacks towards positive-x,
        -1.0 if towards negative-x.

        Uses home_attacks_right and period to determine direction.
        In Period 2 (and extra-time periods 3/4), directions swap.
        """
        home_right = self.home_attacks_right
        # Swap direction in even periods (2nd half, ET 2nd half)
        if self.period % 2 == 0:
            home_right = not home_right

        if self._is_home(player):
            return 1.0 if home_right else -1.0
        return -1.0 if home_right else 1.0

    def teammates(self, player: Player) -> list[Player]:
        """Get all teammates of a player (excluding the player itself)."""
        team = (
            self.home_players
            if self._is_home(player)
            else self.away_players
        )
        return [p for p in team if p.player_id != player.player_id]

    def opponents(self, player: Player) -> list[Player]:
        """Get all opponents of a player."""
        if self._is_home(player):
            return list(self.away_players)
        return list(self.home_players)


@dataclass(frozen=True)
class PassOption:
    """A scored pass option — one possible target for the ball carrier."""

    target: Player
    pass_score: float  # 0–100
    completion_probability: float  # 0–1
    zone_value: float  # xT value at target position
    receiving_pressure: float  # 0–10
    space_available: float  # meters
    penetration_score: float  # 0–1

    @property
    def rank_key(self) -> float:
        """Key for sorting (higher is better)."""
        return self.pass_score


@dataclass(frozen=True)
class PassEvent:
    """A pass event extracted from PFF event data.

    Contains the actual pass decision, outcome, and the full snapshot
    at the moment of the pass.
    """

    game_id: int
    game_event_id: int
    game_clock: int  # seconds
    period: int

    passer_id: int
    passer_name: str
    target_id: int
    target_name: str
    receiver_id: int | None  # None if incomplete
    receiver_name: str | None

    pass_type: str  # "S" (short), "O" (over-the-top/long)
    outcome: str  # "C" (complete), "D" (defended/incomplete)
    pressure_type: str | None  # "P" (pressed), "N" (not pressed)
    lines_broken: str | None
    body_part: str | None  # "L" (left), "R" (right)

    # PFF's own "better option" annotation — our validation ground truth
    better_option_type: str | None
    better_option_player_id: int | None
    better_option_player_name: str | None

    # High-precision video timestamp (seconds) for tracking data alignment
    event_time: float | None = None

    # The full snapshot at the moment of this pass
    snapshot: Snapshot = field(default=None)  # type: ignore[assignment]

    @property
    def is_complete(self) -> bool:
        return self.outcome == "C"


@dataclass
class AnalysisResult:
    """Complete analysis of a passing moment.

    Contains the passer, the snapshot, and all scored pass options
    ranked from best to worst.
    """

    passer: Player
    snapshot: Snapshot
    options: list[PassOption] = field(default_factory=list)
    team_mean_xT: float | None = None  # Mean xT of outfield teammates (GK excluded)

    @property
    def ranked_options(self) -> list[PassOption]:
        """Options sorted by Pass Score, highest first."""
        return sorted(self.options, key=lambda o: o.rank_key, reverse=True)

    @property
    def best_option(self) -> PassOption | None:
        ranked = self.ranked_options
        return ranked[0] if ranked else None
