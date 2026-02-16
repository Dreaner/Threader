"""Core data models for Threader pass analysis.

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
    """

    home_players: list[Player]
    away_players: list[Player]
    ball: BallPosition
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    @property
    def all_players(self) -> list[Player]:
        return self.home_players + self.away_players

    def teammates(self, player: Player) -> list[Player]:
        """Get all teammates of a player (excluding the player itself)."""
        team = (
            self.home_players
            if player.team_id == self.home_players[0].team_id
            else self.away_players
        )
        return [p for p in team if p.player_id != player.player_id]

    def opponents(self, player: Player) -> list[Player]:
        """Get all opponents of a player."""
        if self.home_players and player.team_id == self.home_players[0].team_id:
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

    # The full snapshot at the moment of this pass
    snapshot: Snapshot

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

    @property
    def ranked_options(self) -> list[PassOption]:
        """Options sorted by Pass Score, highest first."""
        return sorted(self.options, key=lambda o: o.rank_key, reverse=True)

    @property
    def best_option(self) -> PassOption | None:
        ranked = self.ranked_options
        return ranked[0] if ranked else None
