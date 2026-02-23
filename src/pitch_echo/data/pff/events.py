"""
Project: PitchEcho
Author: Xingnan Zhu
File Name: events.py
Description:
    PFF FC Event Data parser.
    Extracts pass events and player position snapshots from PFF's
    event data JSON files (FIFA World Cup 2022).
    Each event in the JSON array contains:
      - possessionEvents: dict with pass/shot/carry details
      - homePlayers / awayPlayers: list of player positions (x, y)
      - ball: list with ball position (x, y, z)
      - gameEvents: dict with period, team, player info
      - stadiumMetadata: pitch dimensions, team directions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pitch_echo.core.models import BallPosition, Player, Snapshot


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


def _parse_player(raw: dict[str, Any], team_id: int) -> Player:
    """Parse a single player from homePlayers/awayPlayers array."""
    return Player(
        player_id=raw["playerId"],
        team_id=team_id,
        x=raw["x"],
        y=raw["y"],
        jersey_num=raw.get("jerseyNum"),
        position=raw.get("positionGroupType"),
    )


def _parse_snapshot(
    event: dict[str, Any],
    home_team_id: int,
    away_team_id: int,
    *,
    home_team_start_left: bool = True,
) -> Snapshot:
    """Build a Snapshot from a PFF event's embedded player/ball data."""
    home_players = [
        _parse_player(p, home_team_id) for p in event.get("homePlayers", [])
    ]
    away_players = [
        _parse_player(p, away_team_id) for p in event.get("awayPlayers", [])
    ]

    ball_data = event.get("ball", [{}])
    ball_raw = ball_data[0] if ball_data else {}
    ball = BallPosition(
        x=ball_raw.get("x", 0.0),
        y=ball_raw.get("y", 0.0),
        z=ball_raw.get("z", 0.0),
    )

    stadium = event.get("stadiumMetadata", {})
    pitch_length = stadium.get("pitchLength", 105.0)
    pitch_width = stadium.get("pitchWidth", 68.0)

    ge = event.get("gameEvents", {})
    period = ge.get("period", 1)

    # homeTeamStartLeft means home attacks towards positive-x in Period 1
    home_attacks_right = home_team_start_left

    return Snapshot(
        home_players=home_players,
        away_players=away_players,
        ball=ball,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        home_attacks_right=home_attacks_right,
        period=period,
    )


def _extract_team_ids(event: dict[str, Any]) -> tuple[int, int]:
    """Extract home and away team IDs from a PFF event.

    Uses the gameEvents.teamId + homeTeam flag to determine which is which.
    """
    ge = event.get("gameEvents", {})
    team_id = ge.get("teamId")
    is_home = ge.get("homeTeam", False)

    # We need actual team IDs — get from gameEvents
    if team_id is not None:
        if is_home:
            home_team_id = team_id
            away_team_id = 0  # placeholder, will be overridden
        else:
            away_team_id = team_id
            home_team_id = 0
    else:
        home_team_id = 0
        away_team_id = 0

    return home_team_id, away_team_id


def load_events(event_path: str | Path) -> list[dict[str, Any]]:
    """Load raw events from a PFF event data JSON file."""
    path = Path(event_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_pass_events(
    event_path: str | Path,
    metadata_path: str | Path | None = None,
) -> list[PassEvent]:
    """Extract all pass events from a PFF event data file.

    Each pass event includes the complete player snapshot at that moment.

    Args:
        event_path: Path to the event data JSON (e.g., "3812.json")
        metadata_path: Optional path to metadata JSON for team IDs.
            If not provided, team IDs are inferred from event data.

    Returns:
        List of PassEvent objects with embedded Snapshots.
    """
    raw_events = load_events(event_path)

    # Resolve team IDs from metadata if available
    home_team_id = 0
    away_team_id = 0
    home_team_start_left = True  # default if no metadata
    if metadata_path:
        with Path(metadata_path).open("r", encoding="utf-8") as f:
            meta = json.load(f)
            if isinstance(meta, list):
                meta = meta[0]
            home_team_id = int(meta["homeTeam"]["id"])
            away_team_id = int(meta["awayTeam"]["id"])
            home_team_start_left = meta.get("homeTeamStartLeft", True)

    pass_events: list[PassEvent] = []

    for event in raw_events:
        pe = event.get("possessionEvents")
        if not isinstance(pe, dict):
            continue
        if pe.get("possessionEventType") != "PA":
            continue

        # Skip if no passer info
        passer_id = pe.get("passerPlayerId")
        if passer_id is None:
            continue

        ge = event.get("gameEvents", {})

        # Resolve team IDs if not from metadata
        h_id = home_team_id
        a_id = away_team_id
        if h_id == 0 or a_id == 0:
            team_id = ge.get("teamId", 0)
            is_home = ge.get("homeTeam", False)
            if is_home:
                h_id = team_id
            else:
                a_id = team_id

        snapshot = _parse_snapshot(
            event, h_id, a_id,
            home_team_start_left=home_team_start_left,
        )

        pass_event = PassEvent(
            game_id=event.get("gameId", 0),
            game_event_id=event.get("gameEventId", 0),
            game_clock=pe.get("gameClock", 0),
            period=ge.get("period", 0),
            passer_id=passer_id,
            passer_name=pe.get("passerPlayerName", ""),
            target_id=pe.get("targetPlayerId", 0),
            target_name=pe.get("targetPlayerName", ""),
            receiver_id=pe.get("receiverPlayerId"),
            receiver_name=pe.get("receiverPlayerName"),
            pass_type=pe.get("passType", ""),
            outcome=pe.get("passOutcomeType", ""),
            pressure_type=pe.get("pressureType"),
            lines_broken=pe.get("linesBrokenType"),
            body_part=pe.get("bodyType"),
            better_option_type=pe.get("betterOptionType"),
            better_option_player_id=pe.get("betterOptionPlayerId"),
            better_option_player_name=pe.get("betterOptionPlayerName"),
            event_time=event.get("eventTime"),
            snapshot=snapshot,
        )
        pass_events.append(pass_event)

    return pass_events
