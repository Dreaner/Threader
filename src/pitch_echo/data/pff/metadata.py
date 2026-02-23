"""
Project: PitchEcho
Author: Xingnan Zhu
File Name: metadata.py
Description:
    Match metadata and roster loading.
    Parses PFF metadata JSON, roster JSON, and players.csv to provide
    match info, team details, and player lookups.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TeamInfo:
    """Basic team information from metadata."""

    team_id: str
    name: str
    short_name: str


@dataclass
class MatchInfo:
    """Match-level metadata."""

    game_id: str
    date: str
    home_team: TeamInfo
    away_team: TeamInfo
    pitch_length: float
    pitch_width: float
    stadium_name: str
    home_team_start_left: bool
    season: str = "2022"
    competition: str = "FIFA Men's World Cup"
    fps: float = 29.97
    period_start_times: dict[int, float] = field(default_factory=dict)


@dataclass
class RosterPlayer:
    """A player entry from the roster file."""

    player_id: str
    nickname: str
    team_id: str
    team_name: str
    shirt_number: str
    position: str
    started: bool


@dataclass
class PlayerInfo:
    """Player info from players.csv."""

    player_id: int
    first_name: str
    last_name: str
    nickname: str
    dob: str
    height: float | None
    position_group: str


def load_match_info(metadata_path: str | Path) -> MatchInfo:
    """Load match metadata from a PFF metadata JSON file."""
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        data = data[0]

    home = data["homeTeam"]
    away = data["awayTeam"]
    stadium = data.get("stadium", {})
    pitches = stadium.get("pitches", [{}])
    pitch = pitches[0] if pitches else {}

    # Extract period start times (startPeriod1, startPeriod2, ...)
    period_start_times: dict[int, float] = {}
    for key, val in data.items():
        m = re.match(r"startPeriod(\d+)", key)
        if m and val is not None:
            period_start_times[int(m.group(1))] = float(val)

    return MatchInfo(
        game_id=str(data["id"]),
        date=data.get("date", ""),
        home_team=TeamInfo(
            team_id=home["id"],
            name=home["name"],
            short_name=home.get("shortName", ""),
        ),
        away_team=TeamInfo(
            team_id=away["id"],
            name=away["name"],
            short_name=away.get("shortName", ""),
        ),
        pitch_length=pitch.get("length", 105.0),
        pitch_width=pitch.get("width", 68.0),
        stadium_name=stadium.get("name", ""),
        home_team_start_left=data.get("homeTeamStartLeft", True),
        season=data.get("season", "2022"),
        fps=float(data.get("fps", 29.97)),
        period_start_times=period_start_times,
    )


def load_roster(roster_path: str | Path) -> list[RosterPlayer]:
    """Load team rosters from a PFF roster JSON file."""
    path = Path(roster_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    players = []
    for entry in data:
        player = entry.get("player", entry)
        team = entry.get("team", {})
        players.append(
            RosterPlayer(
                player_id=str(player.get("id", "")),
                nickname=player.get("nickname", ""),
                team_id=str(team.get("id", "")),
                team_name=team.get("name", ""),
                shirt_number=str(entry.get("shirtNumber", "")),
                position=entry.get("positionGroupType", ""),
                started=entry.get("started", False),
            )
        )
    return players


def load_players_csv(csv_path: str | Path) -> dict[int, PlayerInfo]:
    """Load the global players.csv into a lookup dict keyed by player ID."""
    path = Path(csv_path)
    players: dict[int, PlayerInfo] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["id"])
            height_str = row.get("height", "")
            height = float(height_str) if height_str else None
            players[pid] = PlayerInfo(
                player_id=pid,
                first_name=row.get("firstName", ""),
                last_name=row.get("lastName", ""),
                nickname=row.get("nickname", ""),
                dob=row.get("dob", ""),
                height=height,
                position_group=row.get("positionGroupType", ""),
            )

    return players
