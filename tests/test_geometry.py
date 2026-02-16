"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: test_geometry.py
Description: 
    Tests for geometry utilities — distance and passing lane detection.
"""

import math

from threader.geometry.distance import (
    euclidean,
    player_distance,
    point_to_segment_distance,
    projection_parameter,
)
from threader.geometry.passing_lane import is_in_passing_lane, passing_lane_blocking
from threader.models import Player


def _player(x: float, y: float, pid: int = 1, tid: int = 1) -> Player:
    return Player(player_id=pid, team_id=tid, x=x, y=y)


# --- Distance tests ---


class TestEuclidean:
    def test_same_point(self):
        assert euclidean(0, 0, 0, 0) == 0.0

    def test_horizontal(self):
        assert euclidean(0, 0, 10, 0) == 10.0

    def test_diagonal(self):
        assert math.isclose(euclidean(0, 0, 3, 4), 5.0)


class TestPlayerDistance:
    def test_same_position(self):
        a = _player(10, 20)
        b = _player(10, 20, pid=2)
        assert player_distance(a, b) == 0.0

    def test_known_distance(self):
        a = _player(0, 0)
        b = _player(3, 4, pid=2)
        assert math.isclose(player_distance(a, b), 5.0)


class TestPointToSegmentDistance:
    def test_point_on_segment(self):
        assert point_to_segment_distance(5, 0, 0, 0, 10, 0) == 0.0

    def test_perpendicular_distance(self):
        # Point above the middle of a horizontal segment
        assert math.isclose(
            point_to_segment_distance(5, 3, 0, 0, 10, 0), 3.0
        )

    def test_closest_is_endpoint(self):
        # Point beyond the segment end
        dist = point_to_segment_distance(15, 0, 0, 0, 10, 0)
        assert math.isclose(dist, 5.0)

    def test_degenerate_segment(self):
        dist = point_to_segment_distance(3, 4, 0, 0, 0, 0)
        assert math.isclose(dist, 5.0)


class TestProjectionParameter:
    def test_start(self):
        assert math.isclose(
            projection_parameter(0, 0, 0, 0, 10, 0), 0.0
        )

    def test_midpoint(self):
        assert math.isclose(
            projection_parameter(5, 3, 0, 0, 10, 0), 0.5
        )

    def test_end(self):
        assert math.isclose(
            projection_parameter(10, 0, 0, 0, 10, 0), 1.0
        )

    def test_beyond_end(self):
        t = projection_parameter(20, 0, 0, 0, 10, 0)
        assert t > 1.0


# --- Passing lane tests ---


class TestIsInPassingLane:
    def test_defender_in_lane(self):
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        defender = _player(10, 1, pid=3, tid=2)
        assert is_in_passing_lane(defender, passer, receiver)

    def test_defender_near_passer(self):
        """Defender too close to passer (t < 0.1) should not count."""
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        defender = _player(1, 1, pid=3, tid=2)
        assert not is_in_passing_lane(defender, passer, receiver)

    def test_defender_near_receiver(self):
        """Defender too close to receiver (t > 0.9) should not count."""
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        defender = _player(19, 1, pid=3, tid=2)
        assert not is_in_passing_lane(defender, passer, receiver)

    def test_defender_behind_passer(self):
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        defender = _player(-5, 0, pid=3, tid=2)
        assert not is_in_passing_lane(defender, passer, receiver)


class TestPassingLaneBlocking:
    def test_no_blocking(self):
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        defenders = [_player(10, 20, pid=3, tid=2)]  # Far from lane
        assert passing_lane_blocking(passer, receiver, defenders) == 0.0

    def test_severe_blocking(self):
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        # Defender right on the line
        defenders = [_player(10, 0.5, pid=3, tid=2)]
        assert passing_lane_blocking(passer, receiver, defenders) == 0.4

    def test_moderate_blocking(self):
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        defenders = [_player(10, 2.0, pid=3, tid=2)]
        assert passing_lane_blocking(passer, receiver, defenders) == 0.2

    def test_multiple_blockers_capped(self):
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        # 3 defenders severely blocking
        defenders = [
            _player(5, 0.5, pid=3, tid=2),
            _player(10, 0.5, pid=4, tid=2),
            _player(15, 0.5, pid=5, tid=2),
        ]
        blocking = passing_lane_blocking(passer, receiver, defenders)
        assert blocking == 0.8  # Capped at 0.8
