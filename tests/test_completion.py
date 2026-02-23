"""
Project: PitchEcho
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: test_completion.py
Description: 
    Tests for completion probability scoring.
"""

import math

from pitch_echo.core.models import Player
from pitch_echo.scoring.completion import completion_probability


def _player(x: float, y: float, pid: int = 1, tid: int = 1) -> Player:
    return Player(player_id=pid, team_id=tid, x=x, y=y)


class TestCompletionProbability:
    def test_short_pass_no_defenders(self):
        passer = _player(0, 0)
        receiver = _player(5, 0, pid=2)
        prob = completion_probability(passer, receiver, [])
        assert math.isclose(prob, 0.95)

    def test_medium_pass_no_defenders(self):
        passer = _player(0, 0)
        receiver = _player(15, 0, pid=2)
        prob = completion_probability(passer, receiver, [])
        assert math.isclose(prob, 0.85)

    def test_long_pass_no_defenders(self):
        passer = _player(0, 0)
        receiver = _player(25, 0, pid=2)
        prob = completion_probability(passer, receiver, [])
        assert math.isclose(prob, 0.70)

    def test_very_long_pass(self):
        passer = _player(0, 0)
        receiver = _player(50, 0, pid=2)
        prob = completion_probability(passer, receiver, [])
        # 0.90 - 50*0.015 = 0.15, clamped to 0.40
        assert math.isclose(prob, 0.40)

    def test_defender_blocking_reduces_probability(self):
        passer = _player(0, 0)
        receiver = _player(20, 0, pid=2)
        # Defender right on the line
        defenders = [_player(10, 0.5, pid=3, tid=2)]
        prob = completion_probability(passer, receiver, defenders)
        # distance=20m → base=0.70 (falls into <30m bracket), blocking=0.4
        # result = 0.70 * (1 - 0.4) = 0.42
        assert math.isclose(prob, 0.42)

    def test_no_defenders_gives_base_probability(self):
        passer = _player(0, 0)
        receiver = _player(15, 0, pid=2)
        prob_no_def = completion_probability(passer, receiver, [])
        prob_far_def = completion_probability(
            passer, receiver,
            [_player(10, 50, pid=3, tid=2)],  # Defender far away
        )
        assert math.isclose(prob_no_def, prob_far_def)
