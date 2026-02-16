"""Tests for the overall Pass Score calculation."""

from threader.models import Player
from threader.scoring.pass_score import score_pass_option


def _player(x: float, y: float, pid: int = 1, tid: int = 1, **kw) -> Player:
    return Player(player_id=pid, team_id=tid, x=x, y=y, **kw)


class TestPassScore:
    def test_returns_pass_option(self):
        passer = _player(0, 0)
        target = _player(20, 0, pid=2)
        defenders = [_player(30, 10, pid=3, tid=2)]
        result = score_pass_option(passer, target, defenders)

        assert result.target == target
        assert 0.0 <= result.pass_score <= 100.0
        assert 0.0 <= result.completion_probability <= 1.0
        assert 0.0 <= result.zone_value <= 0.5
        assert 0.0 <= result.receiving_pressure <= 10.0
        assert result.space_available >= 0.0
        assert 0.0 <= result.penetration_score <= 1.0

    def test_forward_pass_scores_higher(self):
        """A forward pass to a good position should beat a backward pass."""
        passer = _player(0, 0)

        # Forward target near attacking third
        forward = _player(30, 0, pid=2)
        # Backward target near defense
        backward = _player(-30, 0, pid=3)

        defenders = [_player(40, 5, pid=10, tid=2)]

        score_fwd = score_pass_option(passer, forward, defenders)
        score_bwd = score_pass_option(passer, backward, defenders)

        assert score_fwd.pass_score > score_bwd.pass_score

    def test_unblocked_beats_blocked(self):
        """An unblocked pass should score higher than a blocked one."""
        passer = _player(0, 0)
        target = _player(20, 0, pid=2)

        no_block = [_player(10, 30, pid=3, tid=2)]  # Far from lane
        blocked = [_player(10, 0.5, pid=3, tid=2)]   # On the lane

        score_clear = score_pass_option(passer, target, no_block)
        score_blocked = score_pass_option(passer, target, blocked)

        assert score_clear.pass_score > score_blocked.pass_score

    def test_score_non_negative(self):
        """Pass Score should never go below 0."""
        passer = _player(0, 0)
        target = _player(-40, 0, pid=2)  # Deep backward pass
        # Heavy pressure
        defenders = [
            _player(-40, 1, pid=3, tid=2),
            _player(-40, -1, pid=4, tid=2),
            _player(-39, 0, pid=5, tid=2),
        ]
        result = score_pass_option(passer, target, defenders)
        assert result.pass_score >= 0.0


class TestPassScoreZoneValue:
    def test_attacking_position_has_higher_zone_value(self):
        passer = _player(0, 0)

        # Near opponent's goal (right side in PFF coords)
        attacking = _player(45, 0, pid=2)
        # Near own goal
        defensive = _player(-45, 0, pid=3)

        defenders = [_player(48, 5, pid=10, tid=2)]

        opt_att = score_pass_option(passer, attacking, defenders)
        opt_def = score_pass_option(passer, defensive, defenders)

        assert opt_att.zone_value > opt_def.zone_value
