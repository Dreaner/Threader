"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: test_pass_score.py
Description: 
    Tests for the overall Pass Score calculation.
"""

from threader.models import Player
from threader.scoring.pass_score import score_pass_option
from threader.scoring.penetration import penetration_score
from threader.scoring.space import space_available


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
        assert -0.3 <= result.penetration_score <= 1.0

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
        target = _player(0, 15, pid=2)  # Lateral pass — no forward gain

        no_block = [_player(10, 30, pid=3, tid=2)]  # Far from lane
        blocked = [_player(0, 7, pid=3, tid=2)]      # On the lane

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


class TestBackwardPassPenalty:
    """Backward passes should receive a mild scoring drag."""

    def test_penetration_negative_for_backward(self):
        passer = _player(0, 0)
        backward = _player(-20, 0, pid=2)
        defenders = [_player(30, 10, pid=3, tid=2)]
        score = penetration_score(passer, backward, defenders)
        assert score < 0, "Backward passes should have negative penetration"

    def test_penetration_capped_at_minus_03(self):
        passer = _player(0, 0)
        far_back = _player(-50, 0, pid=2)
        score = penetration_score(passer, far_back, [])
        assert score == -0.3, "Backward drag should cap at -0.3"

    def test_gk_backpass_ranks_below_midfield(self):
        """GK backpass should rank lower than a moderately-pressured midfielder."""
        passer = _player(-10, 0)  # CB with the ball

        gk = _player(-45, 0, pid=2, position="GK")
        midfielder = _player(5, 10, pid=3)

        defenders = [
            _player(8, 12, pid=10, tid=2),   # Near midfielder
            _player(10, 8, pid=11, tid=2),
            _player(20, 0, pid=12, tid=2),
        ]

        score_gk = score_pass_option(passer, gk, defenders)
        score_mid = score_pass_option(passer, midfielder, defenders)

        assert score_mid.pass_score > score_gk.pass_score, (
            f"Midfielder ({score_mid.pass_score}) should outrank GK ({score_gk.pass_score})"
        )


class TestSpaceCap:
    """Space available should be capped at 15m."""

    def test_space_capped_at_15m(self):
        receiver = _player(-45, 0, pid=1)
        # Defender very far away
        opponents = [_player(30, 0, pid=10, tid=2)]
        space = space_available(receiver, opponents)
        assert space == 15.0

    def test_space_uncapped_below_15(self):
        receiver = _player(0, 0, pid=1)
        opponents = [_player(10, 0, pid=10, tid=2)]
        space = space_available(receiver, opponents)
        assert space == 10.0


class TestAttackDirection:
    """Scoring should respect the team's attacking direction."""

    def test_left_attacking_team_zone_flipped(self):
        """When attack_direction=-1, a player at positive x should have LOW zone."""
        from threader.scoring.zone_value import zone_value

        # Player at x=40 (would be attacking zone if direction=+1)
        val_right = zone_value(40, 0, attack_direction=1.0)
        val_left = zone_value(40, 0, attack_direction=-1.0)

        # When attacking left, x=40 is the defensive zone
        assert val_right > val_left

    def test_penetration_respects_direction(self):
        """A 'forward' pass for a left-attacking team is towards negative x."""
        passer = _player(10, 0)
        receiver = _player(-10, 0, pid=2)  # Towards negative x

        # With attack_direction=-1 (attacks left), this is forward
        pen_left = penetration_score(passer, receiver, [], attack_direction=-1.0)
        # With attack_direction=+1 (attacks right), this is backward
        pen_right = penetration_score(passer, receiver, [], attack_direction=1.0)

        assert pen_left > 0, "Should be positive (forward for left-attacking team)"
        assert pen_right < 0, "Should be negative (backward for right-attacking team)"

    def test_away_team_gk_at_positive_x_ranks_low(self):
        """Full integration: away team GK at positive x (their goal) should rank last."""
        from threader.models import BallPosition, Snapshot
        from threader.analysis.analyzer import analyze_snapshot

        home_tid, away_tid = 1, 2
        # Away team attacks LEFT (negative x).
        # Their GK is at positive x (behind their defense).
        snapshot = Snapshot(
            home_players=[
                _player(20, 0, pid=50, tid=home_tid),   # opponent near midfield
                _player(30, 5, pid=51, tid=home_tid),
            ],
            away_players=[
                _player(0, 0, pid=10, tid=away_tid),      # passer (midfield)
                _player(-15, 5, pid=11, tid=away_tid),     # forward option
                _player(-10, -10, pid=12, tid=away_tid),   # another forward
                _player(40, 0, pid=13, tid=away_tid, position="GK"),  # GK far back
            ],
            ball=BallPosition(x=0, y=0),
            home_attacks_right=True,  # so away attacks left
            period=1,
        )

        passer = snapshot.away_players[0]  # pid=10
        result = analyze_snapshot(snapshot, passer)
        ranked = result.ranked_options

        # GK should be ranked last
        gk_opt = [o for o in ranked if o.target.position == "GK"][0]
        assert ranked[-1] == gk_opt, (
            f"GK should be last, but ranked {ranked.index(gk_opt)+1}/{len(ranked)}"
        )
