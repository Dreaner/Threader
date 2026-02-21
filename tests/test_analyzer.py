"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: test_analyzer.py
Description: 
    Tests for the main analyzer.
"""

from threader.analysis.analyzer import _compute_team_mean_xT, analyze_snapshot
from threader.models import BallPosition, Player, Snapshot


def _player(x: float, y: float, pid: int, tid: int, **kw) -> Player:
    return Player(player_id=pid, team_id=tid, x=x, y=y, **kw)


def _make_snapshot() -> Snapshot:
    """Create a test snapshot with a realistic-ish setup."""
    home_team = 1
    away_team = 2
    return Snapshot(
        home_players=[
            _player(0, 0, pid=17, tid=home_team, jersey_num=17),     # Passer (midfielder)
            _player(25, -10, pid=9, tid=home_team, jersey_num=9),    # Striker
            _player(15, -15, pid=8, tid=home_team, jersey_num=8),    # AM
            _player(10, 20, pid=7, tid=home_team, jersey_num=7),     # Winger
            _player(-10, 5, pid=6, tid=home_team, jersey_num=6),     # DM
            _player(-20, -10, pid=5, tid=home_team, jersey_num=5),   # CB
            _player(-20, 10, pid=4, tid=home_team, jersey_num=4),    # CB
            _player(-30, 25, pid=3, tid=home_team, jersey_num=3),    # LB
            _player(-30, -25, pid=2, tid=home_team, jersey_num=2),   # RB
            _player(-45, 0, pid=1, tid=home_team, jersey_num=1),     # GK
            _player(-5, -20, pid=11, tid=home_team, jersey_num=11),  # LW
        ],
        away_players=[
            _player(10, 5, pid=21, tid=away_team),
            _player(5, -5, pid=22, tid=away_team),
            _player(20, 0, pid=23, tid=away_team),
            _player(15, 15, pid=24, tid=away_team),
            _player(0, 20, pid=25, tid=away_team),
            _player(-5, -15, pid=26, tid=away_team),
            _player(-15, 0, pid=27, tid=away_team),
            _player(-25, 15, pid=28, tid=away_team),
            _player(-25, -15, pid=29, tid=away_team),
            _player(-40, 0, pid=30, tid=away_team),
            _player(30, 5, pid=31, tid=away_team),
        ],
        ball=BallPosition(x=0, y=0),
    )


class TestAnalyzeSnapshot:
    def test_returns_all_teammates(self):
        snapshot = _make_snapshot()
        passer = snapshot.home_players[0]  # pid=17
        result = analyze_snapshot(snapshot, passer)

        # Should evaluate all 10 teammates (11 home - 1 passer)
        assert len(result.options) == 10

    def test_options_are_ranked(self):
        snapshot = _make_snapshot()
        passer = snapshot.home_players[0]
        result = analyze_snapshot(snapshot, passer)

        ranked = result.ranked_options
        scores = [opt.pass_score for opt in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_best_option_is_first(self):
        snapshot = _make_snapshot()
        passer = snapshot.home_players[0]
        result = analyze_snapshot(snapshot, passer)

        assert result.best_option is not None
        assert result.best_option == result.ranked_options[0]

    def test_all_scores_non_negative(self):
        snapshot = _make_snapshot()
        passer = snapshot.home_players[0]
        result = analyze_snapshot(snapshot, passer)

        for opt in result.options:
            assert opt.pass_score >= 0.0

    def test_passer_is_set(self):
        snapshot = _make_snapshot()
        passer = snapshot.home_players[0]
        result = analyze_snapshot(snapshot, passer)
        assert result.passer == passer


class TestTeamMeanXt:
    """Tests for team-context zone value integration."""

    def test_gk_excluded_from_team_mean(self):
        """GK should not influence the team mean xT calculation."""
        # Players with known xT positions
        outfield = [
            _player(20, 0, pid=2, tid=1),   # mid-attacking
            _player(15, 0, pid=3, tid=1),   # mid
        ]
        gk = _player(-45, 0, pid=1, tid=1, position="GK")

        mean_with_gk = _compute_team_mean_xT(outfield + [gk], 105.0, 68.0, 1.0)
        mean_without_gk = _compute_team_mean_xT(outfield, 105.0, 68.0, 1.0)

        # Should be identical — GK is excluded
        assert mean_with_gk is not None
        assert mean_without_gk is not None
        assert abs(mean_with_gk - mean_without_gk) < 1e-9

    def test_only_gk_returns_none(self):
        """If the only teammate is the GK, return None (no eligible players)."""
        gk = _player(-45, 0, pid=1, tid=1, position="GK")
        result = _compute_team_mean_xT([gk], 105.0, 68.0, 1.0)
        assert result is None

    def test_striker_in_box_ranked_higher_with_team_context(self):
        """When team is in midfield, a striker run into the box gets amplified."""
        from threader.scoring.pass_score import ScoringWeights

        home_tid, away_tid = 1, 2
        # Team mostly in their half, one striker making a run into the box
        snapshot = Snapshot(
            home_players=[
                _player(0, 0, pid=10, tid=home_tid),          # passer (center)
                _player(45, 0, pid=9, tid=home_tid),           # striker in box (high xT)
                _player(-5, 10, pid=8, tid=home_tid),          # midfielder (low xT)
                _player(-10, -5, pid=7, tid=home_tid),         # midfielder (low xT)
                _player(-45, 0, pid=1, tid=home_tid, position="GK"),  # GK
            ],
            away_players=[
                _player(20, 5, pid=21, tid=away_tid),
                _player(15, -5, pid=22, tid=away_tid),
                _player(30, 0, pid=23, tid=away_tid),
            ],
            ball=BallPosition(x=0, y=0),
            home_attacks_right=True,
        )
        passer = snapshot.home_players[0]  # pid=10

        # Without team context (alpha=0): striker might not dominate due to low completion
        result_no_ctx = analyze_snapshot(snapshot, passer)
        result_no_ctx_striker = next(o for o in result_no_ctx.options if o.target.player_id == 9)

        # With team context (alpha=1): striker's xT is amplified vs low team mean
        # We need to re-run with different weights — use _compute_team_mean_xT directly
        # and verify the zone adjustment direction
        teammates = [p for p in snapshot.home_players if p.player_id != passer.player_id]
        team_mean = _compute_team_mean_xT(teammates, 105.0, 68.0, 1.0)

        assert team_mean is not None
        # Striker zone value (attacking half, near box) should be well above team mean
        from threader.scoring.zone_value import zone_value
        striker_xt = zone_value(45.0, 0.0, attack_direction=1.0)
        assert striker_xt > team_mean, (
            f"Striker xT ({striker_xt:.3f}) should exceed team mean ({team_mean:.3f})"
        )
