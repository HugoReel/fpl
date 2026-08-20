"""Pin the 2026/27 scoring rules.

These tests exist so that when you refactor the composition layer, or when
the rules change next August, you find out immediately instead of six weeks
into a backtest.
"""

import pytest

from fpl.scoring.rules_2026_27 import (
    MatchStats,
    Position,
    expected_points,
    score_gameweek,
    score_match,
)


def test_no_minutes_scores_nothing():
    assert score_match(MatchStats(minutes=0), Position.MID).total == 0


def test_appearance_thresholds():
    assert score_match(MatchStats(minutes=1), Position.MID).appearance == 1
    assert score_match(MatchStats(minutes=59), Position.MID).appearance == 1
    assert score_match(MatchStats(minutes=60), Position.MID).appearance == 2
    assert score_match(MatchStats(minutes=90), Position.MID).appearance == 2


@pytest.mark.parametrize(
    "position,expected",
    [(Position.GKP, 10), (Position.DEF, 6), (Position.MID, 5), (Position.FWD, 4)],
)
def test_goal_values_by_position(position, expected):
    b = score_match(MatchStats(minutes=90, goals=1), position)
    assert b.goals == expected


def test_clean_sheet_requires_sixty_minutes():
    played_60 = score_match(MatchStats(minutes=60, goals_conceded=0), Position.DEF)
    played_59 = score_match(MatchStats(minutes=59, goals_conceded=0), Position.DEF)
    assert played_60.clean_sheet == 4
    assert played_59.clean_sheet == 0


def test_clean_sheet_by_position():
    stats = MatchStats(minutes=90, goals_conceded=0)
    assert score_match(stats, Position.GKP).clean_sheet == 4
    assert score_match(stats, Position.DEF).clean_sheet == 4
    assert score_match(stats, Position.MID).clean_sheet == 1
    assert score_match(stats, Position.FWD).clean_sheet == 0


def test_goals_conceded_penalty_only_gkp_and_def():
    stats = MatchStats(minutes=90, goals_conceded=3)
    assert score_match(stats, Position.GKP).goals_conceded == -1
    assert score_match(stats, Position.DEF).goals_conceded == -1
    assert score_match(stats, Position.MID).goals_conceded == 0
    stats4 = MatchStats(minutes=90, goals_conceded=4)
    assert score_match(stats4, Position.DEF).goals_conceded == -2


def test_saves_round_down_in_threes():
    for saves, pts in [(0, 0), (2, 0), (3, 1), (8, 2), (9, 3)]:
        b = score_match(MatchStats(minutes=90, saves=saves), Position.GKP)
        assert b.saves == pts, f"{saves} saves"


def test_defcon_thresholds_and_cap():
    # Defender needs 10 CBIT
    assert score_match(MatchStats(minutes=90, defensive_actions=9), Position.DEF).defcon == 0
    assert score_match(MatchStats(minutes=90, defensive_actions=10), Position.DEF).defcon == 2
    # Double the threshold is still only 2, not 4
    assert score_match(MatchStats(minutes=90, defensive_actions=20), Position.DEF).defcon == 2
    # Midfielders and forwards need 12 CBIRT
    assert score_match(MatchStats(minutes=90, defensive_actions=11), Position.MID).defcon == 0
    assert score_match(MatchStats(minutes=90, defensive_actions=12), Position.MID).defcon == 2
    assert score_match(MatchStats(minutes=90, defensive_actions=12), Position.FWD).defcon == 2
    # Goalkeepers do not get DefCon at all
    assert score_match(MatchStats(minutes=90, defensive_actions=30), Position.GKP).defcon == 0


def test_full_realistic_defender_haul():
    # 90 minutes, goal, clean sheet, DefCon threshold hit, 3 bonus
    b = score_match(
        MatchStats(minutes=90, goals=1, goals_conceded=0, defensive_actions=11, bonus=3),
        Position.DEF,
    )
    # 2 appearance + 6 goal + 4 clean sheet + 2 defcon + 3 bonus
    assert b.total == 17


def test_second_yellow_red_stacks():
    b = score_match(MatchStats(minutes=70, yellow_cards=1, red_cards=1), Position.MID)
    assert b.cards == -4


def test_double_gameweek_scores_each_fixture_separately():
    # Two 45 minute cameos are two short appearances, not one long one
    gw = score_gameweek(
        [MatchStats(minutes=45), MatchStats(minutes=45)], Position.MID
    )
    assert gw.appearance == 2
    assert gw.total == 2


def test_element_type_int_accepted():
    assert score_match(MatchStats(minutes=90, goals=1), 1).goals == 10


def test_expected_points_composition_matches_intuition():
    # Nailed premium midfielder, decent fixture
    ep = expected_points(
        Position.MID,
        p_appear=0.97,
        p_60plus=0.92,
        exp_goals=0.45,
        exp_assists=0.30,
        p_clean_sheet=0.30,
        exp_bonus=0.55,
        exp_cards=0.12,
    )
    assert 4.5 < ep < 7.0, ep


def test_expected_points_goalkeeper_uses_saves_and_concessions():
    ep = expected_points(
        Position.GKP,
        p_appear=1.0,
        p_60plus=1.0,
        exp_goals=0.0,
        exp_assists=0.0,
        p_clean_sheet=0.32,
        exp_goals_conceded=1.3,
        exp_saves=3.1,
        exp_bonus=0.4,
    )
    # 2 appearance + 1.28 CS - 0.65 conceded + 1.03 saves + 0.4 bonus
    assert ep == pytest.approx(2 + 1.28 - 0.65 + 3.1 / 3 + 0.4, abs=1e-6)
