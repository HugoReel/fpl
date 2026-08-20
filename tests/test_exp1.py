"""Experiment 1: hand computed checks on the expected points construction.

The experiment's conclusions are only worth as much as its arithmetic, so
one fully hand worked player case is pinned here, along with the trailing
rate machinery that feeds it.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import exp1_minutes_attribution as exp1
from scoring.rules_2026_27 import MatchStats, Position, score_match


def _row(**overrides):
    """A midfielder with round trailing rates, easy to hand compute."""
    base = {
        "element_type": 3,
        "minutes": 90,
        "goals_scored": 0,
        "assists": 0,
        "goals_conceded": 0,
        "saves": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "own_goals": 0,
        "bonus": 0,
        "clean_sheets": 0,
        "defensive_contribution": 0,
        "_hit_defcon": 0.0,
        "rate_goals_p90": 0.5,
        "rate_assists_p90": 0.25,
        "rate_saves_p90": 0.0,
        "rate_conceded_p90": 0.0,
        "rate_clean_sheet": 0.3,
        "rate_defcon": 0.2,
        "rate_bonus": 0.5,
    }
    base.update(overrides)
    return base


def test_expected_points_for_one_hand_built_player():
    """A nailed midfielder, every term computed by hand.

    appearance   1.0 * 1 + 1.0 * (2 - 1)      = 2.00
    goals        0.5 goals * 5 per goal        = 2.50
    assists      0.25 assists * 3 per assist   = 0.75
    clean sheet  1.0 * 0.3 * 1 for a MID       = 0.30
    defcon       0.2 * 2                       = 0.40
    bonus                                      = 0.50
    cards        minus 0.09 per appearance     = -0.09
                                                 -----
                                                 6.36
    """
    got = exp1._ep(_row(), p_appear=1.0, p_60=1.0, minutes_scale=1.0, rate_source="trailing")
    assert got == pytest.approx(6.36)


def test_expected_points_scales_rates_by_minutes():
    """Half the expected minutes means half the expected goals."""
    full = exp1._ep(_row(), 1.0, 1.0, 1.0, "trailing")
    half = exp1._ep(_row(), 1.0, 1.0, 0.5, "trailing")
    # goals and assists halve, worth 0.5 * 5 * 0.5 + 0.25 * 3 * 0.5 = 1.625
    assert full - half == pytest.approx(0.5 * 5 * 0.5 + 0.25 * 3 * 0.5 + 0.2 * 2 * 0.5 + 0.5 * 0.5)


def test_a_player_who_cannot_play_scores_nothing():
    got = exp1._ep(_row(), p_appear=0.0, p_60=0.0, minutes_scale=0.0, rate_source="trailing")
    # Only the clean sheet term survives, and it is gated on p_60 which is 0
    assert got == pytest.approx(0.0)


def test_oracle_outcome_variant_uses_this_fixtures_truth():
    row = _row(goals_scored=2, assists=1, bonus=3, clean_sheets=1, _hit_defcon=1.0)
    got = exp1._ep(row, 1.0, 1.0, 1.0, "outcome")
    # 2 appearance + 2 goals * 5 + 1 assist * 3 + 1 clean sheet + 2 defcon
    # + 3 bonus - 0.09 cards
    assert got == pytest.approx(2 + 10 + 3 + 1 + 2 + 3 - 0.09)


def test_realised_points_matches_the_scoring_module():
    row = _row(minutes=90, goals_scored=1, assists=1, bonus=2, defensive_contribution=12)
    got = exp1.realised_points(row)
    expected = score_match(
        MatchStats(minutes=90, goals=1, assists=1, bonus=2, defensive_actions=12),
        Position.MID,
    ).total
    assert got == expected == 15


def test_realised_points_is_recomputed_not_read_from_the_table():
    """A wrong stored total must not survive into the experiment."""
    row = _row(minutes=90, goals_scored=1, total_points=999)
    assert exp1.realised_points(row) != 999


# --------------------------------------------------------------------------
# Trailing rate machinery
# --------------------------------------------------------------------------


def _series_frame(values):
    return pd.DataFrame(
        {
            "player_code": [1] * len(values),
            "kickoff_time": pd.date_range("2025-08-16", periods=len(values), freq="7D", tz="UTC"),
            "goals_scored": values,
        }
    )


def test_prev_mean_excludes_the_current_match():
    df = _series_frame([1, 0, 2, 0])
    got = exp1._prev_mean(df, "goals_scored", window=3)
    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == pytest.approx(1.0)
    assert got.iloc[2] == pytest.approx(0.5)
    assert got.iloc[3] == pytest.approx(1.0)


def test_season_rates_are_per_90_over_the_whole_season():
    test = pd.DataFrame(
        {
            "player_code": [7, 7, 7],
            "minutes": [90, 90, 90],
            "goals_scored": [1, 0, 2],
            "assists": [0, 0, 0],
            "saves": [0, 0, 0],
            "goals_conceded": [0, 0, 0],
            "bonus": [3, 0, 0],
            "clean_sheets": [1, 0, 0],
            "_hit_defcon": [1.0, 0.0, 0.0],
        }
    )
    rates = exp1.season_rates(test).iloc[0]
    # 3 goals in 270 minutes is 1.0 per 90
    assert rates["srate_goals_p90"] == pytest.approx(1.0)
    assert rates["srate_bonus"] == pytest.approx(1.0)
    assert rates["srate_clean_sheet"] == pytest.approx(1 / 3)
    assert rates["srate_defcon"] == pytest.approx(1 / 3)


def test_summarise_reports_every_variant():
    n = 20
    test = pd.DataFrame({"realised": np.arange(n, dtype=float), "element_type": [1, 2, 3, 4] * 5})
    for name in exp1.VARIANTS:
        test[name] = np.arange(n, dtype=float) + 1.0
    summary = exp1.summarise(test)
    assert list(summary["variant"]) == list(exp1.VARIANTS)
    assert summary["MAE"].tolist() == pytest.approx([1.0] * len(exp1.VARIANTS))
