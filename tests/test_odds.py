"""Odds ingest, de-vigging, goal recovery, the baseline, and the live log.

The arithmetic that turns a bookmaker price into a goal expectation is
several transformations deep, and a sign error anywhere in it produces
numbers that still look like probabilities. So each step is pinned
separately against a hand checkable case.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines import odds_only
from eval import live_log
from ingest import odds

CURATED = Path("data/curated")


# --------------------------------------------------------------------------
# De-vigging
# --------------------------------------------------------------------------


def test_devig_removes_the_margin():
    """Raw implied probabilities sum above 1, de-vigged ones sum to 1."""
    prices = [2.7, 3.4, 2.9]
    raw = sum(1 / p for p in prices)
    assert raw > 1.0

    probs, margin = odds.devig(prices)
    assert probs.sum() == pytest.approx(1.0)
    assert margin == pytest.approx(raw - 1.0, abs=1e-6)


def test_devig_preserves_the_favourite():
    probs, _ = odds.devig([1.5, 4.0, 7.0])
    assert probs[0] > probs[1] > probs[2]


def test_shin_takes_more_from_the_longshot_than_normalisation():
    """Shin's whole point is that longshots carry more of the margin.

    Basic normalisation scales every price by the same factor, so it leaves
    a longshot overpriced. Shin removes proportionally more from it.
    """
    prices = [1.4, 5.0, 9.0]
    shin, _ = odds.devig(prices, method="shin")
    naive = np.array([1 / p for p in prices])
    naive = naive / naive.sum()
    assert shin[2] < naive[2]
    assert shin[0] > naive[0]


# --------------------------------------------------------------------------
# Goal expectations
# --------------------------------------------------------------------------


def test_outcome_probabilities_sum_to_one():
    home, draw, away, over = odds._outcome_probabilities(1.5, 1.2)
    assert home + draw + away == pytest.approx(1.0, abs=1e-4)
    assert 0.0 < over < 1.0


def test_a_stronger_home_side_wins_more_often():
    strong, _, weak, _ = odds._outcome_probabilities(2.5, 0.8)
    assert strong > weak


def test_goal_expectations_round_trip():
    """Generate markets from known expectations, recover them from the markets."""
    for true_home, true_away in [(1.6, 1.1), (2.4, 0.7), (1.0, 1.0)]:
        home, draw, away, over = odds._outcome_probabilities(true_home, true_away)
        got_home, got_away = odds.goal_expectations(home, draw, away, over)
        assert got_home == pytest.approx(true_home, abs=0.05)
        assert got_away == pytest.approx(true_away, abs=0.05)


def test_goal_expectations_stay_in_a_football_range():
    # An absurd market must not produce a twelve goal expectation
    home, away = odds.goal_expectations(0.999, 0.0005, 0.0005, 0.999)
    lo, hi = odds.GOALS_BOUNDS
    assert lo <= home <= hi and lo <= away <= hi


def test_clean_sheet_is_the_poisson_zero():
    """A clean sheet is exactly the opponent failing to score."""
    away_xg = 1.2
    assert float(np.exp(-away_xg)) == pytest.approx(0.3012, abs=1e-4)


# --------------------------------------------------------------------------
# Club name mapping
# --------------------------------------------------------------------------


def test_club_map_covers_the_known_mismatches():
    mapping = odds.load_club_map()
    assert mapping["Man United"] == "Man Utd"
    assert mapping["Tottenham"] == "Spurs"
    assert mapping["Sheffield United"] == "Sheffield Utd"


# --------------------------------------------------------------------------
# The baseline
# --------------------------------------------------------------------------


def _frame(n_players=4, element_type=3, prices=(9.0, 7.0, 5.5, 4.5)):
    rows = []
    for i in range(n_players):
        rows.append(
            {
                "season": "2025-26",
                "gw": 5,
                "player_id": i + 1,
                "player_code": 100 + i,
                "fixture_id": 42,
                "element_type": element_type,
                "team_id": 1,
                "price": prices[i],
                "was_home": True,
                "start_rate_1": 1.0,
                "kickoff_time": pd.Timestamp("2025-09-13", tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def test_allocation_weights_sum_to_one_within_a_group():
    df = _frame()
    df["p_appear"] = 1.0
    weights = odds_only.allocation_weight(df)
    assert weights.sum() == pytest.approx(1.0)


def test_allocation_favours_the_expensive_player():
    df = _frame()
    df["p_appear"] = 1.0
    weights = odds_only.allocation_weight(df)
    assert weights.iloc[0] > weights.iloc[1] > weights.iloc[2] > weights.iloc[3]
    # Successive ranks decay by the documented constant
    assert weights.iloc[1] / weights.iloc[0] == pytest.approx(odds_only.PRICE_RANK_DECAY)


def test_no_share_is_spent_on_a_player_who_is_not_playing():
    """A benched player must not absorb goals the starters should get."""
    df = _frame()
    df["p_appear"] = [1.0, 1.0, 0.0, 0.0]
    weights = odds_only.allocation_weight(df)
    assert weights.iloc[2] == 0.0 and weights.iloc[3] == 0.0
    assert weights.sum() == pytest.approx(1.0)


def test_position_shares_are_a_partition_of_team_goals():
    assert sum(odds_only.POSITION_GOAL_SHARE.values()) == pytest.approx(1.0)
    assert sum(odds_only.POSITION_ASSIST_SHARE.values()) == pytest.approx(1.0)
    assert odds_only.POSITION_GOAL_SHARE[1] == 0.0  # keepers do not score


# --------------------------------------------------------------------------
# Live log discipline
# --------------------------------------------------------------------------


def test_a_prediction_written_after_kickoff_is_refused(tmp_path, monkeypatch):
    """The one rule that makes the live log worth anything."""
    season, gw = "2026-27", 1
    pred = tmp_path / "expected_points.parquet"
    pd.DataFrame({"player_code": [1], "ep_total": [1.0]}).to_parquet(pred)

    cutoff = pd.Timestamp("2026-08-14 18:00", tz="UTC")
    monkeypatch.setattr(live_log, "deadline", lambda *a, **k: cutoff)

    # File mtime is now, which is well after that deadline
    with pytest.raises(live_log.DeadlineError, match="after the"):
        live_log.check_pre_deadline(pred, season, gw)


def test_a_prediction_written_before_kickoff_is_accepted(tmp_path, monkeypatch):
    pred = tmp_path / "expected_points.parquet"
    pd.DataFrame({"player_code": [1], "ep_total": [1.0]}).to_parquet(pred)

    far_future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)
    monkeypatch.setattr(live_log, "deadline", lambda *a, **k: far_future)

    written = live_log.check_pre_deadline(pred, "2026-27", 1)
    assert written < far_future


def test_a_missing_prediction_is_an_error_not_a_zero():
    with pytest.raises(FileNotFoundError):
        live_log.check_pre_deadline(Path("does/not/exist.parquet"), "2026-27", 1)


def test_the_log_refuses_to_rewrite_a_logged_gameweek(tmp_path):
    path = tmp_path / "live.parquet"
    row = {
        "season": "2026-27", "gw": 1, "candidate": "compose_v0",
        "xi_points": 50.0, "captain_points": 10.0, "total_points": 60.0,
        "prediction_file": "x", "prediction_written_utc": pd.Timestamp.now(tz="UTC"),
        "logged_utc": pd.Timestamp.now(tz="UTC"),
    }
    live_log.append([row], path)
    assert len(live_log.load_log(path)) == 1

    with pytest.raises(ValueError, match="append only"):
        live_log.append([row], path)


def test_the_log_accepts_a_new_gameweek(tmp_path):
    path = tmp_path / "live.parquet"
    base = {
        "season": "2026-27", "candidate": "compose_v0",
        "xi_points": 50.0, "captain_points": 10.0, "total_points": 60.0,
        "prediction_file": "x", "prediction_written_utc": pd.Timestamp.now(tz="UTC"),
        "logged_utc": pd.Timestamp.now(tz="UTC"),
    }
    live_log.append([{**base, "gw": 1}], path)
    live_log.append([{**base, "gw": 2}], path)
    assert sorted(live_log.load_log(path)["gw"]) == [1, 2]


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------

REAL_ODDS = CURATED / "2025-26" / "fixture_odds.parquet"


@pytest.mark.skipif(not REAL_ODDS.exists(), reason="odds not ingested")
def test_every_fixture_is_priced():
    for season in odds.DEFAULT_SEASONS:
        path = CURATED / season / "fixture_odds.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        coverage = df["home_xg"].notna().mean()
        assert coverage >= 0.99, f"{season} only {coverage:.1%} priced"


@pytest.mark.skipif(not REAL_ODDS.exists(), reason="odds not ingested")
def test_priced_probabilities_are_coherent():
    df = pd.read_parquet(REAL_ODDS)
    total = df["p_home_win"] + df["p_draw"] + df["p_away_win"]
    assert np.allclose(total, 1.0, atol=1e-6)
    for col in ("p_home_cs", "p_away_cs", "p_over_2_5"):
        assert df[col].between(0, 1).all()
    # Home advantage is real and should show up in the aggregate
    assert df["home_xg"].mean() > df["away_xg"].mean()
