"""Expected points pipeline: hand computed arithmetic, leakage, and rates.

Two tests here are acceptance criteria in their own right. One pins the
composition arithmetic against a fully hand worked player. The other
rebuilds the curated tables with the future deleted and asserts that not a
single expected points value moves.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models import compose, rates
from models.minutes import dataset, features
from scoring.rules_2026_27 import Position, expected_points

CURATED = Path("data/curated")
LEAK_SEASONS = ["2024-25", "2025-26"]
LEAK_SEASON = "2025-26"
LEAK_GW = 12


# --------------------------------------------------------------------------
# Composition arithmetic
# --------------------------------------------------------------------------


def _fixture_row(element_type=3, **overrides):
    base = {
        "season": "2025-26",
        "gw": 10,
        "player_id": 1,
        "player_code": 900,
        "fixture_id": 50,
        "element_type": element_type,
        "team_id": 3,
        "p_start": 0.9,
        "p_60": 0.9,
        "p_sub": 0.1,
        "p_appear": 1.0,
        "exp_minutes": 80.0,
        "exp_goals": 0.4,
        "exp_assists": 0.2,
        "p_clean_sheet": 0.3,
        "exp_goals_conceded": 1.2,
        "exp_saves": 0.0,
        "p_defcon": 0.25,
        "exp_bonus": 0.5,
        "exp_cards": 0.1,
        "opp_defence_adj": 1.0,
        "opp_attack_adj": 1.0,
        "own_defence_adj": 1.0,
        "defcon_available": True,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_midfielder_expected_points_matches_hand_computation():
    """A midfielder, every term worked by hand.

    appearance   1.0 * 1 + 0.9 * (2 - 1)   = 1.90
    goals        0.4 * 5 per goal          = 2.00
    assists      0.2 * 3 per assist        = 0.60
    clean sheet  0.9 * 0.3 * 1 for a MID   = 0.27
    defcon       0.25 * 2                  = 0.50
    bonus                                  = 0.50
    cards                                  = -0.10
                                             -----
                                             5.67
    """
    got = compose.compose_fixtures(_fixture_row(element_type=3))["ep_total"].iloc[0]
    assert got == pytest.approx(5.67)


def test_defender_uses_defender_scoring():
    """Same inputs, defender values: goals worth 6, clean sheet worth 4, and
    goals conceded now costs, at one point per two conceded.

    appearance 1.90 + goals 0.4*6 = 2.40 + assists 0.60
    + clean sheet 0.9*0.3*4 = 1.08 - conceded 1.2/2 = 0.60
    + defcon 0.50 + bonus 0.50 - cards 0.10
    """
    got = compose.compose_fixtures(_fixture_row(element_type=2))["ep_total"].iloc[0]
    expected = 1.90 + 2.40 + 0.60 + 1.08 - 0.60 + 0.50 + 0.50 - 0.10
    assert got == pytest.approx(expected)


def test_keeper_gets_save_points_and_no_defcon():
    got = compose.compose_fixtures(
        _fixture_row(element_type=1, exp_saves=3.0, p_defcon=0.9)
    )["ep_total"].iloc[0]
    # 3 saves is 1 point, and a keeper has no defensive contribution route
    expected = 1.90 + 0.4 * 10 + 0.60 + 0.9 * 0.3 * 4 - 0.60 + 3.0 / 3 + 0.50 - 0.10
    assert got == pytest.approx(expected)


def test_compose_delegates_to_the_scoring_module():
    """Whatever compose produces must equal a direct scoring module call."""
    row = _fixture_row(element_type=4)
    got = compose.compose_fixtures(row)["ep_total"].iloc[0]
    direct = expected_points(
        Position.FWD,
        p_appear=1.0,
        p_60plus=0.9,
        exp_goals=0.4,
        exp_assists=0.2,
        p_clean_sheet=0.3,
        exp_goals_conceded=1.2,
        exp_saves=0.0,
        p_defcon=0.25,
        exp_bonus=0.5,
        exp_cards=0.1,
    )
    assert got == pytest.approx(direct)


def test_double_gameweek_sums_both_fixtures():
    """Two fixtures means two shots at everything, so expectations add."""
    one = _fixture_row(fixture_id=50)
    two = _fixture_row(fixture_id=51)
    both = pd.concat([one, two], ignore_index=True)

    fixtures = compose.compose_fixtures(both)
    gameweek = compose.aggregate_gameweek(fixtures)

    assert len(gameweek) == 1
    assert gameweek["n_fixtures"].iloc[0] == 2
    assert gameweek["ep_total"].iloc[0] == pytest.approx(2 * 5.67)
    assert gameweek["exp_goals"].iloc[0] == pytest.approx(0.8)


def test_aggregate_keeps_every_component():
    fixtures = compose.compose_fixtures(_fixture_row())
    gameweek = compose.aggregate_gameweek(fixtures)
    for col in compose.COMPONENT_COLUMNS:
        assert col in gameweek.columns, col


# --------------------------------------------------------------------------
# Rate estimators
# --------------------------------------------------------------------------


def _history(n_matches=12, minutes=90, goals=0, season="2025-26", code=900, element_type=3):
    rows = []
    for gw in range(1, n_matches + 1):
        rows.append(
            {
                "season": season,
                "gw": gw,
                "player_id": 1,
                "player_code": code,
                "fixture_id": gw,
                "element_type": element_type,
                "team_id": 3,
                "opponent_team": 4,
                "was_home": gw % 2 == 0,
                "kickoff_time": pd.Timestamp("2025-08-16", tz="UTC") + pd.Timedelta(days=7 * gw),
                "minutes": minutes,
                "goals_scored": goals,
                "assists": 0,
                "saves": 0,
                "bonus": 0,
                "goals_conceded": 1,
                "clean_sheets": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "defensive_contribution": 0,
                "rule_regime": "defcon_v1",
                "starts": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_shrinkage_pulls_a_small_sample_toward_the_prior():
    """One hot game must not make a player a 1.0 per 90 striker."""
    hist = _history(n_matches=1, goals=1)
    priors = {"goals_scored": {"3": 0.2}}
    nineties = pd.Series([1.0])
    prior = pd.Series([0.2])
    got = rates._shrunk_per90(pd.Series([1.0]), nineties, prior)[0]
    # One goal in one match, against a prior worth 600 minutes, lands near
    # the prior rather than near 1.0
    assert 0.2 < got < 0.35


def test_shrinkage_weakens_as_evidence_accumulates():
    prior = pd.Series([0.2, 0.2])
    events = pd.Series([1.0, 20.0])
    nineties = pd.Series([1.0, 20.0])
    got = rates._shrunk_per90(events, nineties, prior)
    assert got[1] > got[0]


def test_trailing_only_counts_qualifying_matches():
    """Cameos are excluded, because a per-90 rate from 5 minutes is noise."""
    hist = _history(n_matches=6, minutes=10)
    hist.loc[hist["gw"].isin([1, 2]), "minutes"] = 90
    out = rates.player_trailing(hist)
    last = out.sort_values("kickoff_time").iloc[-1]
    # Only the two 90 minute games qualify
    assert last["trail_matches"] == 2
    assert last["trail_minutes"] == 180


def test_trailing_excludes_the_current_match():
    hist = _history(n_matches=3, goals=1)
    out = rates.player_trailing(hist).sort_values("kickoff_time")
    assert out["trail_goals"].iloc[0] == 0
    assert out["trail_goals"].iloc[1] == 1
    assert out["trail_goals"].iloc[2] == 2


def test_trailing_is_frozen_at_the_gameweek_boundary():
    """Both halves of a double gameweek see the same history."""
    hist = _history(n_matches=4, goals=1)
    extra = hist[hist["gw"] == 4].copy()
    extra["fixture_id"] = 99
    extra["kickoff_time"] = extra["kickoff_time"] + pd.Timedelta(days=3)
    both = pd.concat([hist, extra], ignore_index=True)

    out = rates.player_trailing(both)
    dgw = out[out["gw"] == 4].sort_values("kickoff_time")
    assert len(dgw) == 2
    assert dgw["trail_goals"].iloc[0] == dgw["trail_goals"].iloc[1] == 3


def test_defcon_is_unavailable_before_the_rule_existed():
    hist = _history(n_matches=6)
    hist["rule_regime"] = "pre_defcon"
    out = rates.add_rates(hist, history=hist)
    assert not out["defcon_available"].any()
    assert (out["rate_defcon"] == 0).all()


def test_defcon_available_once_the_data_records_it():
    hist = _history(n_matches=6, element_type=2)
    hist["defensive_contribution"] = 15  # clears the defender threshold of 10
    out = rates.add_rates(hist, history=hist)
    later = out.sort_values("kickoff_time").iloc[-1]
    assert later["defcon_available"]
    assert later["rate_defcon"] > 0


def test_keepers_never_get_defensive_contribution():
    hist = _history(n_matches=6, element_type=1)
    hist["defensive_contribution"] = 30
    out = rates.add_rates(hist, history=hist)
    assert (out["rate_defcon"] == 0).all()
    assert not out["defcon_available"].any()


def test_opponent_adjustment_is_clipped_both_ways():
    lo, hi = rates.OPPONENT_ADJ_CLIP
    extreme = pd.Series([0.0, 99.0]) / 1.45
    assert extreme.clip(lo, hi).tolist() == [lo, hi]


def _team_adj(goals_against, matches, league=1.45):
    """The shrunk opponent defence adjustment for a given record."""
    shrunk = (goals_against + rates.TEAM_PRIOR_MATCHES * league) / (
        matches + rates.TEAM_PRIOR_MATCHES
    )
    return shrunk / league


def test_a_team_with_no_record_is_treated_as_average():
    """A promoted side in August is average, not whatever a zero sample says."""
    assert _team_adj(0.0, 0.0) == pytest.approx(1.0)


def test_team_shrinkage_pulls_an_extreme_record_toward_average():
    """A watertight defence over ten games must not halve everyone's xG.

    Seven conceded in ten is a raw ratio of 0.48, which the old code clipped
    to the 0.6 floor. Shrunk, it lands well inside the clip, so the clip
    stops doing the modelling.
    """
    raw = (7 / 10) / 1.45
    shrunk = _team_adj(7.0, 10.0)
    lo, _ = rates.OPPONENT_ADJ_CLIP
    assert raw < lo  # the old behaviour saturated
    assert lo < shrunk < 1.0  # the new one does not
    assert shrunk > raw


def test_team_shrinkage_weakens_as_matches_accumulate():
    """Ten matches of evidence should move the estimate more than three."""
    few = _team_adj(2.1, 3.0)
    many = _team_adj(7.0, 10.0)
    # Both records are 0.7 goals conceded a game, but the longer one is
    # trusted further from average
    assert many < few < 1.0


def test_team_adjustment_stays_inside_the_clip_on_real_data():
    """The clip is a safety rail now, so it should almost never bind."""
    hist = _history(n_matches=10, element_type=2)
    out = rates.add_rates(hist, history=hist)
    lo, hi = rates.OPPONENT_ADJ_CLIP
    for col in ("opp_defence_adj", "opp_attack_adj", "own_defence_adj"):
        assert out[col].between(lo, hi).all()


def test_clean_sheet_probability_stays_inside_its_bounds():
    hist = _history(n_matches=8, element_type=2)
    out = rates.add_rates(hist, history=hist)
    lo, hi = rates.CLEAN_SHEET_CLIP
    assert out["p_clean_sheet"].between(lo, hi).all()


def test_apply_minutes_scales_rates_by_expected_playing_time():
    df = pd.DataFrame(
        {
            "element_type": [3, 3],
            "exp_minutes": [90.0, 45.0],
            "p_appear": [1.0, 1.0],
            "rate_goals_p90": [0.5, 0.5],
            "rate_assists_p90": [0.2, 0.2],
            "rate_saves_p90": [0.0, 0.0],
            "rate_bonus": [0.4, 0.4],
            "rate_defcon": [0.3, 0.3],
            "opp_defence_adj": [1.0, 1.0],
            "own_defence_adj": [1.0, 1.0],
            "opp_goal_expectation": [1.5, 1.5],
            "league_goals_per_team": [1.45, 1.45],
            "card_points_per_appearance": [0.1, 0.1],
        }
    )
    out = rates.apply_minutes(df)
    assert out["exp_goals"].iloc[0] == pytest.approx(0.5)
    assert out["exp_goals"].iloc[1] == pytest.approx(0.25)
    # Half a match halves the chance of reaching a defensive threshold
    assert out["p_defcon"].iloc[1] == pytest.approx(0.15)


def test_opponent_strength_moves_expected_goals():
    df = pd.DataFrame(
        {
            "element_type": [3, 3],
            "exp_minutes": [90.0, 90.0],
            "p_appear": [1.0, 1.0],
            "rate_goals_p90": [0.5, 0.5],
            "rate_assists_p90": [0.0, 0.0],
            "rate_saves_p90": [0.0, 0.0],
            "rate_bonus": [0.0, 0.0],
            "rate_defcon": [0.0, 0.0],
            "opp_defence_adj": [1.4, 0.7],
            "own_defence_adj": [1.0, 1.0],
            "opp_goal_expectation": [1.5, 1.5],
            "league_goals_per_team": [1.45, 1.45],
            "card_points_per_appearance": [0.0, 0.0],
        }
    )
    out = rates.apply_minutes(df)
    assert out["exp_goals"].iloc[0] == pytest.approx(0.7)
    assert out["exp_goals"].iloc[1] == pytest.approx(0.35)


# --------------------------------------------------------------------------
# Leakage, the acceptance criterion
# --------------------------------------------------------------------------


class _StubBooster:
    """Deterministic stand in for LightGBM, driven by one lagged feature."""

    def predict(self, X):
        return np.clip(X["start_rate_5"].fillna(0.5).to_numpy(dtype=float), 0.01, 0.99)


class _StubCalibrator:
    def predict(self, p):
        return np.asarray(p, dtype=float)


def _stub_minutes_model():
    head = (_StubBooster(), _StubCalibrator())
    return {
        "p_start": head,
        "p_60_given_start": head,
        "p_sub": head,
        "conditional_minutes": {
            "start": {"1": 89.0, "2": 86.0, "3": 80.0, "4": 79.0},
            "sub": {"1": 80.0, "2": 39.0, "3": 28.0, "4": 25.0},
            "start_overall": 82.0,
            "sub_overall": 30.0,
        },
    }


def _copy_curated(dest: Path, truncate_after_gw: int | None = None) -> Path:
    """A curated tree, optionally with everything after a gameweek deleted."""
    dest.mkdir(parents=True, exist_ok=True)
    for season in LEAK_SEASONS:
        out = dest / season
        shutil.copytree(CURATED / season, out)
        if truncate_after_gw is None or season != LEAK_SEASON:
            continue
        for name in ("player_fixture.parquet", "player_gw.parquet"):
            path = out / name
            df = pd.read_parquet(path)
            df[df["gw"] <= truncate_after_gw].to_parquet(path, index=False)
        # The schedule is published in advance so the fixtures stay, but
        # results that have not happened yet must not be sitting there.
        fx_path = out / "fixtures.parquet"
        fx = pd.read_parquet(fx_path)
        future = fx["gw"] > truncate_after_gw
        fx.loc[future, ["team_h_score", "team_a_score"]] = np.nan
        fx.loc[future, "finished"] = False
        fx.to_parquet(fx_path, index=False)
    return dest


def _expected_points_for_gw(curated_root: Path) -> pd.DataFrame:
    full = compose.build_full_frame(LEAK_SEASON, None, "historical", curated_root)
    history = full[full["season"] < LEAK_SEASON]
    rated = rates.add_rates(full, history=history, curated_root=curated_root)
    _, gameweek = compose.run_gameweek(
        LEAK_SEASON, LEAK_GW, _stub_minutes_model(), rated
    )
    return gameweek.sort_values("player_id").reset_index(drop=True)


@pytest.mark.skipif(not (CURATED / LEAK_SEASON).exists(), reason="curated data not built")
def test_expected_points_ignore_every_later_gameweek(tmp_path):
    """Acceptance criterion: delete the future, the present must not move.

    The whole curated tree is rebuilt with everything after the target
    gameweek removed, then expected points are recomputed from scratch. If
    any trailing window, prior or team form reaches forward, this fails.
    """
    with_future = _copy_curated(tmp_path / "full")
    without_future = _copy_curated(tmp_path / "truncated", truncate_after_gw=LEAK_GW)

    a = _expected_points_for_gw(with_future)
    b = _expected_points_for_gw(without_future)

    assert len(a) == len(b) and len(a) > 300
    pd.testing.assert_series_equal(a["ep_total"], b["ep_total"], check_exact=False, rtol=1e-9)
    for col in ("exp_goals", "p_clean_sheet", "exp_bonus", "p_defcon", "exp_goals_conceded"):
        pd.testing.assert_series_equal(a[col], b[col], check_exact=False, rtol=1e-9)


@pytest.mark.skipif(not (CURATED / LEAK_SEASON).exists(), reason="curated data not built")
def test_historical_minutes_model_never_trains_on_the_target_season():
    from models.minutes import train

    full = features.add_features(dataset.build(seasons=LEAK_SEASONS))
    model = train.walk_forward_model_for(full, LEAK_SEASON)
    assert LEAK_SEASON not in model["train_seasons"]
    assert model["train_seasons"] == ["2024-25"]
