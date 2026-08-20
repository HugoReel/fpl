"""Dixon-Coles, the team source chain, and the fallback that must actually work.

The fallback tests are the load bearing ones. Once compose depends on
weekly odds, a missed pull has to degrade to Dixon-Coles rather than to a
null, and a degrade path nothing exercises is a degrade path that does not
work. So the no-odds fixture case is asserted directly rather than assumed
from the fact that a coalesce was written.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models import rates
from models.team import dixon_coles

CURATED = Path("data/curated")


def _matches(teams=("A", "B", "C", "D", "E", "F"), repeats=8, seed=11):
    """A round robin with goals drawn from real team strengths.

    Scores are sampled rather than fixed, because a deterministic scoreline
    is degenerate enough that the fitted correlation parameter runs off to
    an invalid value and the fit stops resembling anything real.
    """
    rng = np.random.default_rng(seed)
    strength = {t: 1.6 - 0.18 * i for i, t in enumerate(teams)}  # A strongest, F weakest
    rows = []
    day = pd.Timestamp("2024-08-16", tz="UTC")
    fixture = 0
    for rep in range(repeats):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                fixture += 1
                rows.append(
                    {
                        "season": "2024-25",
                        "gw": rep + 1,
                        "fixture_id": fixture,
                        "kickoff_time": day + pd.Timedelta(days=7 * rep),
                        "home_name": home,
                        "away_name": away,
                        "team_h_score": int(rng.poisson(strength[home] * 1.15)),
                        "team_a_score": int(rng.poisson(strength[away] * 0.85)),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def test_fit_produces_the_expected_parameters():
    model, n = dixon_coles.fit(_matches())
    params = model.get_params()
    assert n > 0
    assert "home_advantage" in params and "rho" in params
    assert "attack_A" in params and "defence_A" in params


def test_the_stronger_team_gets_the_higher_goal_expectation():
    model, _ = dixon_coles.fit(_matches())
    params = model.get_params()
    strong = dixon_coles.predict_fixture(params, "A", "D")
    weak = dixon_coles.predict_fixture(params, "D", "A")
    assert strong["home_xg"] > weak["home_xg"]
    assert strong["p_home_win"] > strong["p_away_win"]


def test_outcome_probabilities_sum_to_one():
    model, _ = dixon_coles.fit(_matches())
    p = dixon_coles.predict_fixture(model.get_params(), "A", "B")
    assert p["p_home_win"] + p["p_draw"] + p["p_away_win"] == pytest.approx(1.0, abs=1e-6)
    for key in ("p_home_cs", "p_away_cs"):
        assert 0.0 <= p[key] <= 1.0


def test_clean_sheet_comes_from_the_corrected_grid_not_a_poisson_zero():
    """The low score correction is the whole reason to use Dixon-Coles."""
    model, _ = dixon_coles.fit(_matches())
    params = model.get_params()
    p = dixon_coles.predict_fixture(params, "A", "B")
    naive = float(np.exp(-p["away_xg"]))
    # rho is fitted, so the two only coincide by accident
    assert p["p_home_cs"] != pytest.approx(naive, abs=1e-6)


# --------------------------------------------------------------------------
# Promoted teams
# --------------------------------------------------------------------------


def test_an_unseen_team_does_not_crash_the_prediction():
    """penaltyblog raises on an unknown club, so the wrapper has to catch it."""
    model, _ = dixon_coles.fit(_matches())
    params = model.get_params()

    with pytest.raises(ValueError):
        model.predict("Promoted FC", "A")  # the library's own behaviour

    got = dixon_coles.predict_fixture(params, "Promoted FC", "A")
    assert np.isfinite(got["home_xg"]) and got["home_xg"] > 0
    assert got["home_is_promoted"] and not got["away_is_promoted"]


def test_a_promoted_team_is_treated_like_the_weakest_tier():
    model, _ = dixon_coles.fit(_matches())
    params = model.get_params()
    attack, defence = dixon_coles.promoted_prior(params)

    all_attack = [v for k, v in params.items() if k.startswith("attack_")]
    assert attack <= float(np.mean(all_attack))

    promoted = dixon_coles.predict_fixture(params, "Promoted FC", "B")
    strong = dixon_coles.predict_fixture(params, "A", "B")
    assert promoted["home_xg"] < strong["home_xg"]


# --------------------------------------------------------------------------
# Source chain and the fallback
# --------------------------------------------------------------------------


def _write_source(root: Path, season: str, name: str, fixture_ids: list[int], xg: float):
    d = root / season
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "season": season,
            "fixture_id": fixture_ids,
            "home_xg": xg,
            "away_xg": xg / 2,
            "p_home_cs": 0.4,
            "p_away_cs": 0.2,
        }
    ).to_parquet(d / name, index=False)


def test_the_preferred_source_wins_where_it_has_coverage(tmp_path):
    _write_source(tmp_path, "2025-26", "fixture_odds.parquet", [1, 2], xg=2.0)
    _write_source(tmp_path, "2025-26", "team_model.parquet", [1, 2], xg=9.0)

    got = rates.load_team_expectations(["2025-26"], ["market", "dixon_coles"], tmp_path)
    assert set(got["team_source"]) == {"market"}
    assert (got["home_xg"] == 2.0).all()


def test_dixon_coles_covers_the_fixtures_the_market_missed(tmp_path):
    """A fixture with no priced market must still get a number."""
    _write_source(tmp_path, "2025-26", "fixture_odds.parquet", [1], xg=2.0)
    _write_source(tmp_path, "2025-26", "team_model.parquet", [1, 2, 3], xg=9.0)

    got = rates.load_team_expectations(["2025-26"], ["market", "dixon_coles"], tmp_path).set_index("fixture_id")
    assert got.loc[1, "team_source"] == "market"
    assert got.loc[2, "team_source"] == "dixon_coles"
    assert got.loc[3, "team_source"] == "dixon_coles"


def test_no_sources_at_all_returns_empty_rather_than_raising(tmp_path):
    got = rates.load_team_expectations(["2025-26"], ["market", "dixon_coles"], tmp_path)
    assert got.empty


def test_an_unknown_team_source_is_rejected_loudly():
    df = pd.DataFrame({"season": ["2025-26"], "fixture_id": [1]})
    with pytest.raises(ValueError, match="unknown team_source"):
        rates._apply_team_source(df, "crystal_ball", CURATED)


# --------------------------------------------------------------------------
# End to end through add_rates
# --------------------------------------------------------------------------


def _player_rows(fixture_ids=(1, 2)):
    rows = []
    for i, fx in enumerate(fixture_ids):
        rows.append(
            {
                "season": "2025-26",
                "gw": i + 1,
                "player_id": 1,
                "player_code": 500,
                "fixture_id": fx,
                "element_type": 2,
                "team_id": 3,
                "opponent_team": 4,
                "was_home": True,
                "kickoff_time": pd.Timestamp("2025-08-16", tz="UTC") + pd.Timedelta(days=7 * i),
                "minutes": 90,
                "goals_scored": 0,
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


def test_market_clean_sheet_reaches_the_player_rows(tmp_path):
    _write_source(tmp_path, "2025-26", "fixture_odds.parquet", [1, 2], xg=2.0)
    df = _player_rows()
    out = rates.add_rates(df, history=df, curated_root=tmp_path, team_source="market")
    # p_home_cs of 0.4 was written above, and these rows are all home
    assert out["p_clean_sheet"].tolist() == pytest.approx([0.4, 0.4])
    assert set(out["team_source"]) == {"market"}


def test_a_fixture_with_no_source_keeps_the_trailing_estimate(tmp_path):
    """The final fallback: no market, no model, still a usable number."""
    _write_source(tmp_path, "2025-26", "fixture_odds.parquet", [1], xg=2.0)
    df = _player_rows(fixture_ids=(1, 2))
    out = rates.add_rates(df, history=df, curated_root=tmp_path, team_source="market")

    by_fixture = out.set_index("fixture_id")
    assert by_fixture.loc[1, "team_source"] == "market"
    assert by_fixture.loc[2, "team_source"] == "trailing"
    lo, hi = rates.CLEAN_SHEET_CLIP
    assert lo <= by_fixture.loc[2, "p_clean_sheet"] <= hi
    assert np.isfinite(by_fixture.loc[2, "opp_goal_expectation"])


def test_trailing_source_is_unchanged_by_the_new_code_path(tmp_path):
    """The incumbent must behave exactly as before, or the swap is not paired."""
    df = _player_rows()
    out = rates.add_rates(df, history=df, curated_root=tmp_path, team_source="trailing")
    assert set(out["team_source"]) == {"trailing"}
    assert out["opp_goal_expectation"].equals(out["opp_gf_shrunk"])


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------

REAL_DC = CURATED / "2025-26" / "team_model.parquet"


@pytest.mark.skipif(not REAL_DC.exists(), reason="team model not fitted")
def test_real_team_model_is_coherent():
    df = pd.read_parquet(REAL_DC)
    total = df["p_home_win"] + df["p_draw"] + df["p_away_win"]
    assert np.allclose(total, 1.0, atol=1e-6)
    assert df["home_xg"].between(0.1, 6.0).all()
    assert df["home_xg"].mean() > df["away_xg"].mean()  # home advantage


@pytest.mark.skipif(not REAL_DC.exists(), reason="team model not fitted")
def test_every_real_fixture_gets_a_team_expectation():
    """Market plus Dixon-Coles together must leave no fixture uncovered."""
    got = rates.load_team_expectations(["2025-26"], ["market", "dixon_coles"], CURATED)
    fixtures = pd.read_parquet(CURATED / "2025-26" / "fixtures.parquet")
    assert len(got) == len(fixtures)
    assert got["home_xg"].notna().all()
