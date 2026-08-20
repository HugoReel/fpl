"""Minutes model: labels, lagged features, leakage, and the exp minutes maths.

The leakage test is the one that matters. If a feature for gameweek g can
see gameweek g+1, every metric in the report becomes a lie and the failure
is silent, so it is asserted directly rather than inferred from performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.minutes import dataset, features, train

SEASON = "2024-25"
NEXT_SEASON = "2025-26"


def make_frame(n_gw=8, players=(101, 102, 103), season=SEASON, start_gw=1):
    """Synthetic player-fixture rows, one fixture per gameweek per player.

    Player 101 starts every week, 102 never plays, 103 alternates. That
    makes the expected rolling values obvious by hand.
    """
    rows = []
    for gw in range(start_gw, start_gw + n_gw):
        for i, code in enumerate(players):
            if code == 101:
                minutes, starts = 90, 1
            elif code == 102:
                minutes, starts = 0, 0
            else:
                started = gw % 2 == 1
                minutes, starts = (90, 1) if started else (0, 0)
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "player_id": code - 100,
                    "player_code": code,
                    "fixture_id": gw * 10 + i,
                    "element_type": 3,
                    "team_id": 1 + i,
                    "opponent_team": 20 - i,
                    "was_home": gw % 2 == 0,
                    "kickoff_time": pd.Timestamp("2024-08-16", tz="UTC") + pd.Timedelta(days=7 * gw),
                    "minutes": minutes,
                    "starts": float(starts),
                    "price": 5.0 + i,
                    "team_fixtures_in_gw": 1,
                }
            )
    df = pd.DataFrame(rows)
    return dataset.add_labels(df)


def featured(df):
    return features.add_features(df, prior_season=pd.DataFrame())


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_started_prefers_exact_flag_over_minutes_proxy():
    df = pd.DataFrame(
        {
            # a starter subbed off at 40 minutes: the proxy would say no
            "starts": [1.0, 0.0, np.nan, np.nan],
            "minutes": [40, 70, 50, 20],
        }
    )
    out = dataset.add_labels(df)
    assert list(out["started"]) == [True, False, True, False]
    assert list(out["started_is_exact"]) == [True, True, False, False]


def test_sub_appear_and_played_60():
    df = pd.DataFrame({"starts": [0.0, 0.0, 1.0], "minutes": [0, 25, 90]})
    out = dataset.add_labels(df)
    assert list(out["sub_appear"]) == [False, True, False]
    assert list(out["played_60"]) == [False, False, True]


# --------------------------------------------------------------------------
# Feature lagging
# --------------------------------------------------------------------------


def test_rolling_features_use_only_earlier_matches():
    f = featured(make_frame())
    ever_present = f[f["player_code"] == 101].sort_values("gw")

    # First match has no history at all
    assert pd.isna(ever_present["start_rate_1"].iloc[0])
    assert pd.isna(ever_present["minutes_mean_1"].iloc[0])
    # Afterwards an ever present player is a run of ones, never including
    # the current row (which is what makes it a forecast)
    assert (ever_present["start_rate_1"].iloc[1:] == 1.0).all()
    assert (ever_present["minutes_mean_3"].iloc[1:] == 90.0).all()

    alternating = f[f["player_code"] == 103].sort_values("gw")
    # gw1 start, gw2 blank, gw3 start... so at gw3 the last two matches
    # average one start in two
    assert alternating["start_rate_1"].iloc[2] == 0.0
    assert alternating["start_rate_3"].iloc[2] == pytest.approx(0.5)


def test_career_matches_and_new_signing_flag():
    f = featured(make_frame())
    p = f[f["player_code"] == 101].sort_values("gw")
    assert list(p["career_matches"].iloc[:4]) == [0, 1, 2, 3]
    assert p["is_new_to_data"].iloc[0] == 1
    assert (p["is_new_to_data"].iloc[1:] == 0).all()


def test_matches_since_last_start():
    f = featured(make_frame())
    never = f[f["player_code"] == 102].sort_values("gw")
    assert never["matches_since_last_start"].isna().all()

    alternating = f[f["player_code"] == 103].sort_values("gw")
    # gw2 follows a gw1 start, so one match ago
    assert alternating["matches_since_last_start"].iloc[1] == 1
    # gw3 follows a gw2 blank, so the last start is two matches back
    assert alternating["matches_since_last_start"].iloc[2] == 2


def test_days_since_previous_fixture():
    f = featured(make_frame())
    p = f[f["player_code"] == 101].sort_values("gw")
    assert pd.isna(p["days_since_prev_fixture"].iloc[0])
    assert (p["days_since_prev_fixture"].iloc[1:] == 7.0).all()


def test_history_carries_across_the_season_boundary():
    """A gameweek 1 prediction must be able to use last season's evidence."""
    old = make_frame(n_gw=5, season=SEASON)
    new = make_frame(n_gw=2, season=NEXT_SEASON, start_gw=1)
    new["kickoff_time"] = new["kickoff_time"] + pd.Timedelta(days=365)
    f = featured(pd.concat([old, new], ignore_index=True))

    first_of_new_season = f[(f["season"] == NEXT_SEASON) & (f["gw"] == 1) & (f["player_code"] == 101)]
    row = first_of_new_season.iloc[0]
    # Five prior matches exist, all starts, so this is not a blank slate
    assert row["career_matches"] == 5
    assert row["is_new_to_data"] == 0
    assert row["start_rate_5"] == 1.0


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_features_for_earlier_gameweeks_ignore_later_outcomes():
    """Acceptance criterion: rewrite the future, the past must not move.

    Every rolling feature is a shift(1) within player, so corrupting the
    outcomes of later gameweeks cannot touch a feature computed for an
    earlier one. If this ever fails, the walk forward metrics are invalid.
    """
    rng = np.random.default_rng(0)
    base = make_frame(n_gw=8)
    corrupted = base.copy()

    future = corrupted["gw"] >= 5
    corrupted.loc[future, "minutes"] = rng.integers(0, 91, future.sum())
    corrupted.loc[future, "starts"] = rng.integers(0, 2, future.sum()).astype(float)
    corrupted = dataset.add_labels(corrupted.drop(columns=["started", "played_60", "sub_appear", "started_is_exact"]))

    a = featured(base)
    b = featured(corrupted)

    past_a = a[a["gw"] < 5].sort_values(["player_code", "gw"]).reset_index(drop=True)
    past_b = b[b["gw"] < 5].sort_values(["player_code", "gw"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        past_a[features.FEATURES], past_b[features.FEATURES], check_exact=True
    )

    # And prove the corruption was real, so the test cannot pass vacuously
    future_a = a[a["gw"] >= 6][features.FEATURES]
    future_b = b[b["gw"] >= 6][features.FEATURES]
    assert not future_a.equals(future_b)


def test_double_gameweek_fixtures_share_one_information_set():
    """Both fixtures of a double gameweek must see the same past.

    The second fixture kicks off days after the first, but both are chosen
    at the same deadline, so letting the second one see the first one's
    result would be leakage against the only moment that matters.
    """
    df = make_frame(n_gw=4, players=(101,))
    # Give player 101 a second fixture in gameweek 4
    extra = df[df["gw"] == 4].copy()
    extra["fixture_id"] = 999
    extra["kickoff_time"] = extra["kickoff_time"] + pd.Timedelta(days=3)
    extra["minutes"] = 0
    extra["starts"] = 0.0
    combined = dataset.add_labels(
        pd.concat([df, extra], ignore_index=True).drop(
            columns=["started", "played_60", "sub_appear", "started_is_exact"]
        )
    )

    f = featured(combined)
    dgw = f[f["gw"] == 4].sort_values("kickoff_time")
    assert len(dgw) == 2

    outcome_features = [
        "start_rate_1",
        "start_rate_3",
        "minutes_mean_1",
        "career_matches",
        "matches_since_last_start",
    ]
    first, second = dgw.iloc[0], dgw.iloc[1]
    for col in outcome_features:
        assert first[col] == second[col] or (pd.isna(first[col]) and pd.isna(second[col])), col
    # Three earlier matches only, the sibling fixture must not be counted
    assert first["career_matches"] == 3


def test_upcoming_rows_carry_no_outcome_information():
    """Unplayed rows must have null outcomes but real features."""
    played = make_frame(n_gw=5)
    upcoming = make_frame(n_gw=1, start_gw=6)
    for col in ("minutes", "starts"):
        upcoming[col] = float("nan")
    for col in ("started", "played_60", "sub_appear"):
        upcoming[col] = pd.NA

    f = featured(pd.concat([played, upcoming], ignore_index=True))
    row = f[(f["gw"] == 6) & (f["player_code"] == 101)].iloc[0]
    assert pd.isna(row["minutes"])
    assert row["start_rate_5"] == 1.0
    assert row["career_matches"] == 5


# --------------------------------------------------------------------------
# Expected minutes and baselines
# --------------------------------------------------------------------------


def test_conditional_minutes_are_estimated_not_assumed():
    df = pd.DataFrame(
        {
            "element_type": [1, 1, 1, 3, 3, 3],
            "started": [True, True, False, True, True, False],
            "minutes": [90, 88, 0, 70, 60, 20],
        }
    )
    cond = train.conditional_minutes(df)
    assert cond["start"]["1"] == pytest.approx(89.0)
    assert cond["start"]["3"] == pytest.approx(65.0)
    # A sub appearance is far short of 90, which is the entire point of
    # estimating this rather than hardcoding it
    assert cond["sub"]["3"] == pytest.approx(20.0)


def test_expected_minutes_combines_both_routes_onto_the_pitch():
    cond = {"start": {"3": 80.0}, "sub": {"3": 20.0}, "start_overall": 80.0, "sub_overall": 20.0}
    et = pd.Series([3, 3, 3])
    got = train.expected_minutes(
        np.array([1.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.25]), et, cond
    )
    assert got[0] == pytest.approx(80.0)
    assert got[1] == pytest.approx(20.0)
    assert got[2] == pytest.approx(0.5 * 80 + 0.25 * 20)


def test_expected_minutes_falls_back_for_unseen_position():
    cond = {"start": {"3": 80.0}, "sub": {"3": 20.0}, "start_overall": 75.0, "sub_overall": 15.0}
    got = train.expected_minutes(np.array([1.0]), np.array([0.0]), pd.Series([4]), cond)
    assert got[0] == pytest.approx(75.0)


def test_baseline_uses_started_last_match():
    # The alternating player is deliberately excluded: with him the two
    # buckets are symmetric and both land on exactly 0.5, which would make
    # this assertion vacuous rather than wrong.
    f = featured(make_frame(n_gw=10, players=(101, 102)))
    model = train.baseline_start_rates(f)
    p = train.apply_baseline(model, f)
    assert len(p) == len(f)
    assert ((p >= 0) & (p <= 1)).all()
    # A player who started last week must not be given the same probability
    # as one who did not
    started_last = f["start_rate_1"] == 1.0
    benched_last = f["start_rate_1"] == 0.0
    assert p[started_last].mean() > p[benched_last].mean()


# --------------------------------------------------------------------------
# Splits and scoring
# --------------------------------------------------------------------------


def test_chronological_split_is_ordered_never_random():
    f = featured(make_frame(n_gw=10))
    early, late = train.chronological_split(f, frac=0.2)
    assert len(early) + len(late) == len(f)
    assert early["kickoff_time"].max() <= late["kickoff_time"].min()


def test_score_is_finite_for_confident_mistakes():
    y = np.array([1, 0])
    s = train.score(y, np.array([0.0, 1.0]))
    assert np.isfinite(s["log_loss"])
    assert s["n"] == 2


def test_calibration_table_bins_and_counts():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    table = train.calibration_table(y, p, bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 2000
    # Random draws with probability p are calibrated by construction
    dense = table[table["n"] > 50]
    assert (dense["mean_predicted"] - dense["realised"]).abs().max() < 0.1


def make_varied_frame(n_players=60, n_gw=26, seed=7):
    """A frame big enough that LightGBM actually splits, so determinism means something.

    Each player gets a fixed underlying start propensity and their matches
    are drawn from it, which gives the booster real signal to find.
    """
    rng = np.random.default_rng(seed)
    propensity = rng.uniform(0.05, 0.95, n_players)
    rows = []
    for gw in range(1, n_gw + 1):
        for i in range(n_players):
            starts = int(rng.uniform() < propensity[i])
            minutes = int(rng.integers(60, 91)) if starts else int(rng.choice([0, 0, 0, 15, 25]))
            rows.append(
                {
                    "season": SEASON,
                    "gw": gw,
                    "player_id": i,
                    "player_code": 1000 + i,
                    "fixture_id": gw * 100 + (i % 10),
                    "element_type": 1 + (i % 4),
                    "team_id": 1 + (i % 10),
                    "opponent_team": 20 - (i % 10),
                    "was_home": gw % 2 == 0,
                    "kickoff_time": pd.Timestamp("2024-08-16", tz="UTC") + pd.Timedelta(days=7 * gw),
                    "minutes": minutes,
                    "starts": float(starts),
                    "price": float(4 + (i % 8)),
                    "team_fixtures_in_gw": 1,
                }
            )
    return dataset.add_labels(pd.DataFrame(rows))


def test_training_is_deterministic():
    """Same inputs, same model. Seeds are pinned, so this must hold exactly."""
    f = featured(make_varied_frame())
    first = train.predict_head(*train.fit_head(f, "p_start"), f)
    second = train.predict_head(*train.fit_head(f, "p_start"), f)
    np.testing.assert_array_equal(first, second)
    # And the model must actually be doing something, not emitting a constant
    assert len(np.unique(first.round(6))) > 10


def test_model_separates_regular_starters_from_fringe_players():
    """A weak but real sanity check that the pipeline learns the obvious thing."""
    f = featured(make_varied_frame())
    p = train.predict_head(*train.fit_head(f, "p_start"), f)
    f = f.assign(_p=p)
    per_player = f.groupby("player_code").agg(pred=("_p", "mean"), actual=("started", "mean"))
    assert per_player["pred"].corr(per_player["actual"]) > 0.8


# --------------------------------------------------------------------------
# Real data checks, skipped when the pipeline has not been run
# --------------------------------------------------------------------------

CURATED = Path("data/curated")
PREDICTIONS = Path("data/predictions/2026-27/gw1/minutes.parquet")


@pytest.mark.skipif(not (CURATED / "2025-26").exists(), reason="curated data not built")
def test_real_dataset_labels_are_mostly_exact():
    df = dataset.build(seasons=["2025-26"])
    assert df["started_is_exact"].all()
    assert df["started"].mean() > 0.2


@pytest.mark.skipif(not PREDICTIONS.exists(), reason="predictions not generated")
def test_predictions_are_coherent_probabilities():
    df = pd.read_parquet(PREDICTIONS)
    for col in ("p_start", "p_60", "p_sub", "p_appear"):
        assert df[col].between(0, 1).all(), col
    # Identities that must hold by construction
    assert np.allclose(df["p_60"], df["p_start"] * df["p_60_given_start"])
    assert np.allclose(df["p_appear"], df["p_start"] + df["p_sub"])
    # Nobody can play 60 minutes more often than they get on the pitch
    assert (df["p_60"] <= df["p_appear"] + 1e-9).all()
    assert df["exp_minutes"].between(0, 90).all()


@pytest.mark.skipif(not PREDICTIONS.exists(), reason="predictions not generated")
def test_a_nailed_premium_keeper_is_predicted_to_start():
    """Acceptance spot check: the model must be confident about obvious starters."""
    df = pd.read_parquet(PREDICTIONS)
    players = pd.read_parquet(CURATED / "2026-27" / "players.parquet")
    keepers = df[df["element_type"] == 1].merge(
        players[["player_id", "price"]], on="player_id", how="left"
    )
    premium = keepers.sort_values("price", ascending=False).head(3)
    assert premium["p_start"].max() > 0.9
