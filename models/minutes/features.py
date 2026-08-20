"""Lagged features for the minutes model.

Every feature here answers a question that could have been answered before
the deadline. That is enforced structurally rather than by inspection: each
rolling statistic is built from a shift(1) within player, so the row for a
fixture can only ever see rows that kicked off earlier. Rows are ordered by
kickoff time across seasons, so a player's history carries over the summer
instead of resetting.

Two kinds of input are deliberately NOT features, despite being obvious
candidates:

  status and chance_of_playing_next_round
    The archive only preserves the end of season snapshot of these, so a
    2023-24 row would carry a value set in May 2024. Training on that leaks
    the future. They become usable once live snapshots have accumulated
    week by week, which is exactly what ingest/snapshot.py is collecting.

  current season set piece order
    Same problem, same archive limitation. The previous season's order is
    leak free though, because that season had finished before this one
    started, so it is included as prev_pens_order and friends. That keeps
    the highest signal part of set piece data without the leak.

Fixture schedule lookahead (fixtures_next_7_days) IS allowed. The schedule
is published in advance, so counting a team's upcoming fixtures uses no
information about outcomes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.curate import CURATED_ROOT

log = logging.getLogger(__name__)

ROLLING_WINDOWS = (1, 3, 5)

# Columns fed to LightGBM, in a fixed order. The order is persisted with the
# model so predict time cannot silently reorder them.
FEATURES = [
    "element_type",
    "gw",
    "was_home",
    "price",
    "price_rank_in_team_pos",
    "team_fixtures_in_gw",
    "fixtures_next_7_days",
    "days_since_prev_fixture",
    "career_matches",
    "is_new_to_data",
    "matches_since_last_start",
    "start_rate_1",
    "start_rate_3",
    "start_rate_5",
    "minutes_mean_1",
    "minutes_mean_3",
    "minutes_mean_5",
    "played_60_rate_5",
    "sub_rate_5",
    "prev_season_matches",
    "prev_season_minutes",
    "prev_season_start_rate",
    "prev_pens_order",
    "prev_corners_order",
    "prev_fk_order",
]


def _as_float(s: pd.Series) -> pd.Series:
    """Coerce a possibly object/bool/NA column to float, NA becoming nan."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


def as_of_gameweek(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    """Freeze a backward looking series at the gameweek boundary.

    A shift within player is ordered by kickoff, so in a double gameweek the
    second fixture would otherwise see the first fixture's result. That
    result did not exist at the deadline, which is the moment every decision
    is actually made, so taking the gameweek's first value for every fixture
    in it pins the whole gameweek to the information set the manager had.

    Only outcome derived features need this. Schedule facts such as the
    kickoff gap or fixture congestion are published in advance and are
    legitimately known for the second fixture.
    """
    return series.groupby(
        [df["player_code"], df["season"], df["gw"]], sort=False
    ).transform("first")


def _prev_rolling(
    df: pd.DataFrame, values: pd.Series, window: int, stat: str = "mean"
) -> pd.Series:
    """Rolling stat over a player's previous `window` matches.

    The shift happens before the roll, so the current row is excluded no
    matter what. This is the single place the no lookahead property is
    enforced for rolling features.
    """
    shifted = values.groupby(df["player_code"], sort=False).shift(1)
    rolled = getattr(
        shifted.groupby(df["player_code"], sort=False).rolling(window, min_periods=1), stat
    )()
    return rolled.reset_index(level=0, drop=True).reindex(df.index)


def _matches_since_last_start(df: pd.DataFrame, started: pd.Series) -> pd.Series:
    """How many matches ago the player last started, from previous rows only."""
    pos = df.groupby("player_code", sort=False).cumcount()
    prev_started = started.groupby(df["player_code"], sort=False).shift(1)
    # Position of the immediately preceding row, kept only where that row
    # was a start, then carried forward to find the most recent start.
    marker = (pos - 1).where(prev_started.fillna(0) > 0)
    last_start = marker.groupby(df["player_code"], sort=False).ffill()
    return pos - last_start


def load_schedule(seasons: list[str], curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Team fixture schedule from the fixture tables, both sides of each match."""
    frames = []
    for season in seasons:
        path = curated_root / season / "fixtures.parquet"
        if not path.exists():
            continue
        fx = pd.read_parquet(path)
        for side in ("team_h", "team_a"):
            frames.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "team_id": fx[side],
                        "fixture_id": fx["fixture_id"],
                        "kickoff_time": fx["kickoff_time"],
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=["season", "team_id", "fixture_id", "kickoff_time"])
    return pd.concat(frames, ignore_index=True)


def _fixtures_next_7_days(df: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.Series:
    """Count of the team's other fixtures kicking off within the next week.

    Schedule only, so this is known at the deadline and carries no outcome
    information. Congestion is the mechanism managers rotate around.

    The schedule comes from the fixture tables rather than from the player
    rows in hand. Deriving it from the player frame would break whenever
    that frame holds only one gameweek, which is exactly the case when
    predicting a live gameweek, and would silently report an empty week
    ahead.
    """
    own = (
        df[["season", "team_id", "fixture_id", "kickoff_time"]]
        .drop_duplicates()
        .dropna(subset=["kickoff_time"])
    )
    if schedule is None:
        schedule = load_schedule(sorted(df["season"].unique()))
    sched = (
        pd.concat([schedule, own], ignore_index=True)
        .drop_duplicates(subset=["season", "team_id", "fixture_id"])
        .dropna(subset=["kickoff_time"])
    )
    by_team: dict[tuple, np.ndarray] = {}
    for key, grp in sched.groupby(["season", "team_id"], sort=False):
        by_team[key] = np.sort(grp["kickoff_time"].to_numpy())

    window = np.timedelta64(7, "D")
    out = np.full(len(df), np.nan)
    kickoffs = df["kickoff_time"].to_numpy()
    seasons = df["season"].to_numpy()
    teams = df["team_id"].to_numpy()
    for i in range(len(df)):
        arr = by_team.get((seasons[i], teams[i]))
        if arr is None or pd.isna(kickoffs[i]):
            continue
        t = kickoffs[i]
        lo = np.searchsorted(arr, t, side="right")
        hi = np.searchsorted(arr, t + window, side="right")
        out[i] = hi - lo
    return pd.Series(out, index=df.index)


def build_prior_season_table(
    seasons: list[str], curated_root: Path = CURATED_ROOT
) -> pd.DataFrame:
    """Per player, what the previous season says about them.

    Keyed by the season the values apply TO, so joining on (season,
    player_code) attaches last season's summary with no leakage: that
    season had already finished when the current one kicked off.
    """
    rows = []
    for prev, current in zip(seasons, seasons[1:]):
        root = curated_root / prev
        pf = pd.read_parquet(root / "player_fixture.parquet")
        agg = pf.groupby("player_code", as_index=False).agg(
            prev_season_matches=("minutes", "size"),
            prev_season_minutes=("minutes", "sum"),
        )
        starts = pf.copy()
        starts["_started"] = starts["starts"].eq(1).where(
            starts["starts"].notna(), starts["minutes"] >= 45
        )
        rate = starts.groupby("player_code", as_index=False)["_started"].mean()
        rate.columns = ["player_code", "prev_season_start_rate"]
        agg = agg.merge(rate, on="player_code", how="left")

        players = pd.read_parquet(root / "players.parquet")
        setpiece = players[
            ["player_code", "penalties_order", "corners_order", "direct_freekicks_order"]
        ].rename(
            columns={
                "penalties_order": "prev_pens_order",
                "corners_order": "prev_corners_order",
                "direct_freekicks_order": "prev_fk_order",
            }
        )
        agg = agg.merge(setpiece, on="player_code", how="left")
        agg["season"] = current
        rows.append(agg)

    if not rows:
        return pd.DataFrame(
            columns=[
                "player_code",
                "season",
                "prev_season_matches",
                "prev_season_minutes",
                "prev_season_start_rate",
                "prev_pens_order",
                "prev_corners_order",
                "prev_fk_order",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def add_features(
    df: pd.DataFrame,
    prior_season: pd.DataFrame | None = None,
    curated_root: Path = CURATED_ROOT,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach every model feature to a kickoff ordered player-fixture frame.

    The frame may mix played rows and unplayed upcoming rows. Upcoming rows
    have null outcomes, so their rolling features come entirely from earlier
    played rows, which is precisely what prediction needs.
    """
    df = df.sort_values(["kickoff_time", "player_code"], kind="stable").reset_index(drop=True)

    started = _as_float(df["started"])
    minutes = _as_float(df["minutes"])
    played_60 = _as_float(df["played_60"])
    sub_appear = _as_float(df["sub_appear"])

    for w in ROLLING_WINDOWS:
        df[f"start_rate_{w}"] = as_of_gameweek(df, _prev_rolling(df, started, w))
        df[f"minutes_mean_{w}"] = as_of_gameweek(df, _prev_rolling(df, minutes, w))
    df["played_60_rate_5"] = as_of_gameweek(df, _prev_rolling(df, played_60, 5))
    df["sub_rate_5"] = as_of_gameweek(df, _prev_rolling(df, sub_appear, 5))

    df["matches_since_last_start"] = as_of_gameweek(
        df, _matches_since_last_start(df, started)
    )
    df["career_matches"] = as_of_gameweek(
        df, df.groupby("player_code", sort=False).cumcount()
    )
    df["is_new_to_data"] = (df["career_matches"] == 0).astype("int64")

    kickoff = df["kickoff_time"]
    prev_kickoff = kickoff.groupby(df["player_code"], sort=False).shift(1)
    df["days_since_prev_fixture"] = (kickoff - prev_kickoff).dt.total_seconds() / 86400.0

    if schedule is None:
        schedule = load_schedule(sorted(df["season"].unique()), curated_root)
    df["fixtures_next_7_days"] = _fixtures_next_7_days(df, schedule)

    df["price_rank_in_team_pos"] = (
        df.groupby(["season", "gw", "team_id", "element_type"], sort=False)["price"]
        .rank(ascending=False, method="min")
    )

    if prior_season is None:
        seasons = sorted(df["season"].unique())
        prior_season = build_prior_season_table(seasons, curated_root)
    if not prior_season.empty:
        df = df.merge(prior_season, on=["season", "player_code"], how="left")
    else:
        for col in (
            "prev_season_matches",
            "prev_season_minutes",
            "prev_season_start_rate",
            "prev_pens_order",
            "prev_corners_order",
            "prev_fk_order",
        ):
            df[col] = float("nan")

    df["was_home"] = _as_float(df["was_home"])
    for col in FEATURES:
        if col not in df.columns:
            raise KeyError(f"feature {col} was not built")
        df[col] = _as_float(df[col])
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The model input, columns in the fixed FEATURES order."""
    return df[FEATURES]
