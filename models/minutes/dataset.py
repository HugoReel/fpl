"""Training frame for the minutes model: one row per player per fixture.

Reads only curated tables, so historical and live data enter through the
same door. Rows are ordered by kickoff time across seasons, because the
model's whole job is to use a player's past to predict their next match and
a player's past does not reset in August.

Three labels, which together decompose into everything the scoring module
needs:

    started      was in the starting XI
    played_60    reached the 60 minute appearance threshold
    sub_appear   came off the bench and got on the pitch

The heads are trained on nested populations: p_start on every row,
p_60_given_start on starters only, p_sub on non starters only. Multiplying
back out gives p_60 = p_start * p_60_given_start and
p_appear = p_start + p_sub.

The started label prefers the exact starts flag from the archive and falls
back to a minutes >= 45 proxy where it is missing (2021-22 only, and live
rows, since a start scores no points and so never appears in the live
explain blocks). The proxy is measurably worse: in 2022-23 about 15 percent
of true non starters played 45 minutes or more and would be mislabelled, so
the fallback is a last resort rather than the default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ingest.curate import CURATED_ROOT

log = logging.getLogger(__name__)

START_PROXY_MINUTES = 45
SIXTY_MINUTES = 60

# Ordered oldest first. Order matters, every split in this project is
# chronological.
ALL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def available_seasons(curated_root: Path = CURATED_ROOT) -> list[str]:
    """Curated seasons that have a player_fixture table, oldest first."""
    found = [p.parent.name for p in curated_root.glob("*/player_fixture.parquet")]
    return sorted(found)


def load_season(season: str, curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Join one season's curated tables into player-fixture rows."""
    root = curated_root / season
    pf = pd.read_parquet(root / "player_fixture.parquet")
    players = pd.read_parquet(root / "players.parquet")
    fixtures = pd.read_parquet(root / "fixtures.parquet")

    pf = pf.merge(
        players[["player_id", "element_type"]],
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    fixture_cols = ["fixture_id", "kickoff_time", "team_h", "team_a", "h_fixtures_in_gw", "a_fixtures_in_gw"]
    pf = pf.merge(fixtures[fixture_cols], on="fixture_id", how="left", validate="many_to_one")

    # The team on players.parquet is the end of season snapshot and is wrong
    # for anyone transferred mid season. was_home is per fixture, so deriving
    # the team from the fixture pair is correct for every row.
    pf["team_id"] = pf["team_h"].where(pf["was_home"], pf["team_a"])
    pf["team_fixtures_in_gw"] = pf["h_fixtures_in_gw"].where(
        pf["was_home"], pf["a_fixtures_in_gw"]
    )

    price = _season_prices(root)
    if price is not None:
        pf = pf.merge(price, on=["player_id", "gw"], how="left", validate="many_to_one")
    else:
        pf["price"] = float("nan")

    return pf.drop(columns=["team_h", "team_a", "h_fixtures_in_gw", "a_fixtures_in_gw"])


def _season_prices(root: Path) -> pd.DataFrame | None:
    """Per gameweek price, which is the price that was live at that deadline."""
    path = root / "player_gw.parquet"
    if not path.exists():
        return None
    pgw = pd.read_parquet(path)
    if "price" not in pgw.columns:
        return None
    return pgw[["player_id", "gw", "price"]].drop_duplicates(subset=["player_id", "gw"])


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach started, played_60 and sub_appear.

    started_is_exact records which rows used the archive flag rather than
    the minutes proxy, so the report can say how much of the training signal
    is exact.
    """
    df = df.copy()
    exact = df["starts"].notna()
    proxy = df["minutes"] >= START_PROXY_MINUTES
    df["started"] = df["starts"].eq(1).where(exact, proxy)
    df["started_is_exact"] = exact
    df["played_60"] = df["minutes"] >= SIXTY_MINUTES
    df["sub_appear"] = (~df["started"]) & (df["minutes"] > 0)
    return df


def build(
    seasons: list[str] | None = None, curated_root: Path = CURATED_ROOT
) -> pd.DataFrame:
    """Full labelled player-fixture frame across seasons, oldest first.

    Sorted by kickoff time so every downstream shift and rolling window is
    a genuine look backwards.
    """
    if seasons is None:
        seasons = available_seasons(curated_root)
    frames = [load_season(s, curated_root) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df = add_labels(df)
    df = df.sort_values(["kickoff_time", "player_code"], kind="stable").reset_index(drop=True)
    log.info(
        "dataset: %d rows, %d players, seasons %s, %.1f%% of start labels exact",
        len(df),
        df["player_code"].nunique(),
        seasons,
        100 * df["started_is_exact"].mean(),
    )
    return df


def build_upcoming(
    season: str,
    gw: int,
    curated_root: Path = CURATED_ROOT,
) -> pd.DataFrame:
    """Unplayed rows for a target gameweek, shaped like the training frame.

    One row per player per fixture their team plays in that gameweek. The
    outcome columns exist but are null, because they have not happened yet.
    Feeding these through the same feature builder as the training rows is
    what guarantees train and predict see identical feature definitions.
    """
    root = curated_root / season
    players = pd.read_parquet(root / "players.parquet")
    fixtures = pd.read_parquet(root / "fixtures.parquet")

    gw_fixtures = fixtures[fixtures["gw"] == gw]
    if gw_fixtures.empty:
        raise ValueError(f"no fixtures for {season} gw{gw}")

    home = gw_fixtures.rename(columns={"team_h": "team_id", "h_fixtures_in_gw": "team_fixtures_in_gw"})
    home = home[["fixture_id", "gw", "team_id", "team_a", "kickoff_time", "team_fixtures_in_gw"]]
    home = home.rename(columns={"team_a": "opponent_team"})
    home["was_home"] = True

    away = gw_fixtures.rename(columns={"team_a": "team_id", "a_fixtures_in_gw": "team_fixtures_in_gw"})
    away = away[["fixture_id", "gw", "team_id", "team_h", "kickoff_time", "team_fixtures_in_gw"]]
    away = away.rename(columns={"team_h": "opponent_team"})
    away["was_home"] = False

    sides = pd.concat([home, away], ignore_index=True)

    squad = players[["player_id", "player_code", "element_type", "team_id", "price"]]
    df = squad.merge(sides, on="team_id", how="inner")
    df["season"] = season

    for col in ("minutes", "starts"):
        df[col] = float("nan")
    for col in ("started", "played_60", "sub_appear"):
        df[col] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["started_is_exact"] = False

    return df.sort_values(["kickoff_time", "player_code"], kind="stable").reset_index(drop=True)
