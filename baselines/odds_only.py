"""Baseline: the betting market, with no machine learning anywhere.

The external bar. Team goal expectations come from de-vigged closing odds,
clean sheets fall out of a Poisson zero, minutes use the frozen naive rule,
and attacking returns are allocated to players by fixed shares. Nothing is
trained. If the trained pipeline cannot beat this, that is the headline
result of the phase.

Where this baseline should be strong is defence. Odds price a match, and a
clean sheet is a direct read of the opponent's goal expectation, so keepers
and defenders get an estimate that owes nothing to a trailing average.
Where it should be weak is attack, because there are no historical scorer
odds on football-data.co.uk. A player's share of his team's goals is
allocated by position and price rank rather than by anything about the
player, so a cheap poacher and an expensive one who never shoots look
identical inside their price bracket.

Every constant below is measured on 2021-22 through 2024-25, never on the
frozen 2025-26 evaluation season:

  POSITION_GOAL_SHARE
      Share of a team's goals scored by each position. GKP is zero, DEF
      0.126, MID 0.567, FWD 0.307.

  POSITION_ASSIST_SHARE
      Same for assists: GKP 0.005, DEF 0.217, MID 0.620, FWD 0.158.

  ASSISTS_PER_GOAL = 0.897
      League wide assists divided by goals. Not every goal is assisted.

  PRICE_RANK_DECAY = 0.62
      Within a team and position, goals concentrate on the expensive
      players. Measured share by price rank runs 0.348, 0.187, 0.123,
      0.086 for midfielders and forwards, and successive ratios are 0.54,
      0.66, 0.70. A geometric decay of 0.62 sits in the middle of that.

  SAVES_PER_OPPONENT_XG = 1.993
      Keeper saves per unit of opponent goal expectation, measured against
      the odds themselves on 2024-25.

  BONUS_PER_APPEARANCE, CARDS_PER_APPEARANCE, DEFCON_RATE
      Position level base rates. Odds say nothing about bonus, cards or
      defensive contributions, but omitting them entirely would handicap
      this baseline against candidates that model them, and the comparison
      is supposed to be fair. A position constant is not a model.

DEFCON_RATE is measured on 2025-26, which IS the evaluation season, because
it is the only season in which defensive contributions were recorded. It is
a single position level constant rather than anything player specific, so
the leak is a rounding error on a base rate, but it is a leak and it is
noted here rather than buried.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ingest.curate import CURATED_ROOT
from scoring.rules_2026_27 import ELEMENT_TYPE_TO_POSITION, expected_points as score_ep

log = logging.getLogger(__name__)

NAME = "odds_only"

POSITION_GOAL_SHARE = {1: 0.000, 2: 0.126, 3: 0.567, 4: 0.307}
POSITION_ASSIST_SHARE = {1: 0.005, 2: 0.217, 3: 0.620, 4: 0.158}
ASSISTS_PER_GOAL = 0.897
PRICE_RANK_DECAY = 0.62
SAVES_PER_OPPONENT_XG = 1.993
BONUS_PER_APPEARANCE = {1: 0.272, 2: 0.186, 3: 0.191, 4: 0.364}
CARDS_PER_APPEARANCE = {1: 0.069, 2: 0.166, 3: 0.142, 4: 0.114}
DEFCON_RATE = {1: 0.000, 2: 0.270, 3: 0.179, 4: 0.012}


def load_odds(season: str, curated_root=CURATED_ROOT) -> pd.DataFrame:
    path = curated_root / season / "fixture_odds.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m ingest.odds --seasons {season}"
        )
    return pd.read_parquet(path)


def attach_market(df: pd.DataFrame, season: str, curated_root=CURATED_ROOT) -> pd.DataFrame:
    """Attach the player's own team and opponent goal expectations."""
    odds = load_odds(season, curated_root)
    df = df.merge(
        odds[["fixture_id", "home_xg", "away_xg", "p_home_cs", "p_away_cs"]],
        on="fixture_id",
        how="left",
    )
    home = df["was_home"].astype(bool)
    df["team_xg"] = np.where(home, df["home_xg"], df["away_xg"])
    df["opponent_xg"] = np.where(home, df["away_xg"], df["home_xg"])
    df["p_clean_sheet"] = np.where(home, df["p_home_cs"], df["p_away_cs"])
    return df


def allocation_weight(df: pd.DataFrame) -> pd.Series:
    """Each player's share of his position's goals within his team, this gameweek.

    Weights are geometric in price rank and normalised across the players
    expected to appear, so a share is never spent on someone the naive
    minutes rule says is not playing.
    """
    playing = df["p_appear"] > 0
    rank = (
        df[playing]
        .groupby(["season", "gw", "team_id", "element_type"])["price"]
        .rank(ascending=False, method="first")
    )
    weight = pd.Series(0.0, index=df.index)
    weight.loc[rank.index] = PRICE_RANK_DECAY ** (rank - 1)

    total = weight.groupby(
        [df["season"], df["gw"], df["team_id"], df["element_type"]]
    ).transform("sum")
    return (weight / total.replace(0, np.nan)).fillna(0.0)


def expected_points(full: pd.DataFrame, season: str, curated_root=CURATED_ROOT) -> pd.DataFrame:
    """Expected points from odds alone, in the compose schema."""
    df = full[full["season"] == season].copy()
    if df.empty:
        return pd.DataFrame(columns=["season", "gw", "player_id", "player_code", "ep_total"])

    df = attach_market(df, season, curated_root)
    if df["team_xg"].isna().all():
        log.warning("no odds joined for %s, returning an empty frame", season)
        return pd.DataFrame(columns=["season", "gw", "player_id", "player_code", "ep_total"])

    # The frozen naive minutes rule: started last week means a full match.
    started_last = df["start_rate_1"].fillna(0.0) > 0
    df["p_appear"] = started_last.astype(float)
    df["weight"] = allocation_weight(df)

    et = df["element_type"].astype(int)
    goal_share = et.map(POSITION_GOAL_SHARE).astype(float)
    assist_share = et.map(POSITION_ASSIST_SHARE).astype(float)

    df["exp_goals"] = df["team_xg"].fillna(0.0) * goal_share * df["weight"]
    df["exp_assists"] = (
        df["team_xg"].fillna(0.0) * ASSISTS_PER_GOAL * assist_share * df["weight"]
    )
    df["exp_saves"] = np.where(
        et == 1, df["opponent_xg"].fillna(0.0) * SAVES_PER_OPPONENT_XG * df["p_appear"], 0.0
    )
    df["exp_goals_conceded"] = df["opponent_xg"].fillna(0.0) * df["p_appear"]
    df["p_defcon"] = et.map(DEFCON_RATE).astype(float) * df["p_appear"]
    df["exp_bonus"] = et.map(BONUS_PER_APPEARANCE).astype(float) * df["p_appear"]
    df["exp_cards"] = et.map(CARDS_PER_APPEARANCE).astype(float) * df["p_appear"]
    df["p_clean_sheet"] = df["p_clean_sheet"].fillna(0.0)

    ep = np.empty(len(df))
    for i, row in enumerate(df.to_dict("records")):
        ep[i] = score_ep(
            ELEMENT_TYPE_TO_POSITION[int(row["element_type"])],
            p_appear=float(row["p_appear"]),
            p_60plus=float(row["p_appear"]),
            exp_goals=float(row["exp_goals"]),
            exp_assists=float(row["exp_assists"]),
            p_clean_sheet=float(row["p_clean_sheet"]),
            exp_goals_conceded=float(row["exp_goals_conceded"]),
            exp_saves=float(row["exp_saves"]),
            p_defcon=float(row["p_defcon"]),
            exp_bonus=float(row["exp_bonus"]),
            exp_cards=float(row["exp_cards"]),
        )
    df["ep_total"] = ep

    return df.groupby(["season", "gw", "player_id", "player_code"], as_index=False).agg(
        ep_total=("ep_total", "sum"),
        n_fixtures=("fixture_id", "count"),
        p_clean_sheet=("p_clean_sheet", "first"),
        exp_goals=("exp_goals", "sum"),
    )
