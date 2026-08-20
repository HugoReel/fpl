"""v0 rate estimators: everything expected_points needs except minutes.

These are deliberately crude. They exist so the expected points pipeline has
a complete, honest interface to build against, and they are meant to be
replaced by a Poisson attack model and Dixon-Coles team model in a later
phase. Nothing here is tuned. Treating any constant below as a finding
rather than a placeholder would be a mistake.

Every constant, and why it is that value:

  TRAILING_MATCHES = 10
      Window for every trailing statistic. Long enough to survive one blank
      week, short enough to track a player losing their place.

  MIN_MINUTES_FOR_RATE = 30
      A per-90 rate from a 5 minute cameo is noise multiplied by 18. Only
      matches of at least 30 minutes contribute to a rate.

  SHRINK_PRIOR_MINUTES = 600
      Empirical Bayes prior weight, expressed in minutes, so a player with
      600 trailing minutes sits halfway between their own rate and their
      position's mean. Roughly seven full matches, which is about where a
      striker's scoring rate starts to mean something.

  OPPONENT_ADJ_CLIP = (0.6, 1.6)
      Opponent adjustments are ratios to the league average. Clipped because
      a team that has conceded once in ten games would otherwise drive an
      adjustment to zero on a ten match sample.

  CLEAN_SHEET_CLIP = (0.02, 0.75)
      No defence is safe and none is hopeless. Bounds a rate that is
      estimated from at most ten matches.

  VENUE_BLEND = 0.5
      Clean sheet rate is half overall form and half the same team's record
      at this venue. Home and away records differ enough to matter and are
      individually too sparse to trust alone.

  BONUS_PRIOR_MATCHES = 10
      Bonus is shrunk toward zero, not toward a position mean, because most
      players genuinely earn none and the position mean is dragged upward by
      a handful of players.

  FALLBACK_CLEAN_SHEET = 0.25
      Only reached when a club has no trailing record whatsoever, which in
      practice means a promoted side in gameweek 1. Roughly the league wide
      clean sheet rate, so a new team is treated as average rather than as
      hopeless.

  CARD_POINTS_PER_APPEARANCE
      Estimated per position from history rather than assumed. Cards are a
      small, stable, position-level effect and modelling them per player
      would be overfitting a rare event.

All outputs are per player per fixture. Every trailing input is evaluated as
of the gameweek deadline, never as of the individual kickoff, so both halves
of a double gameweek see exactly the same past.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.curate import CURATED_ROOT
from models.minutes.features import as_of_gameweek
from scoring.rules_2026_27 import DEFCON_THRESHOLD, ELEMENT_TYPE_TO_POSITION

log = logging.getLogger(__name__)

TRAILING_MATCHES = 10
MIN_MINUTES_FOR_RATE = 30
SHRINK_PRIOR_MINUTES = 600
SHRINK_PRIOR_90S = SHRINK_PRIOR_MINUTES / 90.0
OPPONENT_ADJ_CLIP = (0.6, 1.6)
CLEAN_SHEET_CLIP = (0.02, 0.75)
VENUE_BLEND = 0.5
BONUS_PRIOR_MATCHES = 10
DEFAULT_LEAGUE_GOALS_PER_TEAM = 1.45
# Used only when a club has no trailing record at all, which in practice
# means a promoted side in gameweek 1. Roughly the league wide rate.
FALLBACK_CLEAN_SHEET = 0.25

# Per-90 statistics that share the empirical Bayes shrinkage machinery, as
# (prior key, trailing total column, output column).
PER90_SPECS = [
    ("goals_scored", "trail_goals", "rate_goals_p90"),
    ("assists", "trail_assists", "rate_assists_p90"),
    ("saves", "trail_saves", "rate_saves_p90"),
]
RATE_COLUMNS = [spec[0] for spec in PER90_SPECS]

# Threshold each position must reach for a defensive contribution point.
DEFCON_BY_ELEMENT_TYPE = {
    et: DEFCON_THRESHOLD[pos] for et, pos in ELEMENT_TYPE_TO_POSITION.items()
}


# --------------------------------------------------------------------------
# Priors estimated from history
# --------------------------------------------------------------------------


def position_priors(history: pd.DataFrame) -> dict:
    """Position level means, estimated only from matches already played.

    History must be the frame of everything before the target, so that
    removing a later gameweek cannot move a prior and therefore cannot move
    any expected points value.
    """
    played = history[history["minutes"] >= MIN_MINUTES_FOR_RATE]
    priors: dict = {"goals_scored": {}, "assists": {}, "saves": {}, "defcon": {}, "cards": {}}
    if played.empty:
        return priors

    for et, grp in played.groupby("element_type"):
        key = str(int(et))
        nineties = max(grp["minutes"].sum() / 90.0, 1e-9)
        for col in RATE_COLUMNS:
            priors[col][key] = float(grp[col].sum() / nineties)
        threshold = DEFCON_BY_ELEMENT_TYPE.get(int(et))
        defcon_rows = grp[grp["rule_regime"] != "pre_defcon"]
        if threshold is None or defcon_rows.empty:
            priors["defcon"][key] = 0.0
        else:
            priors["defcon"][key] = float(
                (defcon_rows["defensive_contribution"] >= threshold).mean()
            )
        # Expected points lost to cards per appearance, a positive number
        priors["cards"][key] = float(
            (grp["yellow_cards"] * 1 + grp["red_cards"] * 3).mean()
        )
    return priors


def league_goals_per_team(history: pd.DataFrame) -> float:
    """Average goals a team scores in a match, over completed history."""
    if history.empty or "goals_conceded" not in history:
        return DEFAULT_LEAGUE_GOALS_PER_TEAM
    starters = history[history["minutes"] >= 60]
    if starters.empty:
        return DEFAULT_LEAGUE_GOALS_PER_TEAM
    # Goals conceded by a player on the pitch for a full match is the
    # opponent's goal count, so its mean is the league scoring rate.
    return float(starters["goals_conceded"].mean())


# --------------------------------------------------------------------------
# Team level trailing form
# --------------------------------------------------------------------------


def team_name_map(seasons: list[str], curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Season scoped team_id to a name that is stable across seasons.

    FPL renumbers teams alphabetically every August, so team_id 1 is not the
    same club in consecutive seasons. Team form has to key on something
    stable for exactly the reason player features key on player_code, or a
    gameweek 1 prediction would treat every side as having no history.
    """
    frames = []
    for season in seasons:
        path = curated_root / season / "players.parquet"
        if not path.exists():
            continue
        players = pd.read_parquet(path)
        frames.append(
            players[["team_id", "team_name"]].drop_duplicates().assign(season=season)
        )
    if not frames:
        return pd.DataFrame(columns=["season", "team_id", "team_name"])
    return pd.concat(frames, ignore_index=True)


def load_team_matches(seasons: list[str], curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """One row per team per fixture, with goals for, against and venue."""
    names = team_name_map(seasons, curated_root)
    frames = []
    for season in seasons:
        path = curated_root / season / "fixtures.parquet"
        if not path.exists():
            continue
        fx = pd.read_parquet(path)
        for side, opp, venue in (("team_h", "team_a", True), ("team_a", "team_h", False)):
            gf = "team_h_score" if venue else "team_a_score"
            ga = "team_a_score" if venue else "team_h_score"
            frames.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "gw": fx["gw"],
                        "fixture_id": fx["fixture_id"],
                        "kickoff_time": fx["kickoff_time"],
                        "team_id": fx[side],
                        "was_home": venue,
                        "goals_for": pd.to_numeric(fx[gf], errors="coerce"),
                        "goals_against": pd.to_numeric(fx[ga], errors="coerce"),
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    tm = pd.concat(frames, ignore_index=True)
    if not names.empty:
        tm = tm.merge(names, on=["season", "team_id"], how="left")
    else:
        tm["team_name"] = tm["team_id"].astype(str)
    tm["team_name"] = tm["team_name"].fillna(tm["team_id"].astype(str))
    tm["clean_sheet"] = (tm["goals_against"] == 0).astype(float)
    tm.loc[tm["goals_against"].isna(), "clean_sheet"] = np.nan
    return tm.sort_values(["kickoff_time", "team_name"], kind="stable").reset_index(drop=True)


TEAM_FORM_COLUMNS = ["team_cs_10", "team_gf_10", "team_ga_10", "team_cs_venue_10"]


def _team_roll(frame: pd.DataFrame, col: str) -> pd.Series:
    """Trailing mean over a club's previous matches, current match excluded."""
    shifted = frame.groupby("team_name", sort=False)[col].shift(1)
    return (
        shifted.groupby(frame["team_name"], sort=False)
        .rolling(TRAILING_MATCHES, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(frame.index)
    )


def team_trailing(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Trailing club form as of each gameweek, keyed per team per fixture.

    Keyed by fixture rather than by gameweek because a club playing twice in
    a double gameweek can play one at home and one away, and the venue
    specific clean sheet record differs between them.
    """
    if team_matches.empty:
        return pd.DataFrame(columns=["season", "gw", "fixture_id", "team_id", *TEAM_FORM_COLUMNS])

    tm = team_matches.sort_values("kickoff_time", kind="stable").copy()

    for src, name in (
        ("clean_sheet", "team_cs_10"),
        ("goals_for", "team_gf_10"),
        ("goals_against", "team_ga_10"),
    ):
        tm[name] = _team_roll(tm, src)
        # Both halves of a double gameweek are chosen at one deadline
        tm[name] = tm.groupby(["team_name", "season", "gw"], sort=False)[name].transform("first")

    tm["team_cs_venue_10"] = np.nan
    for venue in (True, False):
        mask = tm["was_home"] == venue
        if mask.any():
            tm.loc[mask, "team_cs_venue_10"] = _team_roll(tm[mask], "clean_sheet")
    tm["team_cs_venue_10"] = tm.groupby(
        ["team_name", "season", "gw", "was_home"], sort=False
    )["team_cs_venue_10"].transform("first")

    return tm[["season", "gw", "fixture_id", "team_id", *TEAM_FORM_COLUMNS]]


# --------------------------------------------------------------------------
# Player level trailing form
# --------------------------------------------------------------------------


def player_trailing(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing player totals over qualifying matches, as of the deadline.

    A qualifying match is one of at least MIN_MINUTES_FOR_RATE minutes. The
    join is made against the first kickoff of the player's gameweek with
    exact matches excluded, which is what makes both fixtures of a double
    gameweek see the same history.
    """
    df = df.sort_values(["kickoff_time", "player_code"], kind="stable").reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    df["_gw_start"] = df.groupby(["player_code", "season", "gw"], sort=False)[
        "kickoff_time"
    ].transform("min")

    qual = df[df["minutes"] >= MIN_MINUTES_FOR_RATE].copy()
    qual["_is_defcon_season"] = (qual["rule_regime"] != "pre_defcon").astype(float)
    threshold = qual["element_type"].map(DEFCON_BY_ELEMENT_TYPE).astype(float)
    qual["_hit_defcon"] = (
        (pd.to_numeric(qual["defensive_contribution"], errors="coerce") >= threshold)
        .astype(float)
        .where(threshold.notna(), 0.0)
    ) * qual["_is_defcon_season"]

    accum = {
        "trail_minutes": "minutes",
        "trail_goals": "goals_scored",
        "trail_assists": "assists",
        "trail_saves": "saves",
        "trail_bonus": "bonus",
        "trail_defcon_hits": "_hit_defcon",
        "trail_defcon_matches": "_is_defcon_season",
    }
    state = pd.DataFrame({"player_code": qual["player_code"], "kickoff_time": qual["kickoff_time"]})
    grouped = qual.groupby("player_code", sort=False)
    for out_col, src in accum.items():
        state[out_col] = (
            grouped[src]
            .rolling(TRAILING_MATCHES, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .reindex(qual.index)
        )
    state["trail_matches"] = (
        grouped["minutes"]
        .rolling(TRAILING_MATCHES, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
        .reindex(qual.index)
    )

    state = state.sort_values("kickoff_time", kind="stable")
    left = df[["_row", "player_code", "_gw_start"]].sort_values("_gw_start", kind="stable")
    joined = pd.merge_asof(
        left,
        state,
        left_on="_gw_start",
        right_on="kickoff_time",
        by="player_code",
        direction="backward",
        allow_exact_matches=False,
    )
    joined = joined.set_index("_row").sort_index()

    trail_cols = list(accum) + ["trail_matches"]
    for col in trail_cols:
        df[col] = joined[col].to_numpy()
        df[col] = df[col].fillna(0.0)
    return df.drop(columns=["_row", "_gw_start"])


def _shrunk_per90(events: pd.Series, nineties: pd.Series, prior_rate: pd.Series) -> pd.Series:
    """Empirical Bayes per-90 rate: observed events plus prior events, over total 90s."""
    return (events + prior_rate * SHRINK_PRIOR_90S) / (nineties + SHRINK_PRIOR_90S)


def add_rates(
    df: pd.DataFrame,
    history: pd.DataFrame | None = None,
    curated_root: Path = CURATED_ROOT,
) -> pd.DataFrame:
    """Attach every rate expected_points needs, per player per fixture.

    `history` is the frame of already played matches the priors may be
    estimated from. It must exclude the target gameweek and everything after
    it, which is what keeps expected points invariant to future data.
    """
    if history is None:
        history = df[df["minutes"].notna()]
    priors = position_priors(history)
    league_goals = league_goals_per_team(history)

    df = player_trailing(df)

    et = df["element_type"].astype("Int64").astype(str)
    nineties = df["trail_minutes"] / 90.0

    for prior_key, trail_col, out in PER90_SPECS:
        prior = et.map(priors[prior_key]).astype(float).fillna(0.0)
        df[out] = _shrunk_per90(df[trail_col], nineties, prior)

    # Bonus shrinks toward zero rather than a position mean
    df["rate_bonus"] = df["trail_bonus"] / (df["trail_matches"] + BONUS_PRIOR_MATCHES)

    # Defensive contribution, only where the seasons in the window recorded it
    defcon_prior = et.map(priors["defcon"]).astype(float).fillna(0.0)
    df["defcon_available"] = df["trail_defcon_matches"] > 0
    raw_defcon = (df["trail_defcon_hits"] + defcon_prior * BONUS_PRIOR_MATCHES) / (
        df["trail_defcon_matches"] + BONUS_PRIOR_MATCHES
    )
    df["rate_defcon"] = raw_defcon.where(df["defcon_available"], 0.0)
    no_threshold = df["element_type"].map(DEFCON_BY_ELEMENT_TYPE).isna()
    df.loc[no_threshold, "rate_defcon"] = 0.0
    df.loc[no_threshold, "defcon_available"] = False

    df["card_points_per_appearance"] = et.map(priors["cards"]).astype(float).fillna(0.0)

    # Team and opponent form, joined per fixture so a double gameweek with
    # one home and one away leg gets the right venue record on each.
    seasons = sorted(df["season"].unique())
    team_form = team_trailing(load_team_matches(seasons, curated_root))
    if team_form.empty:
        for col in (*TEAM_FORM_COLUMNS, "opp_gf_10", "opp_ga_10"):
            df[col] = np.nan
    else:
        # fixture_id is only unique within a season, so season is part of
        # every join key here.
        own = team_form[["season", "fixture_id", "team_id", *TEAM_FORM_COLUMNS]]
        df = df.merge(
            own, on=["season", "fixture_id", "team_id"], how="left", validate="many_to_one"
        )
        opp = team_form[["season", "fixture_id", "team_id", "team_gf_10", "team_ga_10"]].rename(
            columns={
                "team_id": "opponent_team",
                "team_gf_10": "opp_gf_10",
                "team_ga_10": "opp_ga_10",
            }
        )
        df = df.merge(
            opp,
            on=["season", "fixture_id", "opponent_team"],
            how="left",
            validate="many_to_one",
        )

    lo, hi = OPPONENT_ADJ_CLIP
    # A leaky opponent raises attacking returns, a strong attacking opponent
    # lowers clean sheet chances. Both are ratios to the league average, and
    # a missing value means no form yet, so the neutral adjustment is 1.
    df["opp_defence_adj"] = (df["opp_ga_10"] / league_goals).clip(lo, hi).fillna(1.0)
    df["opp_attack_adj"] = (df["opp_gf_10"] / league_goals).clip(lo, hi).fillna(1.0)
    df["own_defence_adj"] = (df["team_ga_10"] / league_goals).clip(lo, hi).fillna(1.0)

    # Half overall clean sheet form, half the same venue's record. Where one
    # is missing the other carries it, and where both are missing the
    # position wide fallback keeps a new team from being written off.
    overall_cs = df["team_cs_10"]
    venue_cs = df["team_cs_venue_10"]
    base_cs = (
        VENUE_BLEND * overall_cs.fillna(venue_cs) + (1 - VENUE_BLEND) * venue_cs.fillna(overall_cs)
    )
    cs_lo, cs_hi = CLEAN_SHEET_CLIP
    df["p_clean_sheet"] = (base_cs / df["opp_attack_adj"]).clip(cs_lo, cs_hi)
    df["p_clean_sheet"] = df["p_clean_sheet"].fillna(FALLBACK_CLEAN_SHEET)

    df["league_goals_per_team"] = league_goals
    return df


def apply_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Scale rates by expected minutes to get per-fixture expectations.

    Requires exp_minutes, p_appear and p_60 from the minutes model. Rates
    that describe a full match are scaled by expected 90s, while
    probabilities that need a player merely to be involved are scaled by how
    much of a match they are expected to play, capped at one.
    """
    nineties = df["exp_minutes"] / 90.0
    involvement = nineties.clip(upper=1.0)

    df["exp_goals"] = df["rate_goals_p90"] * nineties * df["opp_defence_adj"]
    df["exp_assists"] = df["rate_assists_p90"] * nineties * df["opp_defence_adj"]
    df["exp_saves"] = np.where(
        df["element_type"] == 1, df["rate_saves_p90"] * nineties, 0.0
    )
    df["exp_goals_conceded"] = (
        df["opp_gf_10"].fillna(df["league_goals_per_team"]) * df["own_defence_adj"] * nineties
    )
    df["p_defcon"] = df["rate_defcon"] * involvement
    df["exp_bonus"] = df["rate_bonus"] * involvement
    df["exp_cards"] = df["card_points_per_appearance"] * df["p_appear"]
    return df
