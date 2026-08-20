"""Baseline: a player is worth what they averaged over their last 5 appearances.

This is the baseline that matters. It is what a reasonable human does when
they glance at the form column, it costs nothing to compute, and any model
that cannot beat it is not earning its complexity. Every candidate in the
evaluation harness is reported as a delta against this one.

An appearance means minutes on the pitch. A player who has not appeared is
worth zero rather than being dropped, because "he never plays" is a genuine
and useful prediction, and dropping him would quietly remove him from the
optimiser's pool.

Points are recomputed from components through the scoring module rather
than read from the stored total, so a 2022-23 defender's goal counts for
what it would be worth now. See `scoring/replay.py`.
"""

from __future__ import annotations

import logging

import pandas as pd

from models.minutes.features import as_of_gameweek
from scoring.replay import score_rows

log = logging.getLogger(__name__)

NAME = "last5"
WINDOW = 5


def expected_points(full: pd.DataFrame, season: str) -> pd.DataFrame:
    """Mean recomputed points over each player's previous WINDOW appearances.

    The window is evaluated as of the gameweek deadline, so both halves of
    a double gameweek see the same form figure.
    """
    df = full.sort_values(["kickoff_time", "player_code"], kind="stable").reset_index(drop=True)
    played = df["minutes"].fillna(0) > 0

    points = pd.Series(0.0, index=df.index)
    scored = score_rows(df[played])
    points.loc[scored.index] = scored.astype(float)
    # Only appearances count toward the average, so everything else is
    # hidden from the rolling window rather than counted as a zero.
    points = points.where(played)

    shifted = points.groupby(df["player_code"], sort=False).shift(1)
    rolled = (
        shifted.groupby(df["player_code"], sort=False)
        .rolling(WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )
    df["ep_total"] = as_of_gameweek(df, rolled).fillna(0.0)

    out = df[df["season"] == season]
    return out.groupby(["season", "gw", "player_id", "player_code"], as_index=False).agg(
        ep_total=("ep_total", "first"), n_fixtures=("fixture_id", "count")
    )
