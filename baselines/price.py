"""Baseline: a player is worth exactly what they cost.

Price is the crowd's own expected points estimate, set by millions of
managers and adjusted every night. As a ranking signal it is surprisingly
hard to beat, and it needs no data, no model and no history.

It is a deliberately awkward baseline for the optimiser, because the
objective becomes "spend as much as possible on the eleven". That is the
point. It shows what a squad built purely on reputation returns, and any
model claiming to add value has to beat a strategy that requires no
thought whatsoever.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

NAME = "price"


def expected_points(full: pd.DataFrame, season: str) -> pd.DataFrame:
    """Expected points equal to the player's price at that gameweek."""
    out = full[full["season"] == season].copy()
    out["ep_total"] = out["price"].astype(float)
    return out.groupby(["season", "gw", "player_id", "player_code"], as_index=False).agg(
        ep_total=("ep_total", "first"), n_fixtures=("fixture_id", "count")
    )
