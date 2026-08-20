"""Baseline: FPL's own published expected points.

The game publishes an `ep_next` figure for every player. It is a genuine
external benchmark, produced by people with access to data this project
does not have, so beating it means something.

It is only usable on live snapshots. The vaastav archive preserves the
end of season bootstrap, not a weekly one, so for a historical season there
is no honest way to recover what ep_next said before each deadline. This
module returns an empty frame and says so in the log rather than
substituting the end of season value, which would be a leak dressed up as
a baseline.

That gap closes on its own. `ingest/snapshot.py` captures ep_next every
week, so once a season of live snapshots exists this baseline starts
working for that season with no changes here.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

NAME = "ep_next"

EMPTY_COLUMNS = ["season", "gw", "player_id", "player_code", "ep_total", "n_fixtures"]


def expected_points(full: pd.DataFrame, season: str) -> pd.DataFrame:
    """FPL's ep_next per player per gameweek, where it was actually captured."""
    rows = full[full["season"] == season]
    if "ep_next" not in rows.columns or rows["ep_next"].notna().sum() == 0:
        log.warning(
            "ep_next is not available for %s. The archive preserves only the end of "
            "season bootstrap, so no pre deadline value exists for a past gameweek. "
            "This baseline starts working once a season of live snapshots has been "
            "collected. Returning an empty frame rather than faking it.",
            season,
        )
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    out = rows.copy()
    out["ep_total"] = pd.to_numeric(out["ep_next"], errors="coerce").fillna(0.0)
    return out.groupby(["season", "gw", "player_id", "player_code"], as_index=False).agg(
        ep_total=("ep_total", "first"), n_fixtures=("fixture_id", "count")
    )
