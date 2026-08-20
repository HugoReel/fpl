"""Baseline: the pre-model system, frozen.

This is variant B from experiment 1: trailing per-90 rates, and a minutes
rule of "if he started last week he plays 90, otherwise he plays none".
Composed through the scoring module exactly as the real pipeline is.

It is the system this project would have had without a minutes model, so
it isolates what the minutes model and the rate estimators are actually
worth end to end.

Deliberately frozen. It is defined in full here rather than importing from
`experiments/` or from `models/rates.py`, because a reference point that
moves when the thing it references moves is not a reference point. The
duplication is the feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.minutes.features import as_of_gameweek
from scoring.rules_2026_27 import ELEMENT_TYPE_TO_POSITION, expected_points as score_ep

log = logging.getLogger(__name__)

NAME = "naive_minutes_ep"

# Frozen constants. Do not tune these to make the baseline look better or
# worse, and do not sync them with models/rates.py when that changes.
TRAILING_MATCHES = 5
CARD_COST_PER_APPEARANCE = 0.09
DEFCON_THRESHOLDS = {1: None, 2: 10, 3: 12, 4: 12}


def _prev_mean(df: pd.DataFrame, values: pd.Series, window: int = TRAILING_MATCHES) -> pd.Series:
    """Mean over a player's previous matches, frozen at the gameweek boundary."""
    shifted = values.groupby(df["player_code"], sort=False).shift(1)
    rolled = (
        shifted.groupby(df["player_code"], sort=False)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )
    return as_of_gameweek(df, rolled)


def expected_points(full: pd.DataFrame, season: str) -> pd.DataFrame:
    """Naive minutes plus trailing rates, composed through the scoring module."""
    df = full.sort_values(["kickoff_time", "player_code"], kind="stable").reset_index(drop=True)

    num = lambda col: pd.to_numeric(df[col], errors="coerce")  # noqa: E731
    prev_minutes = _prev_mean(df, num("minutes"))
    per90 = (prev_minutes / 90.0).clip(lower=0.05)

    rate_goals = (_prev_mean(df, num("goals_scored")) / per90).fillna(0.0)
    rate_assists = (_prev_mean(df, num("assists")) / per90).fillna(0.0)
    rate_saves = (_prev_mean(df, num("saves")) / per90).fillna(0.0)
    rate_conceded = (_prev_mean(df, num("goals_conceded")) / per90).fillna(0.0)
    rate_bonus = _prev_mean(df, num("bonus")).fillna(0.0)
    rate_cs = _prev_mean(df, num("clean_sheets")).fillna(0.0).clip(0, 1)

    thresholds = df["element_type"].map(DEFCON_THRESHOLDS).astype(float)
    hit_defcon = (num("defensive_contribution") >= thresholds).astype(float)
    hit_defcon = hit_defcon.where(thresholds.notna(), 0.0)
    rate_defcon = _prev_mean(df, hit_defcon).fillna(0.0).clip(0, 1)

    # The naive rule: started last week means a full match, otherwise nothing.
    started_last = df["start_rate_1"].fillna(0.0) > 0
    p_appear = started_last.astype(float)

    ep = np.empty(len(df))
    records = df.to_dict("records")
    for i, row in enumerate(records):
        ep[i] = score_ep(
            ELEMENT_TYPE_TO_POSITION[int(row["element_type"])],
            p_appear=float(p_appear.iloc[i]),
            p_60plus=float(p_appear.iloc[i]),
            exp_goals=float(rate_goals.iloc[i] * p_appear.iloc[i]),
            exp_assists=float(rate_assists.iloc[i] * p_appear.iloc[i]),
            p_clean_sheet=float(rate_cs.iloc[i]),
            exp_goals_conceded=float(rate_conceded.iloc[i] * p_appear.iloc[i]),
            exp_saves=float(rate_saves.iloc[i] * p_appear.iloc[i]),
            p_defcon=float(rate_defcon.iloc[i] * p_appear.iloc[i]),
            exp_bonus=float(rate_bonus.iloc[i] * p_appear.iloc[i]),
            exp_cards=CARD_COST_PER_APPEARANCE * float(p_appear.iloc[i]),
        )
    df["ep_total"] = ep

    out = df[df["season"] == season]
    return out.groupby(["season", "gw", "player_id", "player_code"], as_index=False).agg(
        ep_total=("ep_total", "sum"), n_fixtures=("fixture_id", "count")
    )
