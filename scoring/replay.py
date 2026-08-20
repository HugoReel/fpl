"""Recompute realised points from curated components.

Stored totals are not trusted anywhere in this project. They were awarded
under whichever rules applied that season, and they arrive through an
archive that has been wrong before. Everything that needs to know what a
player actually scored comes through here instead, which routes every
calculation into `rules_2026_27.py`.

Points are always restated under the current rules rather than the rules of
the season they were earned in. That is deliberate. A model predicting
2026/27 points needs history expressed on the 2026/27 scale, so a defender's
goal in 2022-23 is worth the 6 points it would be worth now, not the 6 it
was worth then, and a midfielder's is worth 5 either way.

The one component that cannot be restated is defensive contribution, which
simply was not recorded before 2025-26. Those rows carry zero, and
`rule_regime` on the curated tables is what tells downstream code the
difference between "did nothing" and "nobody was counting". Do not read a
pre_defcon zero as evidence about a player.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ingest.curate import CURATED_ROOT
from scoring.rules_2026_27 import ELEMENT_TYPE_TO_POSITION, MatchStats, score_match

log = logging.getLogger(__name__)

# Component columns the scoring module needs, as they appear on the curated
# player_fixture table.
COMPONENT_COLUMNS = [
    "minutes",
    "goals_scored",
    "assists",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "defensive_contribution",
]


def score_rows(df: pd.DataFrame) -> pd.Series:
    """Points for each player-fixture row, recomputed from its components."""
    missing = [c for c in COMPONENT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"cannot recompute points, missing components: {missing}")
    if "element_type" not in df.columns:
        raise KeyError("cannot recompute points without element_type")

    points = []
    for row in df.to_dict("records"):
        stats = MatchStats(
            minutes=int(row["minutes"]),
            goals=int(row["goals_scored"]),
            assists=int(row["assists"]),
            goals_conceded=int(row["goals_conceded"]),
            saves=int(row["saves"]),
            penalties_saved=int(row["penalties_saved"]),
            penalties_missed=int(row["penalties_missed"]),
            yellow_cards=int(row["yellow_cards"]),
            red_cards=int(row["red_cards"]),
            own_goals=int(row["own_goals"]),
            bonus=int(row["bonus"]),
            defensive_actions=int(row["defensive_contribution"]),
        )
        points.append(score_match(stats, ELEMENT_TYPE_TO_POSITION[int(row["element_type"])]).total)
    return pd.Series(points, index=df.index, dtype="int64")


def load_player_fixtures(
    season: str, curated_root: Path = CURATED_ROOT, gw: int | None = None
) -> pd.DataFrame:
    """Curated player-fixture rows with element_type and recomputed points."""
    pf = pd.read_parquet(curated_root / season / "player_fixture.parquet")
    if gw is not None:
        pf = pf[pf["gw"] == gw]
    players = pd.read_parquet(curated_root / season / "players.parquet")
    pf = pf.merge(players[["player_id", "element_type"]], on="player_id", how="left")
    return pf.assign(realised=score_rows(pf))


def realised_gameweek_points(
    season: str, gw: int | None = None, curated_root: Path = CURATED_ROOT
) -> pd.DataFrame:
    """Points per player per gameweek, summed across fixtures.

    Summing after scoring rather than before is what keeps a double
    gameweek correct, because every threshold in FPL is per match.
    """
    pf = load_player_fixtures(season, curated_root, gw)
    out = pf.groupby(["player_id", "gw"], as_index=False).agg(
        realised=("realised", "sum"), n_fixtures=("fixture_id", "count")
    )
    return out
