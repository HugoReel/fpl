"""Reconcile curated player_gw against archive season totals.

Every player's summed goals, assists and minutes from player_gw must equal
the season totals in players_raw.csv exactly. Tolerance zero, because both
numbers come from the same upstream and any gap means the curation dropped
or duplicated rows.

Each run writes data/curated/{season}/validation_report.csv listing every
mismatching stat. The run fails if more than 1 percent of players mismatch.

Usage:
    python -m ingest.validate --all
    python -m ingest.validate --seasons 2025-26
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ingest.curate import CURATED_ROOT

log = logging.getLogger(__name__)

RECONCILE_COLUMNS = ["minutes", "goals_scored", "assists"]
MISMATCH_LIMIT = 0.01


class ValidationError(RuntimeError):
    pass


def reconcile(player_gw: pd.DataFrame, players_raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player per mismatching stat. Empty means clean.

    Players absent from player_gw must have zero season totals, so the
    comparison runs over everyone in players_raw, not just those who played.
    Manager rows (element_type outside 1-4) are excluded upstream and here.
    """
    raw = players_raw[players_raw["element_type"].isin([1, 2, 3, 4])]
    expected = raw[["id", "web_name", *RECONCILE_COLUMNS]].rename(columns={"id": "player_id"})

    if player_gw.empty:
        got = pd.DataFrame(columns=["player_id", *RECONCILE_COLUMNS])
    else:
        got = player_gw.groupby("player_id", as_index=False)[RECONCILE_COLUMNS].sum()

    merged = expected.merge(got, on="player_id", how="left", suffixes=("_expected", "_got"))

    rows = []
    for col in RECONCILE_COLUMNS:
        exp = merged[f"{col}_expected"].fillna(0)
        act = merged[f"{col}_got"].fillna(0)
        bad = exp != act
        for _, r in merged[bad].iterrows():
            rows.append(
                {
                    "player_id": r["player_id"],
                    "web_name": r["web_name"],
                    "stat": col,
                    "expected": r[f"{col}_expected"],
                    "got": 0 if pd.isna(r[f"{col}_got"]) else r[f"{col}_got"],
                }
            )
    return pd.DataFrame(rows, columns=["player_id", "web_name", "stat", "expected", "got"])


def validate_season(
    season: str,
    curated_root: Path = CURATED_ROOT,
    external_root: Path | None = None,
) -> pd.DataFrame:
    """Reconcile one season, write its report, raise if over the limit."""
    # Deferred import: history imports this module at run time.
    from ingest.history import EXTERNAL_ROOT

    if external_root is None:
        external_root = EXTERNAL_ROOT

    player_gw = pd.read_parquet(curated_root / season / "player_gw.parquet")
    players_raw = pd.read_csv(external_root / season / "players_raw.csv", low_memory=False)

    report = reconcile(player_gw, players_raw)
    report_path = curated_root / season / "validation_report.csv"
    report.to_csv(report_path, index=False)

    n_players = players_raw["element_type"].isin([1, 2, 3, 4]).sum()
    n_bad = report["player_id"].nunique() if not report.empty else 0
    share = n_bad / n_players if n_players else 0.0
    log.info(
        "%s: %d of %d players mismatch (%.2f%%), report at %s",
        season,
        n_bad,
        n_players,
        100 * share,
        report_path,
    )
    if share > MISMATCH_LIMIT:
        raise ValidationError(
            f"{season}: {n_bad}/{n_players} players fail reconciliation "
            f"({100 * share:.2f}% > {100 * MISMATCH_LIMIT:.0f}%), see {report_path}"
        )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from ingest.history import DEFAULT_SEASONS

    ap = argparse.ArgumentParser(description="Reconcile curated tables against archive totals")
    ap.add_argument("--seasons", nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help=f"validate {DEFAULT_SEASONS}")
    args = ap.parse_args()

    if args.all:
        seasons = DEFAULT_SEASONS
    elif args.seasons:
        seasons = args.seasons
    else:
        raise SystemExit("pass --seasons or --all")

    for season in seasons:
        validate_season(season)


if __name__ == "__main__":
    main()
