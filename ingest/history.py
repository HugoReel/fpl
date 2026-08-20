"""Historical seasons from the vaastav archive, curated to curate.py shapes.

Downstream code must not be able to tell historical and live data apart, so
this module produces the same tables with the same column names and dtypes
that curate.py writes, plus a source column ("vaastav" here, "live" there).

Source: github.com/vaastav/Fantasy-Premier-League under data/{season}/.
Files are downloaded once into data/external/vaastav/{season}/ and never
re-downloaded while present. Only four files per season are needed:
merged_gw.csv (player x fixture rows), players_raw.csv (season player list
with the stable code field), fixtures.csv and teams.csv.

Identity: FPL element ids are NOT stable across seasons. The stable key is
code from players_raw.csv, carried as player_code on every player table.
data/curated/player_index.parquet maps code to per-season ids, names and
positions. Positions are kept per season, not global, because players get
reclassified between seasons.

Usage:
    python -m ingest.history --all
    python -m ingest.history --seasons 2021-22 2022-23
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests

from ingest.curate import (
    CURATED_ROOT,
    EXPLAIN_IDENTIFIERS,
    _write,
    build_fixtures,
    build_players,
    build_player_gw,
    rule_regime,
)
from scoring.rules_2026_27 import ELEMENT_TYPE_TO_POSITION

EXTERNAL_ROOT = Path("data/external/vaastav")
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
USER_AGENT = "fpl-research/0.1 (personal project)"
TIMEOUT = 60

DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Local filename -> path within the archive's data/{season}/ directory
ARCHIVE_FILES = {
    "merged_gw.csv": "gws/merged_gw.csv",
    "players_raw.csv": "players_raw.csv",
    "fixtures.csv": "fixtures.csv",
    "teams.csv": "teams.csv",
}

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Column contracts. The archive's column set drifts between seasons, so the
# expectation is explicit per season. A missing expected column is a hard
# failure, a new column is a warning worth reading.
# --------------------------------------------------------------------------

MERGED_GW_BASE = {
    "name",
    "position",
    "team",
    "xP",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "element",
    "fixture",
    "ict_index",
    "influence",
    "threat",
    "transfers_balance",
    "goals_conceded",
    "goals_scored",
    "kickoff_time",
    "minutes",
    "opponent_team",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "round",
    "saves",
    "selected",
    "team_a_score",
    "team_h_score",
    "total_points",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "yellow_cards",
    "GW",
}

_XG_COLUMNS = {
    "starts",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
}

_MNG_COLUMNS = {
    "mng_clean_sheets",
    "mng_draw",
    "mng_goals_scored",
    "mng_loss",
    "mng_underdog_draw",
    "mng_underdog_win",
    "mng_win",
}

_DEFCON_COLUMNS = {
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "recoveries",
    "tackles",
}

MERGED_GW_EXPECTED = {
    "2021-22": MERGED_GW_BASE,
    "2022-23": MERGED_GW_BASE | _XG_COLUMNS,
    "2023-24": MERGED_GW_BASE | _XG_COLUMNS,
    "2024-25": MERGED_GW_BASE | _XG_COLUMNS | _MNG_COLUMNS | {"modified"},
    "2025-26": MERGED_GW_BASE | _XG_COLUMNS | _DEFCON_COLUMNS | {"modified"},
}

PLAYERS_RAW_REQUIRED = {
    "id",
    "code",
    "element_type",
    "team",
    "web_name",
    "first_name",
    "second_name",
    "now_cost",
    "status",
    "minutes",
    "goals_scored",
    "assists",
    "total_points",
}

# merged_gw stat columns already carry the live explain identifier names,
# with defensive_contribution existing only from 2025-26.
STAT_COLUMNS = sorted(EXPLAIN_IDENTIFIERS)

PLAYABLE_ELEMENT_TYPES = set(ELEMENT_TYPE_TO_POSITION)  # 1-4, excludes managers


# --------------------------------------------------------------------------
# Download and load
# --------------------------------------------------------------------------


def ensure_files(season: str, root: Path = EXTERNAL_ROOT) -> dict[str, Path]:
    """Download the season's archive files unless already cached."""
    season_dir = root / season
    season_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for local_name, remote_path in ARCHIVE_FILES.items():
        path = season_dir / local_name
        if not path.exists():
            url = f"{BASE}/{season}/{remote_path}"
            log.info("GET %s", url)
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            log.info("cached %s (%d bytes)", path, path.stat().st_size)
        out[local_name] = path
    return out


def load_season_files(season: str, root: Path = EXTERNAL_ROOT) -> dict[str, pd.DataFrame]:
    files = ensure_files(season, root)
    return {name: pd.read_csv(path, low_memory=False) for name, path in files.items()}


def check_merged_gw_columns(df: pd.DataFrame, season: str) -> None:
    """Fail loud on schema drift, per season."""
    expected = MERGED_GW_EXPECTED.get(season)
    if expected is None:
        raise ValueError(
            f"no merged_gw column contract for season {season}. Inspect the file "
            "and add an entry to MERGED_GW_EXPECTED."
        )
    present = set(df.columns)
    missing = expected - present
    if missing:
        raise ValueError(f"{season} merged_gw missing expected columns: {sorted(missing)}")
    new = present - expected
    if new:
        log.warning("%s merged_gw has new columns, worth a look: %s", season, sorted(new))


def check_players_raw_columns(df: pd.DataFrame, season: str) -> None:
    missing = PLAYERS_RAW_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{season} players_raw missing required columns: {sorted(missing)}")


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------


def build_player_fixture_history(
    merged_gw: pd.DataFrame, players_raw: pd.DataFrame, season: str
) -> pd.DataFrame:
    """One row per player per fixture, matching curate.build_player_fixture.

    Position identity comes from players_raw element_type, not from the
    merged_gw position strings, which are inconsistent between and even
    within seasons (GK vs GKP in 2021-22, AM manager rows in 2024-25).
    """
    df = merged_gw.copy()

    # The archive occasionally carries exact duplicate rows for a player
    # listed under two name spellings. Values are identical, keep one.
    before = len(df)
    df = df.drop_duplicates(subset=["element", "fixture"])
    if len(df) < before:
        log.warning("%s merged_gw: dropped %d duplicate element+fixture rows", season, before - len(df))

    # xP is scraped after the gameweek finishes, so it describes the future
    # relative to the deadline. It leaks. Drop it entirely, do not shift it.
    df = df.drop(columns=["xP"], errors="ignore")

    meta = players_raw[["id", "code", "element_type"]].rename(
        columns={"id": "element", "code": "player_code"}
    )
    df = df.merge(meta, on="element", how="left")
    if df["element_type"].isna().any():
        orphans = df[df["element_type"].isna()]["element"].nunique()
        raise ValueError(f"{season}: {orphans} merged_gw elements missing from players_raw")

    non_playable = ~df["element_type"].isin(PLAYABLE_ELEMENT_TYPES)
    if non_playable.any():
        log.info(
            "%s: dropping %d rows for non-player element types (managers)",
            season,
            int(non_playable.sum()),
        )
        df = df[~non_playable]

    df = df.rename(columns={"element": "player_id", "round": "gw", "fixture": "fixture_id"})

    out = pd.DataFrame(
        {
            "season": season,
            "gw": df["gw"].astype("int64"),
            "player_id": df["player_id"].astype("int64"),
            "fixture_id": df["fixture_id"].astype("int64"),
        }
    )
    for col in STAT_COLUMNS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        else:
            # Pre-2025-26 seasons did not record defensive contributions.
            # Zero here means "not recorded", and rule_regime is how
            # downstream knows the difference. Do not treat these zeros as
            # real counts for pre_defcon seasons.
            out[col] = 0
    out["total_points"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0).astype("int64")
    out["started"] = out["minutes"].gt(0)
    out["played_60"] = out["minutes"].ge(60)
    out["player_code"] = df["player_code"].astype("int64")
    out["source"] = "vaastav"
    out["rule_regime"] = rule_regime(season)
    return out.reset_index(drop=True)


def build_players_history(
    players_raw: pd.DataFrame, teams: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Season player dimension via curate.build_players, for automatic parity.

    players_raw.csv is the bootstrap elements array as CSV and teams.csv is
    the bootstrap teams array, so the live builder consumes them directly.
    """
    check_players_raw_columns(players_raw, season)
    playable = players_raw[players_raw["element_type"].isin(PLAYABLE_ELEMENT_TYPES)]
    dropped = len(players_raw) - len(playable)
    if dropped:
        log.info("%s: dropping %d non-player elements (managers)", season, dropped)
    bootstrap = {
        "elements": playable.to_dict("records"),
        "teams": teams.to_dict("records"),
    }
    return build_players(bootstrap, season, source="vaastav")


def build_fixtures_history(fixtures: pd.DataFrame, season: str) -> pd.DataFrame:
    return build_fixtures(fixtures.to_dict("records"), season, source="vaastav")


def attach_prices_history(player_gw: pd.DataFrame, merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Attach per-gameweek price from merged_gw's value column.

    value is the price in tenths at that gameweek, so history is preserved
    the same way attach_prices does for live data by walking snapshots.
    The archive has no per-gw ownership percent, status or ep_next, so
    those columns exist for parity but hold nulls.
    """
    if player_gw.empty:
        return player_gw

    prices = (
        merged_gw.drop_duplicates(subset=["element", "round"], keep="last")[
            ["element", "round", "value"]
        ]
        .rename(columns={"element": "player_id", "round": "gw"})
    )
    prices["price"] = pd.to_numeric(prices["value"], errors="coerce") / 10.0
    prices = prices.drop(columns=["value"])

    df = player_gw.merge(prices, on=["player_id", "gw"], how="left")
    df["selected_by_percent"] = pd.Series(float("nan"), index=df.index, dtype="float64")
    df["status"] = pd.Series(None, index=df.index, dtype="object")
    df["ep_next"] = pd.Series(float("nan"), index=df.index, dtype="float64")
    return df


def build_player_index(curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Map stable player_code to per-season identity, across every curated season.

    Positions are per season on purpose. The same code can be a MID one
    season and a FWD the next, and points depend on which one was true.
    """
    frames = []
    for players_path in sorted(curated_root.glob("*/players.parquet")):
        df = pd.read_parquet(players_path)
        if "player_code" not in df.columns:
            log.warning("%s has no player_code, skipping (rebuild it)", players_path)
            continue
        frames.append(
            df[
                [
                    "player_code",
                    "season",
                    "player_id",
                    "web_name",
                    "first_name",
                    "second_name",
                    "element_type",
                    "team_id",
                    "team_name",
                    "source",
                ]
            ].copy()
        )
    if not frames:
        return pd.DataFrame()
    index = pd.concat(frames, ignore_index=True)
    index["position"] = index["element_type"].map(
        {k: v.value for k, v in ELEMENT_TYPE_TO_POSITION.items()}
    )
    return index.sort_values(["player_code", "season"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_season(season: str, root: Path = EXTERNAL_ROOT) -> dict[str, Path]:
    files = load_season_files(season, root)
    merged_gw = files["merged_gw.csv"]
    players_raw = files["players_raw.csv"]

    check_merged_gw_columns(merged_gw, season)

    out_dir = CURATED_ROOT / season
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    players = build_players_history(players_raw, files["teams.csv"], season)
    written["players"] = _write(players, out_dir / "players.parquet")

    fixtures = build_fixtures_history(files["fixtures.csv"], season)
    written["fixtures"] = _write(fixtures, out_dir / "fixtures.parquet")

    pf = build_player_fixture_history(merged_gw, players_raw, season)
    written["player_fixture"] = _write(pf, out_dir / "player_fixture.parquet")

    pgw = build_player_gw(pf, season)
    pgw = attach_prices_history(pgw, merged_gw)
    written["player_gw"] = _write(pgw, out_dir / "player_gw.parquet")

    return written


def run(seasons: list[str], root: Path = EXTERNAL_ROOT) -> dict[str, dict[str, Path]]:
    # Deferred import: validate imports helpers from this module.
    from ingest.validate import validate_season

    written: dict[str, dict[str, Path]] = {}
    for season in seasons:
        log.info("curating %s", season)
        written[season] = run_season(season, root)
        validate_season(season, external_root=root)

    index = build_player_index()
    if not index.empty:
        path = _write(index, CURATED_ROOT / "player_index.parquet")
        written["player_index"] = {"player_index": path}
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Curate vaastav archive seasons into Parquet")
    ap.add_argument("--seasons", nargs="+", default=None, help="e.g. 2021-22 2022-23")
    ap.add_argument("--all", action="store_true", help=f"curate {DEFAULT_SEASONS}")
    args = ap.parse_args()

    if args.all:
        seasons = DEFAULT_SEASONS
    elif args.seasons:
        seasons = args.seasons
    else:
        raise SystemExit("pass --seasons or --all")

    for season, tables in run(seasons).items():
        for name, path in tables.items():
            print(f"{season} {name}: {path}")


if __name__ == "__main__":
    main()
