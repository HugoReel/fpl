"""Turn raw API snapshots into tidy Parquet tables.

Raw is immutable and messy. Curated is typed, joined and modelable. This is
the only place that knows about the API's JSON shape, so when the shape
changes you fix it here and nothing downstream cares.

Produces four tables under data/curated/{season}/:

    players.parquet         one row per player, latest snapshot (dimension)
    fixtures.parquet        one row per fixture
    player_gw.parquet       one row per player per gameweek (aggregate)
    player_fixture.parquet  one row per player per fixture (DGW-safe)

player_fixture is the one that matters for modelling. Every FPL threshold
(60 minutes, clean sheet, DefCon) is per match, not per gameweek, so a
double gameweek aggregated first gives you wrong answers. Aggregate last.

Time travel: pass as_of to build the tables using only snapshots taken at
or before that timestamp. That is how the backtest replays the information
set a manager actually had at a deadline.

Usage:
    python -m ingest.curate --season 2026-27
    python -m ingest.curate --season 2026-27 --as-of 2026-09-12T10:00:00Z
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_ROOT = Path("data/raw")
CURATED_ROOT = Path("data/curated")

log = logging.getLogger(__name__)

_STAMP = re.compile(r"_(\d{8}T\d{6}Z)\.json$")

# Scoring identifiers that appear in the live endpoint's per-fixture
# "explain" blocks. Anything not in here is logged and ignored, which is
# how you find out FPL added a new scoring category.
EXPLAIN_IDENTIFIERS = {
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "defensive_contribution",
}

PLAYER_COLUMNS = {
    "id": "player_id",
    "element_type": "element_type",
    "team": "team_id",
    "web_name": "web_name",
    "first_name": "first_name",
    "second_name": "second_name",
    "now_cost": "now_cost",
    "status": "status",
    "chance_of_playing_next_round": "chance_of_playing_next_round",
    "selected_by_percent": "selected_by_percent",
    "transfers_in_event": "transfers_in_event",
    "transfers_out_event": "transfers_out_event",
    "ep_next": "ep_next",
    "penalties_order": "penalties_order",
    "corners_and_indirect_freekicks_order": "corners_order",
    "direct_freekicks_order": "direct_freekicks_order",
}

NUMERIC_PLAYER_COLUMNS = [
    "now_cost",
    "chance_of_playing_next_round",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "ep_next",
    "penalties_order",
    "corners_order",
    "direct_freekicks_order",
]


# --------------------------------------------------------------------------
# Snapshot discovery
# --------------------------------------------------------------------------


def snapshot_timestamp(path: Path) -> datetime:
    m = _STAMP.search(path.name)
    if not m:
        raise ValueError(f"cannot parse timestamp from {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def list_snapshots(season: str, name: str, as_of: datetime | None = None) -> list[Path]:
    """All snapshots for an endpoint, oldest first, optionally time-limited."""
    directory = RAW_ROOT / season / name
    if not directory.exists():
        return []
    paths = sorted(directory.glob(f"{name}_*.json"), key=snapshot_timestamp)
    if as_of is not None:
        paths = [p for p in paths if snapshot_timestamp(p) <= as_of]
    return paths


def latest_snapshot(season: str, name: str, as_of: datetime | None = None) -> Path | None:
    paths = list_snapshots(season, name, as_of)
    return paths[-1] if paths else None


def load(path: Path):
    return json.loads(path.read_text())


def live_gameweeks(season: str, as_of: datetime | None = None) -> dict[int, Path]:
    """Latest live snapshot per gameweek, keyed by gw number."""
    root = RAW_ROOT / season
    if not root.exists():
        return {}
    out: dict[int, Path] = {}
    for directory in sorted(root.glob("live_gw*")):
        gw = int(directory.name.replace("live_gw", ""))
        path = latest_snapshot(season, directory.name, as_of)
        if path is not None:
            out[gw] = path
    return out


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------


def build_players(bootstrap: dict, season: str) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["elements"])
    keep = [c for c in PLAYER_COLUMNS if c in df.columns]
    df = df[keep].rename(columns=PLAYER_COLUMNS)

    for col in NUMERIC_PLAYER_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # now_cost is in tenths of a million. Store the real number so nobody
    # downstream has to remember.
    if "now_cost" in df.columns:
        df["price"] = df["now_cost"] / 10.0

    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]]
    teams.columns = ["team_id", "team_name", "team_short"]
    df = df.merge(teams, on="team_id", how="left")

    df["has_penalties"] = df.get("penalties_order", pd.Series(dtype=float)).eq(1)
    df["season"] = season
    return df


def build_fixtures(fixtures: list[dict], season: str) -> pd.DataFrame:
    df = pd.DataFrame(fixtures)
    keep = [
        "id",
        "event",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
        "team_h_score",
        "team_a_score",
        "kickoff_time",
        "finished",
    ]
    df = df[[c for c in keep if c in df.columns]].rename(
        columns={"id": "fixture_id", "event": "gw"}
    )
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)

    # Blank and double gameweek flags, derived per team per gw
    played = df.dropna(subset=["gw"])
    counts = (
        pd.concat(
            [
                played[["gw", "team_h"]].rename(columns={"team_h": "team_id"}),
                played[["gw", "team_a"]].rename(columns={"team_a": "team_id"}),
            ]
        )
        .value_counts()
        .rename("fixtures_in_gw")
        .reset_index()
    )
    df = df.merge(
        counts.rename(columns={"team_id": "team_h", "fixtures_in_gw": "h_fixtures_in_gw"}),
        on=["gw", "team_h"],
        how="left",
    ).merge(
        counts.rename(columns={"team_id": "team_a", "fixtures_in_gw": "a_fixtures_in_gw"}),
        on=["gw", "team_a"],
        how="left",
    )

    df["season"] = season
    return df


def build_player_fixture(
    live_by_gw: dict[int, Path], season: str
) -> pd.DataFrame:
    """One row per player per fixture, from the live endpoint's explain blocks.

    This is the DGW-safe table. Use it for anything involving a per-match
    threshold, which is most things.
    """
    rows: list[dict] = []
    unknown: set[str] = set()

    for gw, path in sorted(live_by_gw.items()):
        payload = load(path)
        for element in payload.get("elements", []):
            player_id = element["id"]
            for block in element.get("explain", []):
                row = {
                    "season": season,
                    "gw": gw,
                    "player_id": player_id,
                    "fixture_id": block.get("fixture"),
                }
                points = 0
                for stat in block.get("stats", []):
                    ident = stat.get("identifier")
                    if ident not in EXPLAIN_IDENTIFIERS:
                        unknown.add(ident)
                        continue
                    row[ident] = stat.get("value", 0)
                    points += stat.get("points", 0)
                row["total_points"] = points
                rows.append(row)

    if unknown:
        log.warning("unrecognised explain identifiers, check for rule changes: %s", sorted(unknown))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in EXPLAIN_IDENTIFIERS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["started"] = df["minutes"].gt(0)
    df["played_60"] = df["minutes"].ge(60)
    return df


def build_player_gw(player_fixture: pd.DataFrame, season: str) -> pd.DataFrame:
    """Gameweek aggregate. Derived from player_fixture, never built directly.

    Note the appearance columns: n_fixtures and n_played_60 are counts, not
    booleans, because a double gameweek can produce two of each.
    """
    if player_fixture.empty:
        return pd.DataFrame()

    sum_cols = [c for c in EXPLAIN_IDENTIFIERS if c in player_fixture.columns]
    agg = {c: "sum" for c in sum_cols}
    agg["total_points"] = "sum"

    df = player_fixture.groupby(["season", "gw", "player_id"], as_index=False).agg(agg)
    counts = (
        player_fixture.groupby(["season", "gw", "player_id"], as_index=False)
        .agg(n_fixtures=("fixture_id", "count"), n_played_60=("played_60", "sum"))
    )
    return df.merge(counts, on=["season", "gw", "player_id"], how="left")


def attach_prices(player_gw: pd.DataFrame, season: str, as_of: datetime | None) -> pd.DataFrame:
    """Attach the price and ownership as at each gameweek's last snapshot.

    Prices move nightly, so a single latest-price column silently rewrites
    history. This walks the bootstrap snapshots instead.
    """
    if player_gw.empty:
        return player_gw

    frames = []
    for path in list_snapshots(season, "bootstrap-static", as_of):
        payload = load(path)
        gw = next((e["id"] for e in payload.get("events", []) if e.get("is_current")), None)
        if gw is None:
            continue
        snap = pd.DataFrame(payload["elements"])[
            ["id", "now_cost", "selected_by_percent", "status", "ep_next"]
        ]
        snap.columns = ["player_id", "now_cost", "selected_by_percent", "status", "ep_next"]
        snap["gw"] = gw
        snap["snapshot_ts"] = snapshot_timestamp(path)
        frames.append(snap)

    if not frames:
        log.warning("no bootstrap snapshots with a current gameweek, skipping prices")
        return player_gw

    prices = pd.concat(frames).sort_values("snapshot_ts")
    prices = prices.drop_duplicates(subset=["player_id", "gw"], keep="last")
    prices["price"] = pd.to_numeric(prices["now_cost"], errors="coerce") / 10.0
    prices = prices.drop(columns=["now_cost", "snapshot_ts"])
    return player_gw.merge(prices, on=["player_id", "gw"], how="left")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run(season: str, as_of: datetime | None = None) -> dict[str, Path]:
    out_dir = CURATED_ROOT / season
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    bootstrap_path = latest_snapshot(season, "bootstrap-static", as_of)
    if bootstrap_path is None:
        raise FileNotFoundError(
            f"no bootstrap snapshots for {season}. Run ingest.snapshot first."
        )
    bootstrap = load(bootstrap_path)
    log.info("using bootstrap from %s", snapshot_timestamp(bootstrap_path).isoformat())

    players = build_players(bootstrap, season)
    written["players"] = _write(players, out_dir / "players.parquet")

    fixtures_path = latest_snapshot(season, "fixtures", as_of)
    if fixtures_path is not None:
        fixtures = build_fixtures(load(fixtures_path), season)
        written["fixtures"] = _write(fixtures, out_dir / "fixtures.parquet")

    live_by_gw = live_gameweeks(season, as_of)
    if live_by_gw:
        log.info("found live snapshots for gameweeks %s", sorted(live_by_gw))
        pf = build_player_fixture(live_by_gw, season)
        written["player_fixture"] = _write(pf, out_dir / "player_fixture.parquet")

        pgw = build_player_gw(pf, season)
        pgw = attach_prices(pgw, season, as_of)
        written["player_gw"] = _write(pgw, out_dir / "player_gw.parquet")
    else:
        log.info("no live snapshots yet, season has not started or none captured")

    return written


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Curate raw FPL snapshots into Parquet")
    ap.add_argument("--season", required=True, help="e.g. 2026-27")
    ap.add_argument(
        "--as-of",
        default=None,
        help="ISO timestamp. Only use snapshots at or before this, for backtest replay.",
    )
    args = ap.parse_args()

    as_of = None
    if args.as_of:
        as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))

    for name, path in run(args.season, as_of).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
