"""Immutable snapshots of the FPL API.

Run this on a schedule. The FPL API overwrites itself, so anything you do
not capture at the time is gone. These snapshots become your backtest
ground truth for every future season, and they cost about 2 MB a week.

Write once, never mutate. Curation happens downstream in curate.py.

Usage:
    python -m ingest.snapshot --season 2026-27
    python -m ingest.snapshot --season 2026-27 --gw 3

Schedule it at minimum:
    - Friday evening   (team news is in, prices settled)
    - Saturday pre deadline
    - Tuesday          (gameweek finalised, points locked)
Cron is fine. GitHub Actions is fine. Do not overthink it.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://fantasy.premierleague.com/api"
RAW_ROOT = Path("data/raw")
USER_AGENT = "fpl-research/0.1 (personal project)"
TIMEOUT = 30

log = logging.getLogger(__name__)

# Columns we expect to exist. The API changes shape most seasons, so we
# check rather than assume. A missing column is a hard failure, a new
# column is a warning worth reading.
EXPECTED_ELEMENT_FIELDS = {
    "id",
    "element_type",
    "team",
    "web_name",
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "ep_next",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "total_points",
    # 2025/26 onwards
    "defensive_contribution",
    # set piece order fields, official since 2024/25
    "penalties_order",
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
}

ENDPOINTS = {
    "bootstrap-static": "/bootstrap-static/",
    "fixtures": "/fixtures/",
}


def fetch(path: str) -> dict | list:
    url = f"{BASE}{path}"
    log.info("GET %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def snapshot_path(season: str, name: str, ts: datetime) -> Path:
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    return RAW_ROOT / season / name / f"{name}_{stamp}.json"


def write_snapshot(payload, season: str, name: str, ts: datetime) -> Path:
    path = snapshot_path(season, name, ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite snapshot {path}")
    path.write_text(json.dumps(payload, separators=(",", ":")))
    log.info("wrote %s (%d bytes)", path, path.stat().st_size)
    return path


def validate_bootstrap(payload: dict) -> None:
    """Fail loud on schema drift. This is not optional."""
    if "elements" not in payload or not payload["elements"]:
        raise ValueError("bootstrap-static has no elements, API shape changed")

    present = set(payload["elements"][0].keys())
    missing = EXPECTED_ELEMENT_FIELDS - present
    if missing:
        raise ValueError(f"bootstrap elements missing expected fields: {sorted(missing)}")

    new = present - EXPECTED_ELEMENT_FIELDS
    if new:
        log.warning("new bootstrap element fields, worth a look: %s", sorted(new))


def current_gameweek(bootstrap: dict) -> int | None:
    for event in bootstrap.get("events", []):
        if event.get("is_current"):
            return event["id"]
    return None


def run(season: str, gw: int | None = None) -> dict[str, Path]:
    ts = datetime.now(timezone.utc)
    written: dict[str, Path] = {}

    bootstrap = fetch(ENDPOINTS["bootstrap-static"])
    validate_bootstrap(bootstrap)
    written["bootstrap-static"] = write_snapshot(bootstrap, season, "bootstrap-static", ts)

    fixtures = fetch(ENDPOINTS["fixtures"])
    written["fixtures"] = write_snapshot(fixtures, season, "fixtures", ts)

    target_gw = gw if gw is not None else current_gameweek(bootstrap)
    if target_gw is not None:
        live = fetch(f"/event/{target_gw}/live/")
        written["live"] = write_snapshot(live, season, f"live_gw{target_gw:02d}", ts)
    else:
        log.info("no current gameweek, skipping live pull")

    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Snapshot the FPL API to immutable JSON")
    ap.add_argument("--season", required=True, help="e.g. 2026-27")
    ap.add_argument("--gw", type=int, default=None, help="override the live gameweek")
    args = ap.parse_args()

    written = run(args.season, args.gw)
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
