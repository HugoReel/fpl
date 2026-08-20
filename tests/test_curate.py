"""Curation tests against synthetic snapshots shaped like the real API.

The double gameweek case is the one that matters. If player_fixture
collapses a DGW into one row, every threshold in the scoring module gets
applied to the wrong denominator and the whole model is quietly wrong.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from fpl.ingest import curate


def _write(tmp_path, season, name, payload, stamp):
    d = tmp_path / "data" / "raw" / season / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}_{stamp}.json").write_text(json.dumps(payload))


def _element(pid, element_type=3, team=1, cost=75, pens=None):
    return {
        "id": pid,
        "element_type": element_type,
        "team": team,
        "web_name": f"Player{pid}",
        "first_name": "A",
        "second_name": f"B{pid}",
        "now_cost": cost,
        "status": "a",
        "chance_of_playing_next_round": 100,
        "selected_by_percent": "12.3",
        "transfers_in_event": 1000,
        "transfers_out_event": 500,
        "ep_next": "4.5",
        "penalties_order": pens,
        "corners_and_indirect_freekicks_order": None,
        "direct_freekicks_order": None,
    }


def _explain(fixture_id, minutes, goals=0, points=0, defcon=0):
    stats = [
        {"identifier": "minutes", "value": minutes, "points": 2 if minutes >= 60 else 1},
        {"identifier": "goals_scored", "value": goals, "points": goals * 5},
    ]
    if defcon:
        stats.append(
            {"identifier": "defensive_contribution", "value": defcon, "points": 2}
        )
    return {"fixture": fixture_id, "stats": stats}


@pytest.fixture
def season_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(curate, "RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(curate, "CURATED_ROOT", tmp_path / "data" / "curated")

    season = "2026-27"

    bootstrap_gw1 = {
        "elements": [_element(1, pens=1), _element(2, element_type=2, cost=55)],
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Chelsea", "short_name": "CHE"},
        ],
        "events": [{"id": 1, "is_current": True}, {"id": 2, "is_current": False}],
    }
    # Price rise for player 1 by gw2
    bootstrap_gw2 = json.loads(json.dumps(bootstrap_gw1))
    bootstrap_gw2["elements"][0]["now_cost"] = 76
    bootstrap_gw2["events"] = [{"id": 1, "is_current": False}, {"id": 2, "is_current": True}]

    _write(tmp_path, season, "bootstrap-static", bootstrap_gw1, "20260815T180000Z")
    _write(tmp_path, season, "bootstrap-static", bootstrap_gw2, "20260822T180000Z")

    fixtures = [
        {
            "id": 10, "event": 1, "team_h": 1, "team_a": 2,
            "team_h_difficulty": 2, "team_a_difficulty": 4,
            "team_h_score": 2, "team_a_score": 0,
            "kickoff_time": "2026-08-15T14:00:00Z", "finished": True,
        },
        # Gameweek 2 is a double for team 1
        {
            "id": 11, "event": 2, "team_h": 2, "team_a": 1,
            "team_h_difficulty": 3, "team_a_difficulty": 3,
            "team_h_score": 1, "team_a_score": 1,
            "kickoff_time": "2026-08-22T14:00:00Z", "finished": True,
        },
        {
            "id": 12, "event": 2, "team_h": 1, "team_a": 2,
            "team_h_difficulty": 2, "team_a_difficulty": 4,
            "team_h_score": 3, "team_a_score": 1,
            "kickoff_time": "2026-08-24T19:00:00Z", "finished": True,
        },
    ]
    _write(tmp_path, season, "fixtures", fixtures, "20260822T180000Z")

    live_gw1 = {
        "elements": [
            {"id": 1, "explain": [_explain(10, 90, goals=1)]},
            {"id": 2, "explain": [_explain(10, 45, defcon=11)]},
        ]
    }
    live_gw2 = {
        "elements": [
            # Player 1 plays both legs of the double
            {"id": 1, "explain": [_explain(11, 90), _explain(12, 75, goals=2)]},
            {"id": 2, "explain": [_explain(11, 90, defcon=12), _explain(12, 20)]},
        ]
    }
    _write(tmp_path, season, "live_gw01", live_gw1, "20260818T090000Z")
    _write(tmp_path, season, "live_gw02", live_gw2, "20260825T090000Z")

    return season


def test_end_to_end_run(season_data):
    written = curate.run(season_data)
    assert set(written) == {"players", "fixtures", "player_fixture", "player_gw"}
    for path in written.values():
        assert path.exists()


def test_players_table(season_data):
    written = curate.run(season_data)
    players = pd.read_parquet(written["players"])
    assert len(players) == 2
    # Price stored as real millions, not tenths
    assert players.loc[players.player_id == 1, "price"].iloc[0] == 7.6
    assert players.loc[players.player_id == 1, "has_penalties"].iloc[0]
    assert not players.loc[players.player_id == 2, "has_penalties"].iloc[0]
    assert players.loc[players.player_id == 1, "team_short"].iloc[0] == "ARS"


def test_double_gameweek_stays_two_rows(season_data):
    written = curate.run(season_data)
    pf = pd.read_parquet(written["player_fixture"])
    gw2 = pf[(pf.gw == 2) & (pf.player_id == 1)]
    assert len(gw2) == 2, "double gameweek must not be collapsed"
    assert set(gw2.fixture_id) == {11, 12}


def test_player_gw_aggregates_double_correctly(season_data):
    written = curate.run(season_data)
    pgw = pd.read_parquet(written["player_gw"])
    row = pgw[(pgw.gw == 2) & (pgw.player_id == 1)].iloc[0]
    assert row.minutes == 165
    assert row.goals_scored == 2
    assert row.n_fixtures == 2
    assert row.n_played_60 == 2


def test_blank_and_double_flags_on_fixtures(season_data):
    written = curate.run(season_data)
    fixtures = pd.read_parquet(written["fixtures"])
    gw2 = fixtures[fixtures.gw == 2]
    # Team 1 plays twice in gw2, so both its fixtures know that
    assert (gw2[gw2.team_a == 1].a_fixtures_in_gw == 2).all()
    assert (gw2[gw2.team_h == 1].h_fixtures_in_gw == 2).all()


def test_prices_are_per_gameweek_not_latest(season_data):
    written = curate.run(season_data)
    pgw = pd.read_parquet(written["player_gw"])
    p1 = pgw[pgw.player_id == 1].set_index("gw")["price"]
    assert p1.loc[1] == 7.5, "gw1 must keep its historical price"
    assert p1.loc[2] == 7.6


def test_time_travel_excludes_later_snapshots(season_data):
    as_of = datetime(2026, 8, 20, tzinfo=timezone.utc)
    written = curate.run(season_data, as_of=as_of)
    pgw = pd.read_parquet(written["player_gw"])
    assert set(pgw.gw) == {1}, "gw2 data was captured after as_of and must not leak"

    players = pd.read_parquet(written["players"])
    assert players.loc[players.player_id == 1, "price"].iloc[0] == 7.5


def test_unknown_explain_identifier_is_logged_not_fatal(season_data, caplog):
    raw = curate.RAW_ROOT / season_data / "live_gw01"
    path = next(raw.glob("*.json"))
    payload = json.loads(path.read_text())
    payload["elements"][0]["explain"][0]["stats"].append(
        {"identifier": "brand_new_2027_stat", "value": 3, "points": 1}
    )
    path.write_text(json.dumps(payload))

    with caplog.at_level("WARNING"):
        curate.run(season_data)
    assert "brand_new_2027_stat" in caplog.text