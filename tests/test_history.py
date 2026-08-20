"""Historical ingest: archive curation, identity, parity and validation.

Synthetic seasons are built by hand so tests hit no network and no real
data files, except the last test which spot checks the real 2025-26 output
when it exists on disk.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import ingest.curate as curate
import ingest.history as history
from ingest.validate import ValidationError, reconcile, validate_season

SEASON = "2021-22"


def make_players_raw():
    base = {
        "chance_of_playing_next_round": 100,
        "selected_by_percent": "5.0",
        "transfers_in_event": 0,
        "transfers_out_event": 0,
        "ep_next": "2.5",
        "penalties_order": None,
        "corners_and_indirect_freekicks_order": None,
        "direct_freekicks_order": None,
    }
    rows = [
        # keeper, plays every match including a double in gw2
        dict(base, id=1, code=501, element_type=1, team=1, web_name="Keeper",
             first_name="Ken", second_name="Keeper", now_cost=55, status="a",
             minutes=270, goals_scored=0, assists=0, total_points=18),
        # defender, 45 minutes and a goal in gw1
        dict(base, id=2, code=502, element_type=2, team=1, web_name="Back",
             first_name="Bob", second_name="Back", now_cost=45, status="a",
             minutes=45, goals_scored=1, assists=0, total_points=7),
        # midfielder on the other team, two assists
        dict(base, id=3, code=503, element_type=3, team=2, web_name="Mid",
             first_name="May", second_name="Mid", now_cost=80, status="a",
             minutes=90, goals_scored=0, assists=2, total_points=8, penalties_order=1),
        # assistant manager, must be dropped everywhere
        dict(base, id=9, code=509, element_type=5, team=2, web_name="Boss",
             first_name="Big", second_name="Boss", now_cost=10, status="a",
             minutes=0, goals_scored=0, assists=0, total_points=12),
    ]
    return pd.DataFrame(rows)


def make_merged_gw():
    def row(element, position, gw, fixture, minutes, goals=0, assists=0, points=0):
        return {
            "name": f"player {element}",
            "position": position,
            "team": "Alpha",
            "xP": 4.2,
            "assists": assists,
            "bonus": 0,
            "bps": 10,
            "clean_sheets": 1 if minutes >= 60 else 0,
            "creativity": 0.0,
            "element": element,
            "fixture": fixture,
            "goals_conceded": 0,
            "goals_scored": goals,
            "ict_index": 0.0,
            "influence": 0.0,
            "kickoff_time": "2021-08-14T14:00:00Z",
            "minutes": minutes,
            "opponent_team": 2,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "red_cards": 0,
            "round": gw,
            "saves": 3,
            "selected": 1000,
            "team_a_score": 0,
            "team_h_score": 1,
            "threat": 0.0,
            "total_points": points,
            "transfers_balance": 0,
            "transfers_in": 0,
            "transfers_out": 0,
            "value": 55,
            "was_home": True,
            "yellow_cards": 0,
            "GW": gw,
        }

    return pd.DataFrame(
        [
            # note the archive's GK vs GKP inconsistency is irrelevant here,
            # identity comes from players_raw element_type
            row(1, "GK", 1, 1, 90, points=6),
            row(1, "GKP", 2, 2, 90, points=6),
            row(1, "GKP", 2, 3, 90, points=6),  # double gameweek
            row(2, "DEF", 1, 1, 45, goals=1, points=7),
            row(3, "MID", 1, 1, 90, assists=2, points=8),
            row(9, "AM", 1, 1, 0, points=12),  # manager row
        ]
    )


def make_fixtures():
    def fixture(fid, event, team_h, team_a):
        return {
            "code": 2210000 + fid,
            "event": event,
            "finished": True,
            "id": fid,
            "kickoff_time": "2021-08-14T14:00:00Z",
            "team_a": team_a,
            "team_a_score": 0,
            "team_h": team_h,
            "team_h_score": 1,
            "team_h_difficulty": 2,
            "team_a_difficulty": 3,
        }

    return pd.DataFrame([fixture(1, 1, 1, 2), fixture(2, 2, 1, 2), fixture(3, 2, 2, 1)])


def make_teams():
    return pd.DataFrame(
        [
            {"id": 1, "name": "Alpha FC", "short_name": "ALP"},
            {"id": 2, "name": "Beta Town", "short_name": "BET"},
        ]
    )


@pytest.fixture
def player_fixture_hist():
    return history.build_player_fixture_history(make_merged_gw(), make_players_raw(), SEASON)


@pytest.fixture
def player_gw_hist(player_fixture_hist):
    pgw = curate.build_player_gw(player_fixture_hist, SEASON)
    return history.attach_prices_history(pgw, make_merged_gw())


def test_xp_is_dropped_not_carried(player_fixture_hist, player_gw_hist):
    assert "xP" not in player_fixture_hist.columns
    assert "xP" not in player_gw_hist.columns


def test_manager_rows_dropped(player_fixture_hist):
    assert 9 not in player_fixture_hist["player_id"].values
    players = history.build_players_history(make_players_raw(), make_teams(), SEASON)
    assert 9 not in players["player_id"].values
    assert len(players) == 3


def test_player_code_on_every_player_table(player_fixture_hist, player_gw_hist):
    players = history.build_players_history(make_players_raw(), make_teams(), SEASON)
    for table in (players, player_fixture_hist, player_gw_hist):
        assert "player_code" in table.columns
        assert table["player_code"].notna().all()
    assert set(player_fixture_hist["player_code"]) == {501, 502, 503}


def test_dgw_keeps_per_fixture_rows_and_counts(player_fixture_hist, player_gw_hist):
    dgw_rows = player_fixture_hist.query("player_id == 1 and gw == 2")
    assert len(dgw_rows) == 2
    assert set(dgw_rows["fixture_id"]) == {2, 3}

    agg = player_gw_hist.query("player_id == 1 and gw == 2").iloc[0]
    assert agg["n_fixtures"] == 2
    assert agg["minutes"] == 180
    assert agg["n_played_60"] == 2


def test_defensive_contribution_zero_filled_pre_defcon(player_fixture_hist):
    # 2021-22 never recorded defensive actions. The column exists for
    # parity, holds zeros, and rule_regime says why.
    assert (player_fixture_hist["defensive_contribution"] == 0).all()
    assert player_fixture_hist["defensive_contribution"].dtype == "int64"
    assert (player_fixture_hist["rule_regime"] == "pre_defcon").all()
    assert (player_fixture_hist["source"] == "vaastav").all()


def test_column_contract_fails_loud():
    broken = make_merged_gw().drop(columns=["minutes"])
    with pytest.raises(ValueError, match="missing expected columns"):
        history.check_merged_gw_columns(broken, SEASON)
    with pytest.raises(ValueError, match="no merged_gw column contract"):
        history.check_merged_gw_columns(make_merged_gw(), "1999-00")


def _live_player_gw(tmp_path, monkeypatch):
    """Run the live curation path on synthetic payloads, no network."""
    season = "2026-27"
    elements = [
        {
            "id": 1, "code": 601, "element_type": 1, "team": 1, "web_name": "Keeper",
            "first_name": "Ken", "second_name": "Keeper", "now_cost": 55, "status": "a",
            "chance_of_playing_next_round": 100, "selected_by_percent": "5.0",
            "transfers_in_event": 0, "transfers_out_event": 0, "ep_next": "2.5",
            "penalties_order": None, "corners_and_indirect_freekicks_order": None,
            "direct_freekicks_order": None,
        }
    ]
    bootstrap = {
        "elements": elements,
        "teams": make_teams().to_dict("records"),
        "events": [{"id": 1, "is_current": True}],
    }
    raw_root = tmp_path / "raw"
    boot_dir = raw_root / season / "bootstrap-static"
    boot_dir.mkdir(parents=True)
    (boot_dir / "bootstrap-static_20260814T100000Z.json").write_text(json.dumps(bootstrap))
    monkeypatch.setattr(curate, "RAW_ROOT", raw_root)

    live = {
        "elements": [
            {
                "id": 1,
                "explain": [
                    {
                        "fixture": 1,
                        "stats": [
                            {"identifier": "minutes", "points": 2, "value": 90},
                            {"identifier": "saves", "points": 1, "value": 3},
                            {"identifier": "clean_sheets", "points": 4, "value": 1},
                        ],
                    }
                ],
            }
        ]
    }
    live_path = tmp_path / "live_gw01.json"
    live_path.write_text(json.dumps(live))

    pf = curate.build_player_fixture({1: live_path}, season, {1: 601})
    pgw = curate.build_player_gw(pf, season)
    return curate.attach_prices(pgw, season, as_of=None)


def test_live_and_historical_player_gw_identical_columns(
    tmp_path, monkeypatch, player_gw_hist
):
    live_pgw = _live_player_gw(tmp_path, monkeypatch)
    assert set(live_pgw.columns) == set(player_gw_hist.columns)

    # dtypes must agree for every stat column or models will notice
    for col in sorted(curate.EXPLAIN_IDENTIFIERS) + ["total_points", "n_fixtures", "n_played_60"]:
        assert live_pgw[col].dtype == player_gw_hist[col].dtype, col


def test_ensure_files_never_redownloads(tmp_path, monkeypatch):
    season_dir = tmp_path / SEASON
    season_dir.mkdir()
    for name in history.ARCHIVE_FILES:
        (season_dir / name).write_text("cached")

    def explode(*args, **kwargs):
        raise AssertionError("network hit despite warm cache")

    monkeypatch.setattr(history.requests, "get", explode)
    out = history.ensure_files(SEASON, root=tmp_path)
    assert set(out) == set(history.ARCHIVE_FILES)


def test_player_index_positions_are_per_season(tmp_path):
    curated = tmp_path / "curated"
    season_a, season_b = "2021-22", "2022-23"
    raw_a = make_players_raw()
    # player 2 is reclassified DEF -> MID between seasons, index must keep both
    raw_b = make_players_raw()
    raw_b.loc[raw_b["id"] == 2, "element_type"] = 3

    for season, raw in ((season_a, raw_a), (season_b, raw_b)):
        players = history.build_players_history(raw, make_teams(), season)
        out = curated / season
        out.mkdir(parents=True)
        players.to_parquet(out / "players.parquet", index=False)

    index = history.build_player_index(curated)
    rows = index[index["player_code"] == 502]
    assert len(rows) == 2
    assert set(zip(rows["season"], rows["position"])) == {(season_a, "DEF"), (season_b, "MID")}


def test_reconcile_clean_then_mismatch(player_gw_hist):
    raw = make_players_raw()
    assert reconcile(player_gw_hist, raw).empty

    raw.loc[raw["id"] == 2, "goals_scored"] = 5
    report = reconcile(player_gw_hist, raw)
    assert len(report) == 1
    assert report.iloc[0]["stat"] == "goals_scored"
    assert report.iloc[0]["expected"] == 5
    assert report.iloc[0]["got"] == 1


def test_validate_season_fails_over_limit(tmp_path, player_gw_hist):
    curated = tmp_path / "curated" / SEASON
    curated.mkdir(parents=True)
    player_gw_hist.to_parquet(curated / "player_gw.parquet", index=False)

    external = tmp_path / "external" / SEASON
    external.mkdir(parents=True)
    raw = make_players_raw()
    raw.loc[raw["id"] == 2, "minutes"] = 9999  # 1 of 3 players, way over 1 percent
    raw.to_csv(external / "players_raw.csv", index=False)

    with pytest.raises(ValidationError):
        validate_season(SEASON, curated_root=tmp_path / "curated", external_root=tmp_path / "external")

    report = pd.read_csv(curated / "validation_report.csv")
    assert (report["player_id"] == 2).all()


REAL_2025_26 = Path("data/curated/2025-26/player_gw.parquet")


@pytest.mark.skipif(not REAL_2025_26.exists(), reason="real 2025-26 data not curated")
def test_real_2025_26_known_dgw():
    # Arsenal double gameweek 26: David Raya (element 1) played twice.
    pgw = pd.read_parquet(REAL_2025_26)
    raya = pgw.query("player_id == 1 and gw == 26").iloc[0]
    assert raya["n_fixtures"] == 2
    assert raya["rule_regime"] == "defcon_v1"
    assert raya["source"] == "vaastav"
