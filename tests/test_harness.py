"""Evaluation harness: scoring one pick, and the paired comparison maths.

The arithmetic is checked against hand worked numbers on three synthetic
gameweeks, because a harness that silently miscounts is worse than no
harness. It would not fail, it would just quietly crown the wrong model and
send every later decision in the wrong direction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines import ep_next, last5, naive_minutes_ep, price
from eval import harness

CURATED = Path("data/curated")


# --------------------------------------------------------------------------
# Scoring one pick
# --------------------------------------------------------------------------


class _FakeSolution:
    def __init__(self, squad):
        self.squad = squad


def _squad(realised_by_player, captain_id, n_xi=11):
    """A 15 player squad where the first n_xi are the eleven."""
    rows = []
    for i, (pid, _) in enumerate(realised_by_player.items()):
        rows.append(
            {
                "player_id": pid,
                "web_name": f"P{pid}",
                "price": 5.0,
                "in_xi": i < n_xi,
                "is_captain": pid == captain_id,
                "is_vice": False,
            }
        )
    return pd.DataFrame(rows)


def test_score_pick_counts_the_captain_twice():
    """Eleven players on 2 points each, captain on 10, is 20 + 10 + 10."""
    realised = {i: 2.0 for i in range(1, 15)}
    realised[1] = 10.0
    squad = _squad(realised, captain_id=1)
    frame = pd.DataFrame(
        {"player_id": list(realised), "realised": list(realised.values())}
    )

    got = harness.score_pick(_FakeSolution(squad), frame)
    # Ten starters on 2 plus the captain on 10, then the captain again
    assert got["xi_points"] == pytest.approx(10 * 2.0 + 10.0)
    assert got["captain_points"] == pytest.approx(10.0)
    assert got["points"] == pytest.approx(30.0 + 10.0)


def test_bench_points_never_count():
    realised = {i: 0.0 for i in range(1, 15)}
    realised.update({i: 100.0 for i in range(12, 15)})  # bench only
    realised[1] = 5.0
    squad = _squad(realised, captain_id=1)
    frame = pd.DataFrame({"player_id": list(realised), "realised": list(realised.values())})

    got = harness.score_pick(_FakeSolution(squad), frame)
    assert got["points"] == pytest.approx(5.0 + 5.0)


def test_a_player_with_no_realised_row_scores_nothing():
    """A blank gameweek is a zero, never a missing value that propagates."""
    realised = {i: 1.0 for i in range(1, 15)}
    squad = _squad(realised, captain_id=1)
    frame = pd.DataFrame({"player_id": [1, 2], "realised": [4.0, 1.0]})

    got = harness.score_pick(_FakeSolution(squad), frame)
    assert np.isfinite(got["points"])
    # Only players 1 and 2 have rows, so the eleven is 4 + 1, captain adds 4
    assert got["points"] == pytest.approx(5.0 + 4.0)


def test_captain_precision_is_measured_inside_the_eleven():
    realised = {i: float(i) for i in range(1, 15)}
    # Player 11 is the best of the eleven, player 1 the worst
    squad = _squad(realised, captain_id=11)
    frame = pd.DataFrame({"player_id": list(realised), "realised": list(realised.values())})
    assert harness.score_pick(_FakeSolution(squad), frame)["captain_in_top_k"]

    poor = _squad(realised, captain_id=1)
    assert not harness.score_pick(_FakeSolution(poor), frame)["captain_in_top_k"]


# --------------------------------------------------------------------------
# Paired comparison, hand checked
# --------------------------------------------------------------------------


def _results_frame():
    """Three gameweeks, hand chosen so every summary number is checkable.

        gw   good   last5   delta
         1     60      50     +10
         2     40      50     -10
         3     70      50     +20

    good: mean 56.667, delta +6.667, one loss in three
    flat: identical to last5 every week, so all ties
    """
    rows = []
    points = {
        "good": {1: 60.0, 2: 40.0, 3: 70.0},
        "perfect": {1: 60.0, 2: 60.0, 3: 60.0},
        "last5": {1: 50.0, 2: 50.0, 3: 50.0},
        "flat": {1: 50.0, 2: 50.0, 3: 50.0},
    }
    for candidate, by_gw in points.items():
        for gw, pts in by_gw.items():
            rows.append(
                {
                    "candidate": candidate,
                    "gw": gw,
                    "points": pts,
                    "xi_points": pts,
                    "captain_points": 0.0,
                    "captain_in_top_k": candidate == "good",
                    "spearman": 0.5,
                    "squad_value": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_mean_points_and_delta_match_hand_arithmetic():
    summary = harness.summarise(_results_frame()).set_index("candidate")

    assert summary.loc["good", "mean_points"] == pytest.approx(170 / 3)
    assert summary.loc["last5", "mean_points"] == pytest.approx(50.0)
    # (10 - 10 + 20) / 3
    assert summary.loc["good", "delta_vs_reference"] == pytest.approx(20 / 3)
    assert summary.loc["last5", "delta_vs_reference"] == pytest.approx(0.0)


def test_win_loss_tie_counts_are_per_gameweek():
    summary = harness.summarise(_results_frame()).set_index("candidate")

    assert (summary.loc["good", "wins"], summary.loc["good", "losses"]) == (2, 1)
    assert summary.loc["good", "ties"] == 0
    # A candidate identical to the reference ties every week
    assert summary.loc["flat", "ties"] == 3
    assert (summary.loc["flat", "wins"], summary.loc["flat", "losses"]) == (0, 0)


def test_sign_test_p_value_matches_the_binomial_by_hand():
    summary = harness.summarise(_results_frame()).set_index("candidate")

    # Two wins from three. Outcome probabilities on a fair coin are 1/8,
    # 3/8, 3/8, 1/8, so every possible split is at least as likely as 2-1
    # and the two sided p is exactly 1. Winning two weeks out of three is
    # no evidence at all, which is the point of reporting this column.
    assert summary.loc["good", "sign_test_p"] == pytest.approx(1.0)

    # Three from three: only 3-0 and 0-3 are as extreme, so 2 * 1/8.
    assert summary.loc["perfect", "sign_test_p"] == pytest.approx(0.25)


def test_all_ties_leaves_the_sign_test_undefined_rather_than_certain():
    """Zero decisive gameweeks must not be reported as overwhelming evidence."""
    summary = harness.summarise(_results_frame()).set_index("candidate")
    assert np.isnan(summary.loc["flat", "sign_test_p"])


def test_summary_is_ordered_by_realised_points():
    summary = harness.summarise(_results_frame())
    # perfect averages 60, good averages 56.7, both references average 50
    assert summary.iloc[0]["candidate"] == "perfect"
    assert list(summary["mean_points"]) == sorted(summary["mean_points"], reverse=True)


def test_captain_precision_averages_over_gameweeks():
    summary = harness.summarise(_results_frame()).set_index("candidate")
    assert summary.loc["good", "captain_precision"] == pytest.approx(1.0)
    assert summary.loc["last5", "captain_precision"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def test_position_spearman_is_perfect_when_ranking_is_perfect():
    n = 40
    ep = pd.DataFrame(
        {"player_id": range(n), "ep_total": np.arange(n, dtype=float)}
    )
    realised = pd.DataFrame({"player_id": range(n), "realised": np.arange(n, dtype=float)})
    meta = pd.DataFrame({"player_id": range(n), "element_type": [3] * n})
    assert harness.position_spearman(ep, realised, meta) == pytest.approx(1.0)


def test_position_spearman_is_negative_when_ranking_is_reversed():
    n = 40
    ep = pd.DataFrame({"player_id": range(n), "ep_total": np.arange(n, dtype=float)})
    realised = pd.DataFrame(
        {"player_id": range(n), "realised": np.arange(n, dtype=float)[::-1]}
    )
    meta = pd.DataFrame({"player_id": range(n), "element_type": [3] * n})
    assert harness.position_spearman(ep, realised, meta) == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def _mini_frame():
    """One player, four gameweeks, scoring 2, 4, 6 then due to play again."""
    rows = []
    for gw, (minutes, goals) in enumerate([(90, 0), (90, 1), (90, 2), (90, 0)], start=1):
        rows.append(
            {
                "season": "2025-26",
                "gw": gw,
                "player_id": 1,
                "player_code": 500,
                "fixture_id": gw,
                "element_type": 3,
                "team_id": 1,
                "price": 7.0,
                "kickoff_time": pd.Timestamp("2025-08-16", tz="UTC") + pd.Timedelta(days=7 * gw),
                "minutes": minutes,
                "goals_scored": goals,
                "assists": 0,
                # Conceding keeps a clean sheet off the total, so the points
                # are just appearance plus goals and stay easy to check.
                "goals_conceded": 1,
                "saves": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "own_goals": 0,
                "bonus": 0,
                "clean_sheets": 0,
                "defensive_contribution": 0,
                "rule_regime": "defcon_v1",
                "start_rate_1": 1.0,
                "ep_next": None,
            }
        )
    return pd.DataFrame(rows)


def test_last5_averages_previous_appearances_only():
    """A midfielder scores 2, then 7, then 12. Form must lag by one match."""
    out = last5.expected_points(_mini_frame(), "2025-26").sort_values("gw")
    # gw1 has no history at all
    assert out["ep_total"].iloc[0] == pytest.approx(0.0)
    # gw2 sees only gw1: two points for playing 90 minutes
    assert out["ep_total"].iloc[1] == pytest.approx(2.0)
    # gw3 sees gw1 and gw2, which is 2 and 7, so 4.5
    assert out["ep_total"].iloc[2] == pytest.approx(4.5)


def test_price_baseline_is_just_the_price():
    out = price.expected_points(_mini_frame(), "2025-26")
    assert (out["ep_total"] == 7.0).all()


def test_ep_next_returns_empty_rather_than_faking_a_value():
    out = ep_next.expected_points(_mini_frame(), "2025-26")
    assert out.empty
    assert list(out.columns) == ep_next.EMPTY_COLUMNS


def test_naive_minutes_baseline_produces_finite_points():
    out = naive_minutes_ep.expected_points(_mini_frame(), "2025-26")
    assert len(out) == 4
    assert np.isfinite(out["ep_total"]).all()
    # Having started the previous match, a full appearance is assumed, which
    # is worth at least the two appearance points
    assert out["ep_total"].iloc[-1] >= 2.0


def test_every_baseline_produces_the_compose_schema():
    required = {"season", "gw", "player_id", "player_code", "ep_total"}
    for module in (last5, price, naive_minutes_ep):
        out = module.expected_points(_mini_frame(), "2025-26")
        assert required <= set(out.columns), module.NAME


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (CURATED / "2025-26").exists(), reason="curated data not built")
def test_harness_runs_end_to_end_on_real_gameweeks():
    candidates = {
        "last5": None,
    }
    from models import compose

    full = compose.build_full_frame("2025-26", None, "historical", CURATED)
    candidates = {
        "last5": last5.expected_points(full, "2025-26"),
        "price": price.expected_points(full, "2025-26"),
    }
    results = harness.run("2025-26", from_gw=36, candidates=candidates)

    assert set(results["candidate"]) == {"last5", "price"}
    assert results["gw"].nunique() == 3
    assert (results["points"] >= 0).all()

    summary = harness.summarise(results)
    assert summary[summary["candidate"] == "last5"]["delta_vs_reference"].iloc[0] == 0.0
