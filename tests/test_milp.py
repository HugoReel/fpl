"""Optimiser: legality, binding constraints, overrides and transfer economics.

Every test runs on a small synthetic pool where the right answer can be
worked out by hand, so a failure points at the model rather than at the
data. Real pool behaviour is covered by one slower test at the end.
"""

from pathlib import Path

import pandas as pd
import pytest

from optimise import milp

CURATED = Path("data/curated")


def make_pool(n_per_position=(4, 8, 8, 6), teams=6, base_ep=1.0, price=5.0):
    """A pool with enough players for a legal squad and predictable values.

    Expected points rise with the index inside each position, so the best
    choice is always the last few, and prices are flat unless a test changes
    them. Clubs are dealt round robin so no club is over represented.
    """
    rows = []
    pid = 1
    for et, count in zip((1, 2, 3, 4), n_per_position):
        for j in range(count):
            rows.append(
                {
                    "player_id": pid,
                    "player_code": 1000 + pid,
                    "web_name": f"{milp.POSITION_NAMES[et]}{j}",
                    "element_type": et,
                    "team_id": (pid % teams) + 1,
                    "team_short": f"T{(pid % teams) + 1}",
                    "price": price,
                    "ep_total": base_ep + j,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def solve(pool, **kwargs):
    kwargs.setdefault("budget", 1000.0)
    return milp.solve(pool, [1], **kwargs)


# --------------------------------------------------------------------------
# Legality
# --------------------------------------------------------------------------


def test_squad_has_the_legal_shape():
    sol = solve(make_pool())
    assert len(sol.squad) == milp.SQUAD_SIZE
    counts = sol.squad["element_type"].value_counts().to_dict()
    assert counts == {1: 2, 2: 5, 3: 5, 4: 3}


def test_starting_eleven_has_a_legal_formation():
    sol = solve(make_pool())
    xi = sol.squad[sol.squad["in_xi"]]
    assert len(xi) == milp.XI_SIZE
    counts = xi["element_type"].value_counts().to_dict()
    assert counts.get(1, 0) == milp.XI_EXACT_GKP
    assert counts.get(2, 0) >= milp.XI_MIN_DEF
    assert counts.get(4, 0) >= milp.XI_MIN_FWD
    # Two midfielders are forced by the squad quotas even though no
    # constraint says so, because five defenders and three forwards plus a
    # keeper is only nine.
    assert counts.get(3, 0) >= 2


def test_exactly_one_captain_and_one_vice_and_never_the_same_player():
    sol = solve(make_pool())
    assert sol.squad["is_captain"].sum() == 1
    assert sol.squad["is_vice"].sum() == 1
    assert not (sol.squad["is_captain"] & sol.squad["is_vice"]).any()
    assert sol.squad.loc[sol.squad["is_captain"], "in_xi"].all()
    assert sol.squad.loc[sol.squad["is_vice"], "in_xi"].all()


def test_captain_is_the_best_player_in_the_eleven():
    sol = solve(make_pool())
    xi = sol.squad[sol.squad["in_xi"]]
    assert xi.loc[xi["is_captain"], "ep_total"].iloc[0] == xi["ep_total"].max()


def test_vice_is_the_best_remaining_starter():
    sol = solve(make_pool())
    xi = sol.squad[sol.squad["in_xi"]]
    # Compared against the best non captain rather than by position in a
    # sorted list, because equal expected points are common and the order
    # between tied players is arbitrary.
    best_remaining = xi.loc[~xi["is_captain"], "ep_total"].max()
    assert xi.loc[xi["is_vice"], "ep_total"].iloc[0] == best_remaining


# --------------------------------------------------------------------------
# Constraints that must bind
# --------------------------------------------------------------------------


def test_club_limit_binds_when_the_best_players_share_a_club():
    """Make one club's players clearly best and check only three are taken."""
    pool = make_pool()
    pool.loc[pool["team_id"] == 1, "ep_total"] += 50
    sol = solve(pool)
    per_club = sol.squad["team_id"].value_counts()
    assert per_club.max() <= milp.MAX_PER_CLUB
    assert per_club.get(1, 0) == milp.MAX_PER_CLUB


def test_budget_binds_and_forces_cheaper_players():
    """A budget that cannot afford the best squad must change the answer."""
    pool = make_pool(price=5.0)
    pool.loc[pool["ep_total"] > 5, "price"] = 12.0

    rich = solve(pool, budget=1000.0)
    poor = solve(pool, budget=80.0)

    assert poor.squad["price"].sum() <= 80.0 + 1e-6
    assert poor.ep_xi < rich.ep_xi


def test_budget_too_small_is_reported_not_crashed():
    pool = make_pool(price=5.0)
    with pytest.raises(milp.InfeasibleError, match="cheapest legal squad"):
        solve(pool, budget=10.0)


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------


def test_ban_removes_a_player_who_would_otherwise_be_picked():
    pool = make_pool()
    best = pool.sort_values("ep_total", ascending=False).iloc[0]
    unbanned = solve(pool)
    assert best["player_code"] in set(unbanned.squad["player_code"])

    banned = solve(pool, overrides={"lock": [], "ban": [int(best["player_code"])], "force_captain": None})
    assert best["player_code"] not in set(banned.squad["player_code"])


def test_lock_forces_a_player_into_the_eleven():
    pool = make_pool()
    worst = pool[pool["element_type"] == 4].sort_values("ep_total").iloc[0]
    free = solve(pool)
    assert worst["player_code"] not in set(free.squad.loc[free.squad["in_xi"], "player_code"])

    locked = solve(
        pool, overrides={"lock": [int(worst["player_code"])], "ban": [], "force_captain": None}
    )
    row = locked.squad[locked.squad["player_code"] == worst["player_code"]]
    assert len(row) == 1 and bool(row["in_xi"].iloc[0])


def test_force_captain_overrides_the_best_player():
    pool = make_pool()
    mid = pool[pool["element_type"] == 3].sort_values("ep_total").iloc[2]
    sol = solve(
        pool,
        overrides={"lock": [int(mid["player_code"])], "ban": [], "force_captain": int(mid["player_code"])},
    )
    assert bool(sol.squad.loc[sol.squad["player_code"] == mid["player_code"], "is_captain"].iloc[0])


def test_infeasible_override_names_the_clashing_constraint():
    pool = make_pool()
    keepers = pool[pool["element_type"] == 1]["player_code"].head(2).astype(int).tolist()
    with pytest.raises(milp.InfeasibleError, match="keepers are locked"):
        solve(pool, overrides={"lock": keepers, "ban": [], "force_captain": None})


def test_banning_a_whole_position_is_explained():
    pool = make_pool()
    gk_codes = pool[pool["element_type"] == 1]["player_code"].astype(int).tolist()
    with pytest.raises(milp.InfeasibleError, match="only 1 GKP remain"):
        solve(pool, overrides={"lock": [], "ban": gk_codes[:-1], "force_captain": None})


def test_too_many_locks_from_one_club_is_explained():
    pool = make_pool()
    # Four defenders from one club, so the club limit is the first thing
    # that breaks rather than a position quota.
    defenders = pool[pool["element_type"] == 2].head(4)
    pool.loc[defenders.index, "team_id"] = 1
    pool.loc[defenders.index, "team_short"] = "T1"
    codes = defenders["player_code"].astype(int).tolist()
    with pytest.raises(milp.InfeasibleError, match="over the limit of 3 per club"):
        solve(pool, overrides={"lock": codes, "ban": [], "force_captain": None})


def test_unknown_override_code_is_explained():
    with pytest.raises(milp.InfeasibleError, match="not in the pool"):
        solve(make_pool(), overrides={"lock": [999999], "ban": [], "force_captain": None})


# --------------------------------------------------------------------------
# Transfer economics
# --------------------------------------------------------------------------


def _state_from(squad: pd.DataFrame, free_transfers=1, bank=0.0):
    return {
        "squad": [
            {"code": int(c), "purchase_price": float(p), "selling_price": float(p)}
            for c, p in zip(squad["player_code"], squad["price"])
        ],
        "bank": bank,
        "free_transfers": free_transfers,
    }


def _upgrade_pool(gain: float):
    """A settled squad plus one unowned player who is `gain` points better."""
    pool = make_pool()
    owned = solve(pool).squad
    state = _state_from(owned, free_transfers=0)

    # Improve one unowned midfielder so a transfer is at least tempting
    unowned = pool[~pool["player_code"].isin(owned["player_code"])]
    target = unowned[unowned["element_type"] == 3].sort_values("ep_total").index[-1]
    worst_owned_mid = owned[owned["element_type"] == 3]["ep_total"].min()
    pool.loc[target, "ep_total"] = worst_owned_mid + gain
    return pool, state, int(pool.loc[target, "player_code"])


def test_a_small_upgrade_does_not_justify_a_four_point_hit():
    pool, state, code = _upgrade_pool(gain=1.0)
    sol = solve(pool, state=state)
    assert code not in set(sol.squad["player_code"])
    assert sol.n_transfers == 0
    assert sol.hit_cost == 0


def test_a_large_upgrade_does_justify_a_four_point_hit():
    pool, state, code = _upgrade_pool(gain=5.0)
    sol = solve(pool, state=state)
    assert code in set(sol.squad["player_code"])
    assert sol.n_transfers == 1
    assert sol.paid_transfers == 1
    assert sol.hit_cost == milp.TRANSFER_HIT


def test_a_free_transfer_costs_nothing():
    pool, state, code = _upgrade_pool(gain=1.0)
    state["free_transfers"] = 1
    sol = solve(pool, state=state)
    assert code in set(sol.squad["player_code"])
    assert sol.n_transfers == 1
    assert sol.paid_transfers == 0
    assert sol.hit_cost == 0


def test_transfers_are_reported_both_ways():
    pool, state, code = _upgrade_pool(gain=5.0)
    sol = solve(pool, state=state)
    assert len(sol.transfers_in) == 1 and len(sol.transfers_out) == 1
    assert sol.transfers_in[0]["player_code"] == code


def test_bank_limits_what_can_be_bought():
    pool = make_pool(price=5.0)
    owned = solve(pool).squad
    state = _state_from(owned, free_transfers=1, bank=0.0)
    # An expensive upgrade that the bank cannot cover, since selling any
    # owned player only returns 5.0
    unowned = pool[~pool["player_code"].isin(owned["player_code"])]
    target = unowned[unowned["element_type"] == 3].index[-1]
    pool.loc[target, "ep_total"] = 99.0
    pool.loc[target, "price"] = 40.0

    sol = solve(pool, state=state)
    assert int(pool.loc[target, "player_code"]) not in set(sol.squad["player_code"])


# --------------------------------------------------------------------------
# Horizon and determinism
# --------------------------------------------------------------------------


def test_multi_gameweek_horizon_raises_rather_than_pretending():
    with pytest.raises(NotImplementedError, match="phase 4"):
        milp.solve(make_pool(), [1, 2], budget=1000.0)


def test_same_inputs_give_the_same_squad_twice():
    pool = make_pool()
    a = solve(pool)
    b = solve(pool)
    assert list(a.squad["player_code"]) == list(b.squad["player_code"])
    assert list(a.squad["in_xi"]) == list(b.squad["in_xi"])
    assert list(a.squad["is_captain"]) == list(b.squad["is_captain"])
    assert a.objective == pytest.approx(b.objective)


def test_objective_matches_its_stated_definition():
    """XI points, captain again, a weighted bench, minus any hit."""
    sol = solve(make_pool())
    expected = sol.ep_xi + sol.ep_captain + milp.BENCH_WEIGHT * sol.ep_bench
    # The vice tie break is deliberately tiny and must not move the total
    assert sol.objective == pytest.approx(expected, abs=1e-2)


# --------------------------------------------------------------------------
# Real pool
# --------------------------------------------------------------------------

REAL_EP = Path("data/predictions/2026-27/gw1/expected_points.parquet")


@pytest.mark.skipif(not REAL_EP.exists(), reason="expected points not generated")
def test_full_pool_solves_quickly_and_legally():
    pool = milp.load_pool("2026-27", 1)
    assert len(pool) > 500
    sol = milp.solve(pool, [1])

    assert sol.solve_seconds < 5.0
    assert len(sol.squad) == milp.SQUAD_SIZE
    assert sol.squad["element_type"].value_counts().to_dict() == {1: 2, 2: 5, 3: 5, 4: 3}
    assert sol.squad["price"].sum() <= milp.FRESH_BUDGET + 1e-6
    assert sol.squad["team_id"].value_counts().max() <= milp.MAX_PER_CLUB
