"""Single gameweek squad optimiser: the best legal 15, XI, captain and vice.

Solved with HiGHS through highspy directly. Its modelling layer turned out
to be perfectly usable, so the PuLP fallback the plan allowed for is not
needed and nothing here depends on PuLP.

The model is built in one function that takes a horizon list. Only a horizon
of length one is supported today, and a longer one raises rather than
quietly optimising the wrong thing. The shape is deliberate: multi gameweek
planning, chips and banked free transfer state all attach to that horizon,
and building this around a single scalar gameweek would have to be undone.

Two modes:

  fresh     no existing squad, a 100.0m budget, pick the best 15. This is
            the preseason and wildcard case, and it is also how models get
            compared, since every candidate faces the same clean decision.
  transfer  an existing squad from team_state.yaml, with a bank and banked
            free transfers. Transfers beyond the free allowance cost 4
            points each, priced inside the objective so the solver decides
            whether a move pays for itself.

Selling prices are supplied by the human in team_state.yaml. The 50 percent
sell-on rule is deliberately not computed here, but purchase_price is
carried through so a later phase can.

Usage:
    python -m optimise.milp --season 2026-27 --gw 1
    python -m optimise.milp --season 2026-27 --gw 1 --mode transfer
    python -m optimise.milp --season 2025-26 --gw 20 --overrides overrides.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import highspy
import pandas as pd

from ingest.curate import CURATED_ROOT
from optimise import overrides as overrides_mod
from optimise.overrides import OVERRIDES_PATH, load_overrides

log = logging.getLogger(__name__)

PREDICTIONS_ROOT = Path("data/predictions")
DECISIONS_ROOT = Path("data/decisions")
TEAM_STATE_PATH = Path("team_state.yaml")

SQUAD_SIZE = 15
XI_SIZE = 11
FRESH_BUDGET = 100.0
MAX_PER_CLUB = 3
TRANSFER_HIT = 4.0
MAX_FREE_TRANSFERS = 5

# Squad composition, by FPL element_type.
POSITION_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Starting eleven shape. Exactly one keeper, at least three defenders and at
# least one forward. A minimum of two midfielders is not stated because it
# is already implied: five defenders and three forwards is the most the
# squad can supply, so with one keeper the eleventh place forces two
# midfielders regardless.
XI_EXACT_GKP = 1
XI_MIN_DEF = 3
XI_MIN_FWD = 1

# Crude placeholder for the value of a bench that mostly does not play.
# Phase 4 replaces this with real autosub probabilities, at which point a
# bench player's worth becomes P(a starter blanks) times their own points
# rather than a flat fraction of everyone's.
BENCH_WEIGHT = 0.1

# The objective has no vice captain term, because a vice only pays out when
# the captain does not play and that is an autosub question phase 4 owns.
# Left alone the solver picks any eligible player, which produces a correct
# but baffling armband. This weight is small enough that it cannot change
# the squad, the eleven or the captain, and large enough to break the tie
# toward the best remaining starter.
VICE_TIEBREAK = 1e-4

# HiGHS is deterministic for a fixed model, and the model is built in a
# fixed player order, so repeated runs return the same squad.
SOLVER_SEED = 20262027


class InfeasibleError(RuntimeError):
    """Raised when no legal squad exists, with the reason attached."""


@dataclass
class Solution:
    squad: pd.DataFrame
    objective: float
    ep_xi: float
    ep_captain: float
    ep_bench: float
    n_transfers: int = 0
    paid_transfers: int = 0
    hit_cost: float = 0.0
    transfers_in: list = field(default_factory=list)
    transfers_out: list = field(default_factory=list)
    spend: float = 0.0
    solve_seconds: float = 0.0
    mode: str = "fresh"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_pool(
    season: str,
    gw: int,
    curated_root: Path = CURATED_ROOT,
    predictions_root: Path = PREDICTIONS_ROOT,
    owned_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Every selectable player with a price and an expected points value.

    A player with no fixture in the gameweek still belongs in the pool. They
    score nothing, but they can be kept, sold or benched, and removing them
    would make an existing squad unrepresentable.
    """
    ep_path = predictions_root / season / f"gw{gw}" / "expected_points.parquet"
    if not ep_path.exists():
        raise FileNotFoundError(
            f"{ep_path} not found. Run: python -m models.compose --season {season} --gw {gw}"
        )
    ep = pd.read_parquet(ep_path)[["player_id", "player_code", "element_type", "ep_total"]]
    return pool_from_ep(ep, season, gw, curated_root, owned_codes)


def pool_from_ep(
    ep: pd.DataFrame,
    season: str,
    gw: int,
    curated_root: Path = CURATED_ROOT,
    owned_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Build a selectable pool from any expected points frame.

    Taking the frame rather than a path is what lets the evaluation harness
    put every candidate model through the identical optimiser, which is the
    only way a comparison between them means anything.
    """
    players = pd.read_parquet(curated_root / season / "players.parquet")
    meta = players[
        ["player_id", "player_code", "element_type", "web_name", "team_id", "team_short", "price"]
    ]

    pool = meta.merge(
        ep[["player_code", "ep_total"]].drop_duplicates("player_code"),
        on="player_code",
        how="left",
        validate="one_to_one",
    )
    pool["ep_total"] = pool["ep_total"].fillna(0.0)

    price = _gameweek_prices(season, gw, curated_root)
    if price is not None:
        pool = pool.merge(price, on="player_id", how="left", suffixes=("", "_gw"))
        pool["price"] = pool["price_gw"].fillna(pool["price"])
        pool = pool.drop(columns=["price_gw"])

    if owned_codes:
        missing = set(owned_codes) - set(pool["player_code"])
        if missing:
            raise ValueError(
                f"squad contains player codes absent from {season}: {sorted(missing)}. "
                "Check team_state.yaml against this season's player list."
            )

    pool = pool.dropna(subset=["price"])
    return pool.sort_values("player_id", kind="stable").reset_index(drop=True)


def _gameweek_prices(season: str, gw: int, curated_root: Path) -> pd.DataFrame | None:
    """Price as it stood in that gameweek, which is what a replay must use."""
    path = curated_root / season / "player_gw.parquet"
    if not path.exists():
        return None
    pgw = pd.read_parquet(path)
    rows = pgw[pgw["gw"] == gw]
    if rows.empty or "price" not in rows.columns:
        return None
    return rows[["player_id", "price"]].drop_duplicates("player_id")


def load_team_state(path: Path = TEAM_STATE_PATH) -> dict:
    """Existing squad, bank and banked free transfers."""
    import yaml

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy team_state.example.yaml to {path} and fill it in, "
            "or use --mode fresh."
        )
    state = yaml.safe_load(path.read_text()) or {}
    squad = state.get("squad") or []
    if len(squad) != SQUAD_SIZE:
        raise ValueError(f"{path} lists {len(squad)} players, a squad is {SQUAD_SIZE}")

    ft = int(state.get("free_transfers", 1))
    if not 1 <= ft <= MAX_FREE_TRANSFERS:
        raise ValueError(f"free_transfers must be between 1 and {MAX_FREE_TRANSFERS}, got {ft}")

    for entry in squad:
        if "code" not in entry:
            raise ValueError(f"every squad entry needs a code, got {entry}")
        entry.setdefault("purchase_price", entry.get("selling_price"))
        if entry.get("selling_price") is None:
            raise ValueError(f"squad entry {entry['code']} needs a selling_price")
    return {"squad": squad, "bank": float(state.get("bank", 0.0)), "free_transfers": ft}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_and_solve(
    pool: pd.DataFrame,
    horizon: list[int],
    state: dict | None = None,
    overrides: dict | None = None,
    budget: float = FRESH_BUDGET,
) -> Solution:
    """Construct and solve the squad model over a horizon of gameweeks.

    Only a single gameweek horizon is supported. The argument is a list
    because multi gameweek planning is the next thing that lands here, and
    it needs per gameweek transfer, chip and squad variables rather than the
    single set built below.
    """
    if len(horizon) != 1:
        raise NotImplementedError(
            f"horizon {horizon} has {len(horizon)} gameweeks. Only single gameweek "
            "optimisation exists so far. Multi gameweek planning is phase 4."
        )
    overrides = overrides or dict(overrides_mod.EMPTY)

    n = len(pool)
    ep = pool["ep_total"].astype(float).tolist()
    price = pool["price"].astype(float).tolist()
    element_type = pool["element_type"].astype(int).tolist()
    team = pool["team_id"].astype(int).tolist()
    codes = pool["player_code"].astype(int).tolist()
    index_of = {c: i for i, c in enumerate(codes)}

    owned = [False] * n
    selling = [0.0] * n
    free_transfers = 0
    bank = 0.0
    if state is not None:
        free_transfers = state["free_transfers"]
        bank = state["bank"]
        for entry in state["squad"]:
            i = index_of[int(entry["code"])]
            owned[i] = True
            selling[i] = float(entry["selling_price"])

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("random_seed", SOLVER_SEED)

    squad = h.addBinaries(n)
    xi = h.addBinaries(n)
    captain = h.addBinaries(n)
    vice = h.addBinaries(n)

    h.addConstr(sum(squad) == SQUAD_SIZE)
    for et, quota in POSITION_QUOTA.items():
        members = [squad[i] for i in range(n) if element_type[i] == et]
        h.addConstr(sum(members) == quota)

    for club in sorted(set(team)):
        members = [squad[i] for i in range(n) if team[i] == club]
        h.addConstr(sum(members) <= MAX_PER_CLUB)

    h.addConstr(sum(xi) == XI_SIZE)
    for i in range(n):
        h.addConstr(xi[i] <= squad[i])
        h.addConstr(captain[i] <= xi[i])
        h.addConstr(vice[i] <= xi[i])
        h.addConstr(captain[i] + vice[i] <= 1)
    h.addConstr(sum(captain) == 1)
    h.addConstr(sum(vice) == 1)

    h.addConstr(sum(xi[i] for i in range(n) if element_type[i] == 1) == XI_EXACT_GKP)
    h.addConstr(sum(xi[i] for i in range(n) if element_type[i] == 2) >= XI_MIN_DEF)
    h.addConstr(sum(xi[i] for i in range(n) if element_type[i] == 4) >= XI_MIN_FWD)

    paid = None
    if state is None:
        h.addConstr(sum(price[i] * squad[i] for i in range(n)) <= budget)
    else:
        # Money only moves for players entering or leaving. What is kept is
        # already paid for, whatever it is worth now.
        bought = [squad[i] for i in range(n) if not owned[i]]
        buy_cost = sum(price[i] * squad[i] for i in range(n) if not owned[i])
        sale_income = sum(selling[i] * (1 - squad[i]) for i in range(n) if owned[i])
        h.addConstr(buy_cost <= bank + sale_income)

        paid = h.addVariable(lb=0, ub=SQUAD_SIZE)
        h.addConstr(paid >= sum(bought) - free_transfers)

    # Checked before use so an unknown code becomes an explained failure
    # rather than a KeyError from deep inside model construction.
    referenced = [int(c) for c in (overrides.get("lock") or [])]
    referenced += [int(c) for c in (overrides.get("ban") or [])]
    if overrides.get("force_captain"):
        referenced.append(int(overrides["force_captain"]))
    unknown = [c for c in referenced if c not in index_of]
    if unknown:
        raise InfeasibleError(f"unknown override player codes: {sorted(set(unknown))}")

    for code in overrides.get("lock") or []:
        h.addConstr(xi[index_of[int(code)]] == 1)
    for code in overrides.get("ban") or []:
        h.addConstr(squad[index_of[int(code)]] == 0)
    if overrides.get("force_captain"):
        h.addConstr(captain[index_of[int(overrides["force_captain"])]] == 1)

    objective = sum(
        ep[i] * (xi[i] + captain[i] + BENCH_WEIGHT * (squad[i] - xi[i]) + VICE_TIEBREAK * vice[i])
        for i in range(n)
    )
    if paid is not None:
        objective = objective - TRANSFER_HIT * paid

    started = time.perf_counter()
    h.maximize(objective)
    elapsed = time.perf_counter() - started

    status = h.getModelStatus().name
    if status != "kOptimal":
        raise InfeasibleError(status)

    return _extract(h, pool, squad, xi, captain, vice, owned, selling, elapsed, state)


def _extract(h, pool, squad, xi, captain, vice, owned, selling, elapsed, state) -> Solution:
    picked = [v > 0.5 for v in h.vals(squad)]
    starting = [v > 0.5 for v in h.vals(xi)]
    is_cap = [v > 0.5 for v in h.vals(captain)]
    is_vice = [v > 0.5 for v in h.vals(vice)]

    out = pool.copy()
    out["in_squad"] = picked
    out["in_xi"] = starting
    out["is_captain"] = is_cap
    out["is_vice"] = is_vice
    out["position"] = out["element_type"].map(POSITION_NAMES)
    chosen = out[out["in_squad"]].copy()

    ep_xi = float(chosen.loc[chosen["in_xi"], "ep_total"].sum())
    ep_cap = float(chosen.loc[chosen["is_captain"], "ep_total"].sum())
    ep_bench = float(chosen.loc[~chosen["in_xi"], "ep_total"].sum())

    transfers_in: list = []
    transfers_out: list = []
    spend = float(chosen["price"].sum())
    n_transfers = 0
    if state is not None:
        owned_codes = {int(e["code"]) for e in state["squad"]}
        new_codes = set(chosen["player_code"].astype(int))
        in_codes = new_codes - owned_codes
        out_codes = owned_codes - new_codes
        transfers_in = pool[pool["player_code"].isin(in_codes)][
            ["player_code", "web_name", "team_short", "price", "ep_total"]
        ].to_dict("records")
        by_code = {int(c): i for i, c in enumerate(pool["player_code"].astype(int))}
        transfers_out = [
            {
                "player_code": c,
                "web_name": pool.iloc[by_code[c]]["web_name"],
                "team_short": pool.iloc[by_code[c]]["team_short"],
                "selling_price": selling[by_code[c]],
                "ep_total": float(pool.iloc[by_code[c]]["ep_total"]),
            }
            for c in sorted(out_codes)
        ]
        n_transfers = len(in_codes)

    paid_transfers = max(0, n_transfers - state["free_transfers"]) if state else 0
    hit = TRANSFER_HIT * paid_transfers

    chosen = chosen.sort_values(
        ["in_xi", "element_type", "ep_total"], ascending=[False, True, False]
    ).reset_index(drop=True)

    return Solution(
        squad=chosen,
        objective=float(h.getObjectiveValue()),
        ep_xi=ep_xi,
        ep_captain=ep_cap,
        ep_bench=ep_bench,
        n_transfers=n_transfers,
        paid_transfers=paid_transfers,
        hit_cost=hit,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        spend=spend,
        solve_seconds=elapsed,
        mode="transfer" if state else "fresh",
    )


# --------------------------------------------------------------------------
# Infeasibility, explained rather than announced
# --------------------------------------------------------------------------


def explain_infeasibility(
    pool: pd.DataFrame,
    horizon: list[int],
    state: dict | None,
    overrides: dict,
    budget: float,
) -> str:
    """Work out which constraint actually made the problem impossible.

    Cheap structural checks first, because they can name the exact clash.
    Failing that, the overrides are removed one group at a time and the
    model re-solved, which identifies the culprit empirically.
    """
    reasons: list[str] = []
    structural = overrides_mod.structural_conflict(
        pool,
        overrides,
        quota=POSITION_QUOTA,
        names=POSITION_NAMES,
        xi_size=XI_SIZE,
        xi_exact_gkp=XI_EXACT_GKP,
        max_per_club=MAX_PER_CLUB,
        budget=budget if state is None else None,
    )
    if structural:
        return structural

    # Counting could not prove a clash, so isolate it empirically instead.
    empty = dict(overrides_mod.EMPTY)
    try:
        build_and_solve(pool, horizon, state, empty, budget)
    except InfeasibleError:
        if state is None:
            cheapest = (
                pool.sort_values("price").groupby("element_type").head(5)["price"].sum()
            )
            return (
                "the problem is infeasible before any override is applied. The cheapest "
                f"legal squad costs about {cheapest:.1f}m against a budget of {budget:.1f}m"
            )
        return (
            "the problem is infeasible before any override is applied, so the existing "
            "squad, bank or club limits cannot produce a legal team"
        )

    for name in ("lock", "ban", "force_captain"):
        if not overrides.get(name):
            continue
        single = dict(empty)
        single[name] = overrides[name]
        try:
            build_and_solve(pool, horizon, state, single, budget)
        except InfeasibleError:
            return f"the {name} override alone makes the problem infeasible: {overrides[name]}"
        reasons.append(name)

    return (
        "no single override is infeasible on its own, so the combination of "
        f"{' and '.join(reasons)} is what clashes"
    )


def solve(
    pool: pd.DataFrame,
    horizon: list[int],
    state: dict | None = None,
    overrides: dict | None = None,
    budget: float = FRESH_BUDGET,
) -> Solution:
    """Solve, and on failure raise with the reason rather than the status code."""
    overrides = overrides or dict(overrides_mod.EMPTY)
    try:
        return build_and_solve(pool, horizon, state, overrides, budget)
    except InfeasibleError as exc:
        reason = explain_infeasibility(pool, horizon, state, overrides, budget)
        raise InfeasibleError(f"no legal squad exists: {reason}") from exc


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def format_squad(solution: Solution) -> str:
    rows = []
    for _, r in solution.squad.iterrows():
        marker = "C" if r["is_captain"] else ("V" if r["is_vice"] else "")
        rows.append(
            {
                "": "XI" if r["in_xi"] else "bench",
                "player": r["web_name"],
                "pos": r["position"],
                "team": r["team_short"],
                "price": f"{r['price']:.1f}",
                "EP": f"{r['ep_total']:.2f}",
                "C/V": marker,
            }
        )
    table = pd.DataFrame(rows).to_string(index=False)

    lines = [table, ""]
    lines.append(
        f"squad value {solution.spend:.1f}m | XI EP {solution.ep_xi:.2f} "
        f"| captain +{solution.ep_captain:.2f} | bench {solution.ep_bench:.2f} "
        f"(weighted {BENCH_WEIGHT})"
    )
    if solution.mode == "transfer":
        for t in solution.transfers_out:
            lines.append(f"  OUT {t['web_name']:<16} {t['team_short']:<4} {t['selling_price']:.1f}m")
        for t in solution.transfers_in:
            lines.append(f"  IN  {t['web_name']:<16} {t['team_short']:<4} {t['price']:.1f}m")
        lines.append(
            f"transfers {solution.n_transfers}, paid {solution.paid_transfers}, "
            f"hit {-solution.hit_cost:.0f} points"
        )
    lines.append(f"objective {solution.objective:.3f} | solved in {solution.solve_seconds:.2f}s")
    return "\n".join(lines)


def save_team_state(solution: Solution, path: Path, bank: float = 0.0, free_transfers: int = 1) -> Path:
    """Write a solved squad out as team_state.yaml.

    Useful preseason: solve fresh, keep the answer as the starting squad,
    then run transfer mode against it from the next gameweek. Selling price
    starts equal to purchase price, which is true until prices move.
    """
    import yaml

    squad = [
        {
            "code": int(r["player_code"]),
            "purchase_price": float(r["price"]),
            "selling_price": float(r["price"]),
        }
        for _, r in solution.squad.iterrows()
    ]
    payload = {"bank": float(bank), "free_transfers": int(free_transfers), "squad": squad}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def write_decision(solution: Solution, season: str, gw: int) -> Path:
    out_dir = DECISIONS_ROOT / season / f"gw{gw}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "squad.json"
    payload = {
        "season": season,
        "gw": gw,
        "mode": solution.mode,
        "objective": solution.objective,
        "ep_xi": solution.ep_xi,
        "ep_captain": solution.ep_captain,
        "ep_bench": solution.ep_bench,
        "bench_weight": BENCH_WEIGHT,
        "squad_value": solution.spend,
        "n_transfers": solution.n_transfers,
        "paid_transfers": solution.paid_transfers,
        "hit_cost": solution.hit_cost,
        "transfers_in": solution.transfers_in,
        "transfers_out": solution.transfers_out,
        "solve_seconds": solution.solve_seconds,
        "squad": solution.squad[
            [
                "player_id",
                "player_code",
                "web_name",
                "position",
                "team_short",
                "price",
                "ep_total",
                "in_xi",
                "is_captain",
                "is_vice",
            ]
        ].to_dict("records"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Optimise an FPL squad for one gameweek")
    ap.add_argument("--season", required=True)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--mode", choices=["fresh", "transfer"], default="fresh")
    ap.add_argument("--budget", type=float, default=FRESH_BUDGET)
    ap.add_argument("--state", default=str(TEAM_STATE_PATH))
    ap.add_argument("--overrides", default=str(OVERRIDES_PATH))
    ap.add_argument(
        "--save-state",
        default=None,
        help="write the solved squad out as a team_state.yaml to build on",
    )
    args = ap.parse_args()

    state = load_team_state(Path(args.state)) if args.mode == "transfer" else None
    overrides = load_overrides(Path(args.overrides))
    owned = [int(e["code"]) for e in state["squad"]] if state else None

    pool = load_pool(args.season, args.gw, owned_codes=owned)
    log.info("pool: %d players for %s gw%d", len(pool), args.season, args.gw)

    try:
        solution = solve(pool, [args.gw], state, overrides, args.budget)
    except InfeasibleError as exc:
        # An impossible override is a user mistake, not a crash, so it gets
        # a sentence naming the clash rather than a stack trace.
        raise SystemExit(f"\n{exc}\n\nRelax an override in {args.overrides} and try again.")

    path = write_decision(solution, args.season, args.gw)

    print()
    print(format_squad(solution))
    print(f"\nwrote {path}")

    if args.save_state:
        saved = save_team_state(
            solution, Path(args.save_state), bank=args.budget - solution.spend
        )
        print(f"wrote {saved}")


if __name__ == "__main__":
    main()
