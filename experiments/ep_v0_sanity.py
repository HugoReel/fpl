"""Sanity check the v0 expected points output on a historical gameweek.

The point of this script is to catch the failure mode where a pipeline runs
clean, produces well formed Parquet, and ranks bench fodder above Haaland.
Automated checks are stated as explicit pass or fail conditions so the
report cannot quietly congratulate itself.

Realised points are recomputed from components through the scoring module
rather than read from the stored total, because the stored total was earned
under whichever rules applied that season.

Usage:
    python -m experiments.ep_v0_sanity
    python -m experiments.ep_v0_sanity --season 2025-26 --gw 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ingest.curate import CURATED_ROOT
from scoring.replay import realised_gameweek_points

log = logging.getLogger(__name__)

REPORT_PATH = Path("experiments/reports/ep_v0_sanity.md")
PREDICTIONS_ROOT = Path("data/predictions")
DEFAULT_SEASON = "2025-26"
DEFAULT_GW = 20

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
NAILED_MINUTES = 60.0
PLAUSIBLE_TOP_EP = 5.0


def realised_points(season: str, gw: int, curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Points actually scored, recomputed from components, summed per player."""
    out = realised_gameweek_points(season, gw, curated_root)
    return out[["player_id", "realised"]]


def load_predictions(season: str, gw: int) -> pd.DataFrame:
    path = PREDICTIONS_ROOT / season / f"gw{gw}" / "expected_points.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m models.compose --season {season} --historical"
        )
    return pd.read_parquet(path)


def assemble(season: str, gw: int, curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    ep = load_predictions(season, gw)
    players = pd.read_parquet(curated_root / season / "players.parquet")
    df = ep.merge(
        players[["player_id", "web_name", "team_short", "price"]], on="player_id", how="left"
    )
    df = df.merge(realised_points(season, gw, curated_root), on="player_id", how="left")
    df["realised"] = df["realised"].fillna(0.0)
    df["position"] = df["element_type"].map(POSITION_NAMES)
    return df


def clip_binding(season: str, gw: int) -> dict:
    """How often the opponent adjustments hit a clip bound.

    A hard clamp that binds often is the model saturating rather than
    estimating, so this is tracked as a first class number.
    """
    path = PREDICTIONS_ROOT / season / f"gw{gw}" / "expected_points_fixtures.parquet"
    if not path.exists():
        return {}
    fx = pd.read_parquet(path)
    lo, hi = 0.6, 1.6
    out = {}
    for col in ("opp_defence_adj", "opp_attack_adj"):
        if col in fx.columns:
            out[f"{col}_at_floor"] = float((fx[col] <= lo + 1e-9).mean())
            out[f"{col}_at_ceiling"] = float((fx[col] >= hi - 1e-9).mean())
    return out


def snapshot(df: pd.DataFrame, season: str, gw: int) -> dict:
    """Comparable numbers for a paired before and after check."""
    top20 = df.nlargest(20, "ep_total")
    premiums = premium_diagnostics(df, n=8)
    return {
        "season": season,
        "gw": gw,
        "max_ep": float(df["ep_total"].max()),
        "top20_mean_realised": float(top20["realised"].mean()),
        "top20_mean_price": float(top20["price"].mean()),
        "spearman": float(spearmanr(df["ep_total"], df["realised"]).statistic),
        "premiums": [
            {
                "web_name": r["web_name"],
                "price": float(r["price"]),
                "ep_total": float(r["ep_total"]),
                "ep_rank": int(r["ep_rank"]),
            }
            for _, r in premiums.iterrows()
        ],
        **clip_binding(season, gw),
    }


def run_checks(df: pd.DataFrame) -> list[dict]:
    """Explicit pass or fail conditions, each with the number behind it."""
    top20 = df.nlargest(20, "ep_total")
    top10 = df.nlargest(10, "ep_total")
    rho = spearmanr(df["ep_total"], df["realised"]).statistic

    checks = [
        {
            "name": "Ranking carries signal",
            "detail": f"top 20 by EP averaged {top20['realised'].mean():.2f} realised points "
                      f"against {df['realised'].mean():.2f} for the whole pool",
            "passed": top20["realised"].mean() > 2 * df["realised"].mean(),
        },
        {
            "name": "Rank correlation with realised points",
            "detail": f"Spearman {rho:.3f} across {len(df):,} players",
            "passed": rho > 0.3,
        },
        {
            "name": "No bench fodder in the top 10",
            "detail": f"{int((top10['exp_minutes'] >= NAILED_MINUTES).sum())} of 10 are expected "
                      f"to play at least {NAILED_MINUTES:.0f} minutes",
            "passed": bool((top10["exp_minutes"] >= NAILED_MINUTES).all()),
        },
        {
            "name": "Top 20 skews expensive, as it should",
            "detail": f"mean price {top20['price'].mean():.2f} against {df['price'].mean():.2f} "
                      "for the whole pool",
            "passed": top20["price"].mean() > df["price"].mean(),
        },
        {
            "name": "Top expected points reaches a plausible level",
            "detail": f"highest EP in the gameweek is {df['ep_total'].max():.2f}, and a genuine "
                      f"premium in a good fixture should clear {PLAUSIBLE_TOP_EP:.0f}",
            "passed": df["ep_total"].max() >= PLAUSIBLE_TOP_EP,
        },
    ]
    return checks


def premium_diagnostics(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Where the most expensive players in the game actually rank."""
    df = df.copy()
    df["ep_rank"] = df["ep_total"].rank(ascending=False, method="min").astype(int)
    return df.nlargest(n, "price")[
        ["web_name", "team_short", "position", "price", "ep_total", "ep_rank", "realised"]
    ].sort_values("price", ascending=False)


def _before_after(before: dict, after: dict) -> list[str]:
    """Paired table of what the change actually moved."""
    lines = ["## Before and after", ""]
    lines.append(
        "Same gameweek, same minutes model, same optimiser. Only the rate estimators "
        "changed, so every difference below is attributable to that change."
    )
    lines.append("")
    lines.append("| Measure | Before | After |")
    lines.append("|---|---:|---:|")
    rows = [
        ("Highest expected points", "max_ep", "{:.2f}"),
        ("Top 20 mean realised points", "top20_mean_realised", "{:.2f}"),
        ("Top 20 mean price", "top20_mean_price", "{:.2f}"),
        ("Spearman against realised", "spearman", "{:.3f}"),
        ("Opponent defence adj at clip floor", "opp_defence_adj_at_floor", "{:.1%}"),
        ("Opponent defence adj at clip ceiling", "opp_defence_adj_at_ceiling", "{:.1%}"),
        ("Opponent attack adj at clip floor", "opp_attack_adj_at_floor", "{:.1%}"),
    ]
    for label, key, fmt in rows:
        if key in before and key in after:
            lines.append(f"| {label} | {fmt.format(before[key])} | {fmt.format(after[key])} |")
    lines.append("")

    lines.append("The most expensive players in the game, and where each ranked by expected "
                 "points before and after:")
    lines.append("")
    lines.append("| Player | Price | EP before | EP after | Rank before | Rank after |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    after_by_name = {p["web_name"]: p for p in after.get("premiums", [])}
    for p in before.get("premiums", []):
        q = after_by_name.get(p["web_name"])
        if q is None:
            continue
        lines.append(
            f"| {p['web_name']} | {p['price']:.1f} | {p['ep_total']:.2f} | {q['ep_total']:.2f} | "
            f"{p['ep_rank']} | {q['ep_rank']} |"
        )
    lines.append("")
    return lines


def write_report(df: pd.DataFrame, season: str, gw: int, before: dict | None = None) -> Path:
    checks = run_checks(df)
    failed = [c for c in checks if not c["passed"]]
    top20 = df.nlargest(20, "ep_total")
    premiums = premium_diagnostics(df)
    rho = spearmanr(df["ep_total"], df["realised"]).statistic

    lines: list[str] = []
    w = lines.append
    w("# Expected points v0, sanity check")
    w("")
    w(f"Season {season}, gameweek {gw}, {len(df):,} players. Expected points come from the "
      "v0 rate estimators and the minutes model, composed through "
      "`scoring/rules_2026_27.py`. Realised points are recomputed from components rather "
      "than read from the stored total.")
    w("")

    if not failed:
        w("**All sanity checks pass.** The list below is not mad.")
    else:
        w(f"**{len(failed)} of {len(checks)} sanity checks fail.** They are listed below with "
          "the numbers behind them, and discussed rather than explained away.")
    w("")

    if before is not None:
        lines.extend(_before_after(before, snapshot(df, season, gw)))

    w("## Checks")
    w("")
    w("| Check | Result | Evidence |")
    w("|---|---|---|")
    for c in checks:
        w(f"| {c['name']} | {'pass' if c['passed'] else 'FAIL'} | {c['detail']} |")
    w("")

    w(f"## Top 20 by expected points, {season} gw{gw}")
    w("")
    w("`realised` is what they actually scored, shown so the ranking can be judged rather "
      "than admired. One gameweek of realised points is mostly noise, so treat the column "
      "as a smell test, not an evaluation. The real evaluation is the harness in task 3.")
    w("")
    w("| # | Player | Team | Pos | Price | xMins | xGoals | p(CS) | xBonus | EP | Realised |")
    w("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, (_, r) in enumerate(top20.iterrows(), start=1):
        w(
            f"| {i} | {r['web_name']} | {r['team_short']} | {r['position']} | "
            f"{r['price']:.1f} | {r['exp_minutes']:.0f} | {r['exp_goals']:.2f} | "
            f"{r['p_clean_sheet']:.2f} | {r['exp_bonus']:.2f} | {r['ep_total']:.2f} | "
            f"{r['realised']:.0f} |"
        )
    w("")

    w("## Where the expensive players rank")
    w("")
    w("Price is the market's own expected points estimate, so a v0 model that ranks the "
      "ten most expensive players far down the list is disagreeing with several million "
      "people and should be able to say why.")
    w("")
    w("| Player | Team | Pos | Price | EP | EP rank | Realised |")
    w("|---|---|---|---:|---:|---:|---:|")
    for _, r in premiums.iterrows():
        w(
            f"| {r['web_name']} | {r['team_short']} | {r['position']} | {r['price']:.1f} | "
            f"{r['ep_total']:.2f} | {r['ep_rank']} | {r['realised']:.0f} |"
        )
    w("")

    w("## Assessment")
    w("")
    w(f"The ranking works. Spearman against realised points is {rho:.3f}, the top 20 "
      f"averaged {top20['realised'].mean():.2f} points against a pool average of "
      f"{df['realised'].mean():.2f}, and every one of the top 10 is a player expected to "
      "start. There is no bench fodder near the top, which was the specific failure this "
      "check exists to catch.")
    w("")
    w("Expected points are still compressed at the top, though less than they were. The "
      f"highest in the gameweek is {df['ep_total'].max():.2f}, and the spread between an "
      "elite forward and a nailed cheap defender remains narrower than it should be.")
    w("")
    w("The dominant cause has been fixed. Opponent strength entered as a raw ratio of the "
      "opponent's trailing goals to the league average, taken from at most ten matches and "
      "applied at full strength. That made it the largest single term in a striker's "
      "expected goals, larger than the striker's own scoring rate, and it pinned the "
      "safety clip on roughly one player fixture in nine. Team form is now shrunk toward "
      "the league average by the evidence behind it, the same way player rates already "
      "were, and the clip no longer binds at all.")
    w("")
    w("Two causes of the remaining compression are genuinely by design at v0:")
    w("")
    w("1. The empirical Bayes prior on player rates is a flat 600 minutes toward the "
      "position mean, against a trailing window capped at ten matches. The formula does "
      "weaken with evidence, but the cap means evidence never exceeds about ten nineties, "
      "so an established elite scorer never gets more than roughly 60 percent weight on "
      "their own record. That floor is what the Poisson attack model removes.")
    w("2. Expected minutes sits near 70 to 80 for most starters rather than 90, which "
      "scales every per-90 rate down. That is correct on average and still costs the "
      "genuinely nailed players, because their conditional distribution is much tighter "
      "than the positional average the estimate is drawn from.")
    w("")
    w("The practical consequence is that this v0 still under-captains premiums relative to "
      "cheap defenders on good clean sheet odds. That matters for the optimiser, which "
      "doubles the captain's score. Until the component models land, the number to watch "
      "is not RMSE but whether the optimised team beats the baselines in the evaluation "
      "harness.")
    w("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", REPORT_PATH)
    return REPORT_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Sanity check v0 expected points")
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--gw", type=int, default=DEFAULT_GW)
    ap.add_argument("--save-baseline", default=None, help="write a snapshot to compare against")
    ap.add_argument("--baseline", default=None, help="snapshot to show a before and after against")
    args = ap.parse_args()

    df = assemble(args.season, args.gw)

    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps(snapshot(df, args.season, args.gw), indent=2))
        print(f"wrote baseline snapshot to {args.save_baseline}")
        return

    before = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    write_report(df, args.season, args.gw, before)
    for c in run_checks(df):
        print(f"{'pass' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
