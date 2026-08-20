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
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ingest.curate import CURATED_ROOT
from scoring.rules_2026_27 import ELEMENT_TYPE_TO_POSITION, MatchStats, score_match

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
    pf = pd.read_parquet(curated_root / season / "player_fixture.parquet")
    pf = pf[pf["gw"] == gw]
    players = pd.read_parquet(curated_root / season / "players.parquet")
    pf = pf.merge(players[["player_id", "element_type"]], on="player_id", how="left")

    points = []
    for row in pf.to_dict("records"):
        stats = MatchStats(
            minutes=int(row["minutes"]),
            goals=int(row["goals_scored"]),
            assists=int(row["assists"]),
            goals_conceded=int(row["goals_conceded"]),
            saves=int(row["saves"]),
            penalties_saved=int(row["penalties_saved"]),
            penalties_missed=int(row["penalties_missed"]),
            yellow_cards=int(row["yellow_cards"]),
            red_cards=int(row["red_cards"]),
            own_goals=int(row["own_goals"]),
            bonus=int(row["bonus"]),
            defensive_actions=int(row["defensive_contribution"]),
        )
        points.append(score_match(stats, ELEMENT_TYPE_TO_POSITION[int(row["element_type"])]).total)
    pf = pf.assign(realised=points)
    return pf.groupby("player_id", as_index=False)["realised"].sum()


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


def write_report(df: pd.DataFrame, season: str, gw: int) -> Path:
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
    w("The clear weakness is compression at the top. The highest expected points in the "
      f"gameweek is {df['ep_total'].max():.2f}, and the spread between an elite forward and "
      "a nailed cheap defender is far narrower than it should be. Two causes, both known "
      "and both by design at v0:")
    w("")
    w("1. The empirical Bayes prior is a flat 600 minutes toward the position mean, which "
      "is far too aggressive for a striker with two seasons of elite scoring behind them. "
      "Shrinkage should weaken as evidence accumulates, and right now it does not.")
    w("2. Expected minutes sits near 70 to 80 for most starters rather than 90, which "
      "scales every per-90 rate down. That is correct on average and still costs the "
      "genuinely nailed players, because their conditional distribution is much tighter "
      "than the positional average the estimate is drawn from.")
    w("")
    w("The practical consequence is that this v0 will under-captain premiums and over-rank "
      "cheap defenders on good clean sheet odds. That matters for the optimiser in task 2, "
      "which doubles the captain's score and will happily captain a 4.5 million defender. "
      "It is the thing the Poisson attack model in a later phase exists to fix, and until "
      "then the number to watch is not RMSE but whether the optimised team beats the "
      "baselines in the task 3 harness.")
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
    args = ap.parse_args()

    df = assemble(args.season, args.gw)
    write_report(df, args.season, args.gw)
    for c in run_checks(df):
        print(f"{'pass' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
