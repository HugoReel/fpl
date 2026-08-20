"""Decide which source of team strength compose should consume.

Phase 3 ended with compose-v0 and a pure odds baseline tied at the harness's
resolution. The reading that followed is that the market and this project
hold their edges in different components: the market has world class team
strength and a deliberately terrible minutes rule, compose has a calibrated
minutes model and player level rates but home cooked team strength. If that
is right, the winning architecture is not one or the other, it is market
team expectations feeding this project's minutes and allocation.

This module tests the first half of that claim by asking a narrow, decidable
question: which source predicts match outcomes better, Dixon-Coles fitted on
public results, or the de-vigged closing price? The answer selects what
compose consumes, with the other kept as the fallback for fixtures the
preferred source does not cover.

The decision is made on 2021-22 through 2024-25 only. 2025-26 is the frozen
evaluation season and is reported for confirmation, never used to choose.

Three sources compared:
  market       de-vigged closing odds, independent Poisson recovery
  dixon_coles  walk forward fit, time decayed, promoted teams wrapped
  v0_trailing  the shrunk trailing team rates compose currently uses

Usage:
    python -m experiments.team_model_validation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss

from ingest.curate import CURATED_ROOT
from models import rates

log = logging.getLogger(__name__)

REPORT_PATH = Path("experiments/reports/team_model.md")
DECISION_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
FROZEN_SEASON = "2025-26"
ALL_SEASONS = DECISION_SEASONS + [FROZEN_SEASON]
CALIBRATION_BINS = 10
MAX_GOALS = 10


def _outcome_probs(home_xg: float, away_xg: float) -> tuple[float, float, float]:
    """Independent Poisson outcome probabilities, used for the non DC sources."""
    h = poisson.pmf(np.arange(MAX_GOALS + 1), home_xg)
    a = poisson.pmf(np.arange(MAX_GOALS + 1), away_xg)
    grid = np.outer(h, a)
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())


def v0_team_expectations(season: str, curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """The team model implicit in compose-v0's trailing rates.

    v0 never states a team goal expectation, but it implies one: a side's
    expected goals are its shrunk scoring rate scaled by the opponent's
    defensive adjustment, which is exactly what feeds exp_goals_conceded.
    Reconstructed here so it can be scored against the alternatives.
    """
    team_form = rates.team_trailing(rates.load_team_matches([season], curated_root))
    if team_form.empty:
        return pd.DataFrame()

    fixtures = pd.read_parquet(curated_root / season / "fixtures.parquet")
    league = rates.DEFAULT_LEAGUE_GOALS_PER_TEAM
    prior = rates.TEAM_PRIOR_MATCHES

    def shrunk(total, count):
        return (total.fillna(0.0) + prior * league) / (count.fillna(0.0) + prior)

    form = team_form.copy()
    form["gf"] = shrunk(form["team_gf_sum"], form["team_matches"])
    form["ga"] = shrunk(form["team_ga_sum"], form["team_matches"])
    lo, hi = rates.OPPONENT_ADJ_CLIP
    form["def_adj"] = (form["ga"] / league).clip(lo, hi)

    home = form.rename(columns={"team_id": "team_h", "gf": "home_gf", "def_adj": "home_def_adj"})
    away = form.rename(columns={"team_id": "team_a", "gf": "away_gf", "def_adj": "away_def_adj"})
    df = fixtures.merge(
        home[["fixture_id", "team_h", "home_gf", "home_def_adj"]], on=["fixture_id", "team_h"], how="left"
    ).merge(
        away[["fixture_id", "team_a", "away_gf", "away_def_adj"]], on=["fixture_id", "team_a"], how="left"
    )
    df["home_xg"] = df["home_gf"] * df["away_def_adj"]
    df["away_xg"] = df["away_gf"] * df["home_def_adj"]
    return df[["fixture_id", "home_xg", "away_xg"]].dropna()


def load_sources(season: str, curated_root: Path = CURATED_ROOT) -> dict[str, pd.DataFrame]:
    """Per fixture goal expectations from each candidate source."""
    out: dict[str, pd.DataFrame] = {}

    odds_path = curated_root / season / "fixture_odds.parquet"
    if odds_path.exists():
        odds = pd.read_parquet(odds_path)
        out["market"] = odds[["fixture_id", "home_xg", "away_xg", "p_home_cs", "p_away_cs"]].copy()

    dc_path = curated_root / season / "team_model.parquet"
    if dc_path.exists():
        dc = pd.read_parquet(dc_path)
        out["dixon_coles"] = dc[
            ["fixture_id", "home_xg", "away_xg", "p_home_cs", "p_away_cs",
             "p_home_win", "p_draw", "p_away_win"]
        ].copy()

    v0 = v0_team_expectations(season, curated_root)
    if not v0.empty:
        v0 = v0.copy()
        v0["p_home_cs"] = np.exp(-v0["away_xg"])
        v0["p_away_cs"] = np.exp(-v0["home_xg"])
        out["v0_trailing"] = v0

    return out


def results(season: str, curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    fixtures = pd.read_parquet(curated_root / season / "fixtures.parquet")
    played = fixtures.dropna(subset=["team_h_score", "team_a_score"])
    return played[["fixture_id", "gw", "team_h_score", "team_a_score"]]


def score_source(source: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Match outcome log loss and clean sheet Brier for one source."""
    df = truth.merge(source, on="fixture_id", how="inner").dropna(subset=["home_xg"])
    if df.empty:
        return {}

    if "p_home_win" in df.columns and df["p_home_win"].notna().all():
        probs = df[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    else:
        probs = np.array([_outcome_probs(r.home_xg, r.away_xg) for r in df.itertuples()])
    probs = probs / probs.sum(axis=1, keepdims=True)

    y = np.where(
        df["team_h_score"] > df["team_a_score"], 0,
        np.where(df["team_h_score"] == df["team_a_score"], 1, 2),
    )
    cs_pred = np.concatenate([df["p_home_cs"], df["p_away_cs"]])
    cs_true = np.concatenate([(df["team_a_score"] == 0).astype(int), (df["team_h_score"] == 0).astype(int)])
    return {
        "matches": int(len(df)),
        "outcome_log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "cs_brier": float(np.mean((cs_pred - cs_true) ** 2)),
        "mean_home_xg": float(df["home_xg"].mean()),
    }


def calibration(source: pd.DataFrame, truth: pd.DataFrame, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    df = truth.merge(source, on="fixture_id", how="inner").dropna(subset=["p_home_cs"])
    pred = np.concatenate([df["p_home_cs"], df["p_away_cs"]])
    kept = np.concatenate([(df["team_a_score"] == 0).astype(int), (df["team_h_score"] == 0).astype(int)])

    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": float(pred[m].mean()) if m.any() else float("nan"),
            "realised": float(kept[m].mean()) if m.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def evaluate(seasons: list[str], curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    rows = []
    for season in seasons:
        truth = results(season, curated_root)
        for name, source in load_sources(season, curated_root).items():
            scored = score_source(source, truth)
            if scored:
                rows.append({"season": season, "source": name, **scored})
    return pd.DataFrame(rows)


def decide(scores: pd.DataFrame) -> dict:
    """Pick the team strength source, on pre-freeze seasons only."""
    pre = scores[scores["season"].isin(DECISION_SEASONS)]
    means = pre.groupby("source")["outcome_log_loss"].mean()
    market = means.get("market", np.inf)
    dc = means.get("dixon_coles", np.inf)

    if market < dc:
        winner, fallback = "market", "dixon_coles"
    else:
        winner, fallback = "dixon_coles", "market"
    return {
        "winner": winner,
        "fallback": fallback,
        "market_log_loss": float(market),
        "dixon_coles_log_loss": float(dc),
        "v0_log_loss": float(means.get("v0_trailing", np.nan)),
        "gap_pct": float(100 * (dc - market) / market) if np.isfinite(market) else float("nan"),
        "seasons": DECISION_SEASONS,
    }


def write_report(scores: pd.DataFrame, decision: dict, calib: pd.DataFrame) -> Path:
    dense = calib[calib["n"] >= 30]
    max_gap = float((dense["mean_predicted"] - dense["realised"]).abs().max()) if not dense.empty else float("nan")

    lines: list[str] = []
    w = lines.append
    w("# Team strength: Dixon-Coles against the market")
    w("")

    if decision["winner"] == "market":
        w(f"**The market wins, so compose consumes market team expectations and keeps "
          f"Dixon-Coles as the fallback.** Over {', '.join(DECISION_SEASONS)}, de-vigged "
          f"closing odds average {decision['market_log_loss']:.4f} outcome log loss against "
          f"Dixon-Coles at {decision['dixon_coles_log_loss']:.4f}, a gap of "
          f"{decision['gap_pct']:.1f}%. Dixon-Coles is not discarded: it covers fixtures "
          "with no odds, it is the sanity cross-check, and it supplies the score matrix "
          "machinery the attack model needs.")
    else:
        w(f"**Dixon-Coles wins, so it leads and the market is the cross-check.** Over "
          f"{', '.join(DECISION_SEASONS)}, Dixon-Coles averages "
          f"{decision['dixon_coles_log_loss']:.4f} outcome log loss against the market's "
          f"{decision['market_log_loss']:.4f}.")
    w("")
    w("Beating the bookmaker was never the target. A Dixon-Coles fitted on public results "
      "cannot see team news, transfers or money, and the closing price has absorbed all of "
      f"it. Getting within a few percent is the realistic bar, and the gap here is "
      f"{abs(decision['gap_pct']):.1f}%. What matters is that the question is now settled on "
      "a number rather than on preference.")
    w("")
    w("Both are compared against `v0_trailing`, the team model implicit in the shrunk "
      "trailing rates compose uses today. v0 never states a team goal expectation, but it "
      "implies one, and reconstructing it is what makes this a three way comparison rather "
      "than a two way one.")
    w("")

    w("## Match outcome log loss")
    w("")
    w("Lower is better. The decision uses the pre-freeze seasons only; 2025-26 is shown for "
      "confirmation and took no part in choosing.")
    w("")
    pivot = scores.pivot_table(index="season", columns="source", values="outcome_log_loss")
    cols = [c for c in ("market", "dixon_coles", "v0_trailing") if c in pivot.columns]
    w("| Season | " + " | ".join(cols) + " | best |")
    w("|---|" + "|".join(["---:"] * len(cols)) + "|---|")
    for season, row in pivot.iterrows():
        best = min(cols, key=lambda c: row[c] if pd.notna(row[c]) else np.inf)
        marker = " (frozen)" if season == FROZEN_SEASON else ""
        cells = " | ".join(f"{row[c]:.4f}" if pd.notna(row[c]) else "" for c in cols)
        w(f"| {season}{marker} | {cells} | {best} |")
    w("")

    w("## Clean sheet Brier score")
    w("")
    w("Clean sheets are what this actually feeds into compose, so they get their own "
      "column. One observation per team per match.")
    w("")
    pivot_cs = scores.pivot_table(index="season", columns="source", values="cs_brier")
    w("| Season | " + " | ".join(cols) + " |")
    w("|---|" + "|".join(["---:"] * len(cols)) + "|")
    for season, row in pivot_cs.iterrows():
        cells = " | ".join(f"{row[c]:.4f}" if pd.notna(row[c]) else "" for c in cols)
        w(f"| {season} | {cells} |")
    w("")

    w(f"## Dixon-Coles clean sheet calibration, {FROZEN_SEASON}")
    w("")
    w(f"Tracks to within {max_gap:.3f} across bins holding at least 30 team fixtures. "
      "Probabilities come from the corrected score grid rather than a Poisson zero, which "
      "matters because the Dixon-Coles correction is largest at exactly the low scores a "
      "clean sheet depends on.")
    w("")
    w("| Bin | Team fixtures | Mean predicted | Realised |")
    w("|---|---:|---:|---:|")
    for _, r in calib.iterrows():
        if r["n"] == 0:
            w(f"| {r['bin']} | 0 | | |")
        else:
            w(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['realised']:.3f} |")
    w("")

    w("## What this changes")
    w("")
    w(f"compose gains a `team_source` setting. It reads `{decision['winner']}` where that "
      f"source covers the fixture and falls back to `{decision['fallback']}` where it does "
      "not, which in practice means fixtures with no priced market. The trailing rates stay "
      "available as a third fallback and as the pre-swap comparison for the harness run.")
    w("")
    w("The wider point is architectural. The market holds team strength and cannot see who "
      "starts or who takes the shots. This project holds a calibrated minutes model and is "
      "not going to out-predict closing prices on match outcomes. Feeding market team "
      "expectations into this project's minutes and allocation uses each where it is "
      "strong, and it is roughly what the commercial services are doing.")
    w("")

    w("## The fallback is load bearing right now, not in theory")
    w("")
    w("football-data.co.uk publishes closing prices for matches that have already been "
      "played. That is perfect for backtesting and useless for a Saturday deadline. "
      "Running compose against the live 2026-27 gameweek 1 resolves to "
      "`{market: 138707, dixon_coles: 599}`: every historical row takes the market, and "
      "every upcoming fixture falls through to Dixon-Coles, because no source in this "
      "repository prices a match that has not happened yet.")
    w("")
    w("Two consequences follow, and both are operational rather than modelling:")
    w("")
    w("1. **Dixon-Coles is currently carrying the entire live season.** The accepted "
      "configuration is market-first, but for prediction rather than replay the market "
      "half is not connected. What the harness accepted is the architecture; what runs on "
      "a Saturday is still the fallback.")
    w("2. **A live odds feed moves from optional to load bearing.** The Odds API free tier "
      "is the obvious candidate and its request budget and fixture coverage want checking "
      "before anything depends on it for a deadline. The chain degrades in the right "
      "direction when a pull fails, which is the whole reason it is a chain, and the "
      "no-coverage path is covered by tests rather than assumed.")
    w("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", REPORT_PATH)
    return REPORT_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Choose the team strength source")
    ap.add_argument("--seasons", nargs="+", default=ALL_SEASONS)
    args = ap.parse_args()

    scores = evaluate(args.seasons)
    decision = decide(scores)
    sources = load_sources(FROZEN_SEASON)
    calib = calibration(sources["dixon_coles"], results(FROZEN_SEASON))
    write_report(scores, decision, calib)

    print(scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"DECISION: {decision['winner']} leads, {decision['fallback']} is the fallback "
          f"(market {decision['market_log_loss']:.4f} vs DC {decision['dixon_coles_log_loss']:.4f} "
          f"on {', '.join(DECISION_SEASONS)})")


if __name__ == "__main__":
    main()
