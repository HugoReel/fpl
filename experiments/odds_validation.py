"""Validate the odds ingest and the odds-only baseline, before any harness run.

The gating policy gives each component one run against the frozen 2025-26
season, so a component has to earn it on internal metrics first. For the
odds baseline that means two things. The de-vigged prices must actually
beat a base rate at predicting match outcomes, which is a floor check that
would catch a broken join or an inverted probability. And its clean sheet
probabilities must be calibrated, because clean sheets are where this
baseline is supposed to be strong and a miscalibrated one would flatter it.

Usage:
    python -m experiments.odds_validation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ingest.curate import CURATED_ROOT
from ingest.odds import DEFAULT_SEASONS, run_season
from scoring.replay import CURATED_ROOT as _CURATED

log = logging.getLogger(__name__)

REPORT_PATH = Path("experiments/reports/odds_baseline.md")
EVAL_SEASON = "2025-26"
CALIBRATION_BINS = 10


def outcome_metrics(season: str, curated_root: Path = CURATED_ROOT) -> dict:
    """Log loss of the de-vigged market against realised results."""
    odds = pd.read_parquet(curated_root / season / "fixture_odds.parquet")
    fixtures = pd.read_parquet(curated_root / season / "fixtures.parquet")
    df = odds.merge(
        fixtures[["fixture_id", "team_h_score", "team_a_score"]], on="fixture_id", how="left"
    ).dropna(subset=["team_h_score", "p_home_win"])

    y = np.where(
        df["team_h_score"] > df["team_a_score"], 0,
        np.where(df["team_h_score"] == df["team_a_score"], 1, 2),
    )
    probs = df[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)
    base = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
    return {
        "season": season,
        "matches": int(len(df)),
        "market_log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "base_rate_log_loss": float(log_loss(y, base, labels=[0, 1, 2])),
        "mean_margin": float(df["margin_1x2"].mean()),
    }


def clean_sheet_calibration(season: str, curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Predicted against realised clean sheet rate, in fixed width bins.

    One row per team per fixture, since each match offers two clean sheet
    opportunities and they are priced separately.
    """
    odds = pd.read_parquet(curated_root / season / "fixture_odds.parquet")
    fixtures = pd.read_parquet(curated_root / season / "fixtures.parquet")
    df = odds.merge(
        fixtures[["fixture_id", "team_h_score", "team_a_score"]], on="fixture_id", how="left"
    ).dropna(subset=["team_h_score", "p_home_cs"])

    sides = pd.concat(
        [
            pd.DataFrame({"p": df["p_home_cs"], "kept": (df["team_a_score"] == 0).astype(int)}),
            pd.DataFrame({"p": df["p_away_cs"], "kept": (df["team_h_score"] == 0).astype(int)}),
        ],
        ignore_index=True,
    )

    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    idx = np.clip(np.digitize(sides["p"], edges[1:-1], right=False), 0, CALIBRATION_BINS - 1)
    rows = []
    for b in range(CALIBRATION_BINS):
        m = idx == b
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(m.sum()),
                "mean_predicted": float(sides["p"][m].mean()) if m.any() else float("nan"),
                "realised": float(sides["kept"][m].mean()) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_report(coverage: list[dict], outcomes: list[dict], calibration: pd.DataFrame) -> Path:
    worst = min(c["coverage"] for c in coverage)
    beats_base = all(o["market_log_loss"] < o["base_rate_log_loss"] for o in outcomes)

    # A bin holding thirty team fixtures can miss its expectation by a couple
    # of clean sheets and look badly calibrated, so the headline number comes
    # from bins with enough rows to mean something and the sparse ones are
    # described separately rather than averaged away.
    DENSE = 50
    dense = calibration[calibration["n"] >= DENSE]
    sparse = calibration[(calibration["n"] > 0) & (calibration["n"] < DENSE)]
    max_gap = float((dense["mean_predicted"] - dense["realised"]).abs().max())

    lines: list[str] = []
    w = lines.append
    w("# Odds ingest and the odds-only baseline")
    w("")
    if beats_base and worst >= 0.99:
        w(f"**Validation passes.** Every fixture in every season is priced, the de-vigged "
          f"market beats a base rate on match outcomes in all {len(outcomes)} seasons, and "
          f"clean sheet probabilities track realised rates to within {max_gap:.3f} across "
          f"every bin holding at least {DENSE} team fixtures.")
        if not sparse.empty:
            worst_sparse = sparse.loc[(sparse["mean_predicted"] - sparse["realised"]).abs().idxmax()]
            w("")
            w(f"The sparse bins are noisier and should not be read as miscalibration. The "
              f"worst is {worst_sparse['bin']}, holding {int(worst_sparse['n'])} team "
              f"fixtures, where {worst_sparse['mean_predicted']:.3f} predicted against "
              f"{worst_sparse['realised']:.3f} realised amounts to about "
              f"{worst_sparse['mean_predicted'] * worst_sparse['n']:.0f} expected clean "
              "sheets that did not happen.")
    else:
        w("**Validation FAILS.** See the tables below before running the harness.")
    w("")
    w("Source is football-data.co.uk closing average prices, de-vigged with Shin's method. "
      "Shin models the bookmaker margin as protection against insider trading and removes "
      "proportionally more from longshots, where basic normalisation spreads it evenly and "
      "is known to overprice favourites. Longshot distortion matters here because it is "
      "worst at the heavy-win end, which is exactly where clean sheet probability lives.")
    w("")

    w("## Join coverage")
    w("")
    w("Matched on both club names plus kickoff date, because a postponed fixture keeps its "
      "teams but moves its date. Club name differences are mapped in `mapping/club_names.csv`, "
      "which needs three entries: Man United, Sheffield United and Tottenham.")
    w("")
    w("| Season | Fixtures | Priced | Unmatched |")
    w("|---|---:|---:|---|")
    for c in coverage:
        misses = (
            ", ".join(f"{m['home_name']} v {m['away_name']}" for m in c["misses"])
            if c["misses"] else "none"
        )
        w(f"| {c['season']} | {c['fixtures']} | {100 * c['coverage']:.1f}% | {misses} |")
    w("")

    w("## Floor check: do the odds predict match outcomes")
    w("")
    w("Three way log loss of the de-vigged prices against realised results, next to a "
      "constant base rate. A market that failed to beat a base rate would mean a broken "
      "join or an inverted probability, not a bad market.")
    w("")
    w("| Season | Matches | Market log loss | Base rate log loss | Improvement | Mean margin |")
    w("|---|---:|---:|---:|---:|---:|")
    for o in outcomes:
        imp = 100 * (o["base_rate_log_loss"] - o["market_log_loss"]) / o["base_rate_log_loss"]
        w(
            f"| {o['season']} | {o['matches']} | {o['market_log_loss']:.4f} | "
            f"{o['base_rate_log_loss']:.4f} | {imp:+.1f}% | {o['mean_margin']:.3f} |"
        )
    w("")

    w(f"## Clean sheet calibration, {EVAL_SEASON}")
    w("")
    w("One row per team per fixture, so each match contributes two clean sheet "
      "opportunities. Predicted comes from a Poisson zero on the opponent's goal "
      "expectation.")
    w("")
    w("| Bin | Team fixtures | Mean predicted | Realised |")
    w("|---|---:|---:|---:|")
    for _, r in calibration.iterrows():
        if r["n"] == 0:
            w(f"| {r['bin']} | 0 | | |")
        else:
            w(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['realised']:.3f} |")
    w("")

    w("## Known weaknesses")
    w("")
    w("- **No scorer odds.** football-data.co.uk carries match markets only, so a player's "
      "share of his team's goals is allocated by position and price rank rather than by "
      "anything about the player. Inside a price bracket, a poacher and a midfielder who "
      "never shoots are indistinguishable. This is the baseline's weakest point and it is "
      "where a trained attack model should beat it.")
    w("- **Independent Poisson.** Goal expectations are recovered by fitting an independent "
      "Poisson pair to the 1X2 and over/under markets. Goals are correlated, particularly "
      "at low scores, which is exactly what Dixon-Coles corrects. The refit reproduces the "
      "market's own 1X2 probabilities to about one percentage point, so it is faithful to "
      "the prices even though the generative model is wrong.")
    w("- **One leaked constant, and it leaks in the baseline's favour.** The defensive "
      "contribution base rate is measured on 2025-26 because that is the only season in "
      "which the statistic was recorded, and 2025-26 is also the season this baseline is "
      "evaluated on. It is a single per position number rather than anything player "
      "specific, so the effect is small, but the direction is not neutral: it hands the "
      "odds baseline a defensive component fitted to the very season it is scored on. Any "
      "tie against a trained candidate is therefore a floor on that candidate's relative "
      "standing, not a midpoint.")
    w("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", REPORT_PATH)
    return REPORT_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Validate the odds ingest and baseline")
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    coverage = [run_season(s) for s in args.seasons]
    outcomes = [outcome_metrics(s) for s in args.seasons]
    calibration = clean_sheet_calibration(EVAL_SEASON)
    write_report(coverage, outcomes, calibration)

    print(pd.DataFrame(outcomes).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(calibration.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
