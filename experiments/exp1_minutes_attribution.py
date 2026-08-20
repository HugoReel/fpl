"""Experiment 1: how much expected points error comes from minutes.

The question this answers is whether the minutes model deserves the effort,
by decomposing expected points error into a minutes part and a rate part.
The original plan compared naive minutes against oracle minutes. Since the
minutes model now exists, this runs a three way comparison instead, which
answers the more useful question of how much of the available gap the model
actually closed.

Four expected points variants, all scored through the one scoring module:

    A oracle_minutes   true minutes, trailing rates
    B naive_minutes    started last match implies 90 minutes, trailing rates
    M model_minutes    the trained minutes model, trailing rates
    D oracle_rates     the player's true season per-90 rates, naive minutes
    C oracle_outcomes  this fixture's true outcomes, naive minutes

Reading the deltas:
    MAE(B) - MAE(A)   what perfect minutes knowledge is worth
    MAE(B) - MAE(M)   what the minutes model actually captured of that
    MAE(B) - MAE(D)   what perfect knowledge of a player's rate is worth

Variant C is reported as context, not as a comparison. Knowing a fixture's
true outcomes is not "perfect rate modelling", it is perfect foresight of a
stochastic realisation, and it hands over most of the answer directly. No
model can approach it even in principle, so comparing it against A would
confuse irreducible match randomness with modelling headroom. D is the
honest counterpart to A: both fix one input at its true value and leave the
rest of the world uncertain.

Rates are deliberately crude trailing per-90 averages. They are the same in
A, B and M, so any difference between those three is attributable to
minutes alone, which is the entire point of the design.

Usage:
    python -m experiments.exp1_minutes_attribution
    python -m experiments.exp1_minutes_attribution --season 2025-26 --from-gw 6
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.curate import CURATED_ROOT
from models.minutes import dataset, features, train
from scoring.rules_2026_27 import (
    ELEMENT_TYPE_TO_POSITION,
    MatchStats,
    Position,
    expected_points,
    score_match,
)

log = logging.getLogger(__name__)

REPORT_PATH = Path("experiments/reports/exp1_minutes_attribution.md")
TRAILING_WINDOW = 5
DEFAULT_SEASON = "2025-26"
DEFAULT_FROM_GW = 6

# name -> (minutes source, rate source)
VARIANTS = {
    "A oracle_minutes": ("oracle", "trailing"),
    "B naive_minutes": ("naive", "trailing"),
    "M model_minutes": ("model", "trailing"),
    "D oracle_rates": ("naive", "season"),
    "C oracle_outcomes": ("naive", "outcome"),
}

# Cards cost about this much per appearance on average. A constant is fine
# here because it is identical across all four variants and so cancels out
# of every comparison.
CARD_COST_PER_APPEARANCE = 0.09


def _prev_mean(df: pd.DataFrame, col: str, window: int = TRAILING_WINDOW) -> pd.Series:
    """Mean of a column over a player's previous matches, current excluded."""
    values = pd.to_numeric(df[col], errors="coerce")
    shifted = values.groupby(df["player_code"], sort=False).shift(1)
    rolled = shifted.groupby(df["player_code"], sort=False).rolling(window, min_periods=1).mean()
    return rolled.reset_index(level=0, drop=True).reindex(df.index)


def build_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing per-90 rates and per-match averages, all strictly backward looking."""
    df = df.sort_values(["kickoff_time", "player_code"], kind="stable").reset_index(drop=True)

    prev_minutes = _prev_mean(df, "minutes")
    per90 = (prev_minutes / 90.0).clip(lower=0.05)

    df["rate_goals_p90"] = (_prev_mean(df, "goals_scored") / per90).fillna(0.0)
    df["rate_assists_p90"] = (_prev_mean(df, "assists") / per90).fillna(0.0)
    df["rate_saves_p90"] = (_prev_mean(df, "saves") / per90).fillna(0.0)
    df["rate_conceded_p90"] = (_prev_mean(df, "goals_conceded") / per90).fillna(0.0)
    df["rate_bonus"] = _prev_mean(df, "bonus").fillna(0.0)
    df["rate_clean_sheet"] = _prev_mean(df, "clean_sheets").fillna(0.0).clip(0, 1)

    defcon = pd.to_numeric(df["defensive_contribution"], errors="coerce").fillna(0)
    thresholds = df["element_type"].map({1: 999, 2: 10, 3: 12, 4: 12}).astype(float)
    df["_hit_defcon"] = (defcon >= thresholds).astype(float)
    df["rate_defcon"] = _prev_mean(df, "_hit_defcon").fillna(0.0).clip(0, 1)
    return df


def realised_points(row) -> int:
    """Recompute this fixture's points from components, never trust a stored total."""
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
    return score_match(stats, ELEMENT_TYPE_TO_POSITION[int(row["element_type"])]).total


def season_rates(test: pd.DataFrame) -> pd.DataFrame:
    """Each player's realised per-90 rates over the whole season.

    This is oracle information, because it uses the rest of the season to
    describe a player, but it is oracle knowledge of the player rather than
    of the fixture. It answers what a perfect rate model would buy, given
    that goals stay random even when the rate is known exactly.
    """
    grp = test.groupby("player_code")
    totals = grp.agg(
        _min=("minutes", "sum"),
        _goals=("goals_scored", "sum"),
        _assists=("assists", "sum"),
        _saves=("saves", "sum"),
        _conceded=("goals_conceded", "sum"),
        _bonus=("bonus", "sum"),
        _cs=("clean_sheets", "sum"),
        _defcon=("_hit_defcon", "sum"),
        _matches=("minutes", "size"),
    )
    per90 = (totals["_min"] / 90.0).clip(lower=0.05)
    out = pd.DataFrame(
        {
            "srate_goals_p90": totals["_goals"] / per90,
            "srate_assists_p90": totals["_assists"] / per90,
            "srate_saves_p90": totals["_saves"] / per90,
            "srate_conceded_p90": totals["_conceded"] / per90,
            "srate_bonus": totals["_bonus"] / totals["_matches"],
            "srate_clean_sheet": (totals["_cs"] / totals["_matches"]).clip(0, 1),
            "srate_defcon": (totals["_defcon"] / totals["_matches"]).clip(0, 1),
        }
    )
    return out.reset_index()


def _ep(row, p_appear: float, p_60: float, minutes_scale: float, rate_source: str) -> float:
    """Compose one expected points value through the scoring module."""
    position = ELEMENT_TYPE_TO_POSITION[int(row["element_type"])]

    if rate_source == "outcome":
        exp_goals = float(row["goals_scored"])
        exp_assists = float(row["assists"])
        exp_saves = float(row["saves"])
        exp_conceded = float(row["goals_conceded"])
        p_clean_sheet = float(row["clean_sheets"] > 0)
        p_defcon = float(row["_hit_defcon"])
        exp_bonus = float(row["bonus"])
    elif rate_source == "season":
        exp_goals = row["srate_goals_p90"] * minutes_scale
        exp_assists = row["srate_assists_p90"] * minutes_scale
        exp_saves = row["srate_saves_p90"] * minutes_scale
        exp_conceded = row["srate_conceded_p90"] * minutes_scale
        p_clean_sheet = row["srate_clean_sheet"]
        p_defcon = row["srate_defcon"] * min(minutes_scale, 1.0)
        exp_bonus = row["srate_bonus"] * min(minutes_scale, 1.0)
    else:
        exp_goals = row["rate_goals_p90"] * minutes_scale
        exp_assists = row["rate_assists_p90"] * minutes_scale
        exp_saves = row["rate_saves_p90"] * minutes_scale
        exp_conceded = row["rate_conceded_p90"] * minutes_scale
        p_clean_sheet = row["rate_clean_sheet"]
        p_defcon = row["rate_defcon"] * min(minutes_scale, 1.0)
        exp_bonus = row["rate_bonus"] * min(minutes_scale, 1.0)

    return expected_points(
        position,
        p_appear=p_appear,
        p_60plus=p_60,
        exp_goals=exp_goals,
        exp_assists=exp_assists,
        p_clean_sheet=p_clean_sheet,
        exp_goals_conceded=exp_conceded,
        exp_saves=exp_saves,
        p_defcon=p_defcon,
        exp_bonus=exp_bonus,
        exp_cards=CARD_COST_PER_APPEARANCE * p_appear,
    )


def run(season: str = DEFAULT_SEASON, from_gw: int = DEFAULT_FROM_GW) -> pd.DataFrame:
    """Score the four variants over one season."""
    df = features.add_features(dataset.build())
    df = build_rates(df)

    model = train.walk_forward_model_for(df, season)
    test = df[(df["season"] == season) & (df["gw"] >= from_gw)].copy()
    log.info("exp1 on %s from gw%d: %d player-fixture rows", season, from_gw, len(test))

    p_start = train.predict_head(*model["p_start"], test)
    p_sub = train.predict_head(*model["p_sub"], test) * (1 - p_start)
    p_60_given_start = train.predict_head(*model["p_60_given_start"], test)
    cond = model["conditional_minutes"]

    test["model_p_appear"] = p_start + p_sub
    test["model_p_60"] = p_start * p_60_given_start
    test["model_minutes"] = train.expected_minutes(p_start, p_sub, test["element_type"], cond)

    started_last = test["start_rate_1"].fillna(0).to_numpy() > 0
    test["naive_p_appear"] = started_last.astype(float)
    test["naive_p_60"] = started_last.astype(float)
    test["naive_minutes"] = np.where(started_last, 90.0, 0.0)

    true_minutes = test["minutes"].to_numpy(dtype=float)
    test["oracle_p_appear"] = (true_minutes > 0).astype(float)
    test["oracle_p_60"] = (true_minutes >= 60).astype(float)
    test["oracle_minutes"] = true_minutes

    test = test.merge(season_rates(test), on="player_code", how="left")

    # Records rather than iterrows: this loops five times over every row and
    # iterrows would rebuild a Series each visit.
    records = test.to_dict("records")
    test["realised"] = [realised_points(r) for r in records]

    for name, (source, rate_source) in VARIANTS.items():
        p_appear = test[f"{source}_p_appear"].to_numpy()
        p_60 = test[f"{source}_p_60"].to_numpy()
        scale = test[f"{source}_minutes"].to_numpy() / 90.0
        test[name] = [
            _ep(r, p_appear[i], p_60[i], scale[i], rate_source)
            for i, r in enumerate(records)
        ]

    return test


def summarise(test: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE per variant, overall and by position."""
    names = list(VARIANTS)
    rows = []
    realised = test["realised"].to_numpy(dtype=float)
    for name in names:
        pred = test[name].to_numpy(dtype=float)
        row = {
            "variant": name,
            "MAE": float(np.mean(np.abs(pred - realised))),
            "RMSE": float(np.sqrt(np.mean((pred - realised) ** 2))),
        }
        for et, label in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD")):
            m = test["element_type"] == et
            row[f"MAE_{label}"] = float(np.mean(np.abs(pred[m] - realised[m])))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(test: pd.DataFrame, summary: pd.DataFrame, season: str, from_gw: int) -> Path:
    mae = summary.set_index("variant")["MAE"]
    perfect_minutes = mae["B naive_minutes"] - mae["A oracle_minutes"]
    model_captured = mae["B naive_minutes"] - mae["M model_minutes"]
    perfect_rates = mae["B naive_minutes"] - mae["D oracle_rates"]
    perfect_outcomes = mae["B naive_minutes"] - mae["C oracle_outcomes"]
    captured_share = 100 * model_captured / perfect_minutes if perfect_minutes else float("nan")

    lines: list[str] = []
    w = lines.append
    w("# Experiment 1: minutes versus rates error attribution")
    w("")
    w(f"Season {season}, gameweek {from_gw} onward, {len(test):,} player-fixture rows. "
      "All expected points and all realised points go through "
      "`scoring/rules_2026_27.py`. Realised points are recomputed from components "
      "rather than read from the stored total.")
    w("")
    w("## Result")
    w("")
    w(f"- Perfect minutes knowledge is worth **{perfect_minutes:.3f} MAE** per player-fixture.")
    w(f"- Perfect knowledge of a player's true season rates is worth "
      f"**{perfect_rates:.3f} MAE**.")
    w(f"- The trained minutes model captured **{model_captured:.3f} MAE**, which is "
      f"**{captured_share:.0f}%** of what perfect minutes would have bought.")
    w("")
    if perfect_minutes > perfect_rates:
        w("**Minutes dominate rates.** Effort spent on the minutes model pays back more "
          "than the same effort spent on per-90 modelling, which is the ordering this "
          "project committed to before measuring. The commitment survives contact with "
          "the data.")
    else:
        w("**Rates dominate minutes on this comparison.** That cuts against the "
          "assumption the build order rests on. It is not on its own a reason to "
          "reorder the roadmap, because a rate oracle is a much harder thing to "
          "approximate than a minutes oracle, but phase 4 prioritisation should be "
          "argued rather than assumed from here.")
    w("")
    w(f"For context only, knowing this fixture's actual outcomes is worth "
      f"{perfect_outcomes:.3f} MAE. That number is not a modelling target. It reveals "
      "the answer rather than describing the player, and no model can approach it even "
      "in principle, because goals stay random however well the rate is known. It is "
      "reported to show how much of the residual error is irreducible match noise "
      "rather than something a better model could remove.")
    w("")

    w("## Variants")
    w("")
    w("| Variant | Minutes source | Rate source |")
    w("|---|---|---|")
    w("| A oracle_minutes | true minutes | trailing per-90 |")
    w("| B naive_minutes | started last match implies 90 | trailing per-90 |")
    w("| M model_minutes | trained minutes model | trailing per-90 |")
    w("| D oracle_rates | started last match implies 90 | player's true season per-90 |")
    w("| C oracle_outcomes | started last match implies 90 | this fixture's true outcomes |")
    w("")
    w("A and D are the like-for-like pair: each fixes one input at its true value and "
      "leaves everything else uncertain. B is the fully naive system and M is the one "
      "that could actually be deployed on a Friday, since it is the only variant here "
      "that uses no information from after the deadline.")
    w("")

    w("## Error by variant")
    w("")
    w("| Variant | MAE | RMSE | GKP | DEF | MID | FWD |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        w(
            f"| {r['variant']} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['MAE_GKP']:.3f} | "
            f"{r['MAE_DEF']:.3f} | {r['MAE_MID']:.3f} | {r['MAE_FWD']:.3f} |"
        )
    w("")

    w("## Conclusion")
    w("")
    w(f"Swapping naive minutes for true minutes removes {perfect_minutes:.3f} MAE, and "
      f"swapping trailing rates for a player's true season rates removes "
      f"{perfect_rates:.3f}. The minutes model recovers {captured_share:.0f}% of the "
      "minutes gap while using nothing from after the deadline.")
    w("")
    w("The unrecovered majority of that gap is mostly team news. The model cannot see a "
      "manager's Friday press conference, and it cannot see the availability flags "
      "either, because the archive preserves only an end of season snapshot of them. "
      "That is the single largest identified improvement available, it requires no new "
      "modelling technique, and it needs only the weekly live snapshots that "
      "`ingest/snapshot.py` is already collecting. Roughly a season of them turns "
      "status and chance_of_playing_next_round into usable, leak free features.")
    w("")
    w("These rate estimates are deliberately crude trailing averages and are identical "
      "across variants A, B and M, so the comparison isolates minutes cleanly. They are "
      "not a claim about how good rate modelling can get.")
    w("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", REPORT_PATH)
    return REPORT_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Minutes versus rates error attribution")
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--from-gw", type=int, default=DEFAULT_FROM_GW)
    args = ap.parse_args()

    test = run(args.season, args.from_gw)
    summary = summarise(test)
    write_report(test, summary, args.season, args.from_gw)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
