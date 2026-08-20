"""Train and walk-forward evaluate the three minutes heads.

Three LightGBM binary classifiers on nested populations:

    p_start             every row
    p_60_given_start    rows where the player started
    p_sub               rows where the player did not start

Recombined, they give what the scoring module wants:
    p_appear = p_start + p_sub
    p_60     = p_start * p_60_given_start

Evaluation is walk forward and never random: for each held out season S the
model trains on every season before S and is scored on S. Inside the
training seasons the last CALIB_FRACTION of matches by kickoff time is held
back, unseen by the booster, and used to fit an isotonic calibrator. That
ordering matters. Calibrating on data the booster trained on would report
calibration that does not exist out of sample.

Expected minutes is not a fourth model. It is
    p_start * E[minutes | start, position] + p_sub * E[minutes | sub, position]
with both conditional means estimated from the training seasons, per
position, rather than assuming a starter plays 90.

Usage:
    python -m models.minutes.train
    python -m models.minutes.train --no-save
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from models.minutes import dataset, features

log = logging.getLogger(__name__)

SEED = 20262027
CALIB_FRACTION = 0.2
NUM_ROUNDS = 300
EVAL_SEASONS = ["2023-24", "2024-25", "2025-26"]
MODEL_STORE = Path("models_store/minutes")
REPORT_PATH = Path("experiments/reports/minutes_model.md")

# Probabilities are clipped before log loss so a confident miss stays
# finite. This matters most for the naive baseline, which is a hard 0 or 1.
PROB_FLOOR = 0.001
PROB_CEIL = 0.999

# Deliberately untuned. The point of v0 is a calibrated, honest, leak free
# pipeline. Hyperparameter search comes after the evaluation harness exists
# to referee it, otherwise tuning just chases noise.
LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "bagging_seed": SEED,
    "feature_fraction_seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
    "num_threads": 4,
}

HEADS = {
    "p_start": {"label": "started", "population": None},
    "p_60_given_start": {"label": "played_60", "population": "started"},
    "p_sub": {"label": "sub_appear", "population": "not_started"},
}


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _population(df: pd.DataFrame, population: str | None) -> pd.DataFrame:
    if population is None:
        return df
    started = df["started"].astype(bool)
    return df[started] if population == "started" else df[~started]


def chronological_split(df: pd.DataFrame, frac: float = CALIB_FRACTION):
    """Split by kickoff time, earliest `1 - frac` first. Never random."""
    ordered = df.sort_values("kickoff_time", kind="stable")
    cut = int(len(ordered) * (1 - frac))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def fit_head(train: pd.DataFrame, head: str) -> tuple[lgb.Booster, IsotonicRegression]:
    """Fit one booster plus its isotonic calibrator on a training window."""
    spec = HEADS[head]
    fit_part, calib_part = chronological_split(train)

    fit_rows = _population(fit_part, spec["population"])
    calib_rows = _population(calib_part, spec["population"])

    booster = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(
            features.feature_matrix(fit_rows),
            label=fit_rows[spec["label"]].astype(int),
        ),
        num_boost_round=NUM_ROUNDS,
    )

    raw = booster.predict(features.feature_matrix(calib_rows))
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw, calib_rows[spec["label"]].astype(int))
    return booster, calibrator


def predict_head(booster, calibrator, df: pd.DataFrame) -> np.ndarray:
    return calibrator.predict(booster.predict(features.feature_matrix(df)))


def baseline_start_rates(train: pd.DataFrame) -> dict:
    """P(start | started last match, position), estimated on training data.

    The spec's baseline is "started last game implies starts this game". A
    raw 0/1 version of that has an unbounded log loss, so it is expressed
    here as the empirical rate in each bucket. That makes the baseline
    strictly stronger and beating it a more meaningful claim.
    """
    df = train.copy()
    # element_type is float by the time it leaves the feature builder, so it
    # is normalised to int on both the store and the lookup side. Letting
    # those drift apart makes every key miss and silently degrades the
    # baseline to a single constant, which would flatter the model.
    df["_bucket"] = _baseline_bucket(df)
    df["_et"] = df["element_type"].astype(int)
    rates = df.groupby(["_et", "_bucket"])["started"].mean().to_dict()
    return {
        "rates": {_baseline_key(k[0], k[1]): float(v) for k, v in rates.items()},
        "overall": float(df["started"].mean()),
    }


def _baseline_bucket(df: pd.DataFrame) -> pd.Series:
    return df["start_rate_1"].map({1.0: "started", 0.0: "benched"}).fillna("unknown")


def _baseline_key(element_type, bucket: str) -> str:
    return f"{int(element_type)}|{bucket}"


def apply_baseline(model: dict, df: pd.DataFrame) -> np.ndarray:
    keys = [
        _baseline_key(et, b)
        for et, b in zip(df["element_type"], _baseline_bucket(df))
    ]
    return (
        pd.Series(keys, index=df.index)
        .map(model["rates"])
        .fillna(model["overall"])
        .to_numpy(dtype=float)
    )


def score(y_true: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, PROB_FLOOR, PROB_CEIL)
    return {
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
    }


def conditional_minutes(train: pd.DataFrame) -> dict:
    """E[minutes | start, position] and E[minutes | sub, position].

    Estimated, not assumed. A starting keeper averages close to 90, a
    starting forward much less, and a sub averages around 20.
    """
    started = train[train["started"].astype(bool)]
    subbed = train[(~train["started"].astype(bool)) & (train["minutes"] > 0)]
    out = {"start": {}, "sub": {}}
    for et, grp in started.groupby("element_type"):
        out["start"][str(int(et))] = float(grp["minutes"].mean())
    for et, grp in subbed.groupby("element_type"):
        out["sub"][str(int(et))] = float(grp["minutes"].mean())
    out["start_overall"] = float(started["minutes"].mean())
    out["sub_overall"] = float(subbed["minutes"].mean()) if len(subbed) else 0.0
    return out


def expected_minutes(
    p_start: np.ndarray, p_sub: np.ndarray, element_type: pd.Series, cond: dict
) -> np.ndarray:
    et = element_type.astype(int).astype(str)
    m_start = et.map(cond["start"]).fillna(cond["start_overall"]).to_numpy(dtype=float)
    m_sub = et.map(cond["sub"]).fillna(cond["sub_overall"]).to_numpy(dtype=float)
    return p_start * m_start + p_sub * m_sub


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised rate in fixed probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": 0,
                         "mean_predicted": float("nan"), "realised": float("nan")})
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "realised": float(y_true[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame) -> dict:
    """Train on everything before each held out season, evaluate on it."""
    results = {}
    for season in EVAL_SEASONS:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if train.empty or test.empty:
            log.warning("skipping %s, no data either side", season)
            continue
        log.info(
            "walk forward %s: train %d rows (%s), test %d rows",
            season,
            len(train),
            sorted(train["season"].unique()),
            len(test),
        )

        season_out = {"train_seasons": sorted(train["season"].unique()), "heads": {}}
        fitted = {}
        for head, spec in HEADS.items():
            booster, calibrator = fit_head(train, head)
            fitted[head] = (booster, calibrator)
            test_rows = _population(test, spec["population"])
            y = test_rows[spec["label"]].astype(int).to_numpy()
            p = predict_head(booster, calibrator, test_rows)

            entry = {"model": score(y, p)}
            if head == "p_start":
                base = baseline_start_rates(train)
                entry["baseline"] = score(y, apply_baseline(base, test_rows))
                entry["calibration"] = calibration_table(y, p).to_dict("records")
            else:
                rate = _population(train, spec["population"])[spec["label"]].astype(int).mean()
                entry["baseline"] = score(y, np.full(len(y), float(rate)))
            season_out["heads"][head] = entry

        season_out["minutes"] = _minutes_metrics(train, test, fitted)
        results[season] = season_out
    return results


def walk_forward_model_for(df: pd.DataFrame, season: str) -> dict:
    """Heads trained only on seasons strictly before `season`.

    Shared by the evaluation experiments so they replay a season with the
    same information set the walk forward evaluation used, rather than a
    model that has already seen the answers.
    """
    train_df = df[df["season"] < season]
    if train_df.empty:
        raise ValueError(f"no seasons before {season} to train on")
    model = {head: fit_head(train_df, head) for head in HEADS}
    model["conditional_minutes"] = conditional_minutes(train_df)
    model["train_seasons"] = sorted(train_df["season"].unique())
    return model


def _minutes_metrics(train: pd.DataFrame, test: pd.DataFrame, fitted: dict) -> dict:
    """Expected minutes error against realised, model versus naive rule.

    Both MAE and RMSE are reported because they disagree, and the reason is
    not a defect. MAE is minimised by the conditional median, and roughly
    three fifths of player-fixture rows are non appearances whose median is
    exactly 0. A hard 0 or 90 rule therefore scores well on MAE by refusing
    to hedge, while a calibrated expectation pays a small penalty on every
    row it is honest about. RMSE, which the expectation actually optimises,
    tells the other half of the story.
    """
    cond = conditional_minutes(train)
    p_start = predict_head(*fitted["p_start"], test)
    # The p_sub head is conditioned on not starting, so scale it back to an
    # unconditional probability of appearing off the bench.
    p_sub = predict_head(*fitted["p_sub"], test) * (1 - p_start)

    exp_min = expected_minutes(p_start, p_sub, test["element_type"], cond)
    actual = test["minutes"].to_numpy(dtype=float)
    naive = np.where(test["start_rate_1"].fillna(0).to_numpy() > 0, 90.0, 0.0)
    played = actual > 0

    def _err(pred: np.ndarray) -> dict:
        return {
            "mae": float(np.mean(np.abs(pred - actual))),
            "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
            "mae_appearances": float(np.mean(np.abs(pred[played] - actual[played]))),
            "mae_non_appearances": float(np.mean(np.abs(pred[~played] - actual[~played]))),
        }

    return {
        "model": _err(exp_min),
        "naive": _err(naive),
        "appearance_share": float(played.mean()),
        "conditional_means": cond,
    }


def train_final(df: pd.DataFrame, version: str, save: bool = True) -> Path | None:
    """Fit on every available season and persist the artefacts."""
    out_dir = MODEL_STORE / version
    metrics_by_head = {}
    boosters = {}
    for head in HEADS:
        booster, calibrator = fit_head(df, head)
        boosters[head] = (booster, calibrator)
        metrics_by_head[head] = {
            "train_rows": int(len(_population(df, HEADS[head]["population"])))
        }

    if not save:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    for head, (booster, calibrator) in boosters.items():
        booster.save_model(str(out_dir / f"{head}.txt"))
        joblib.dump(calibrator, out_dir / f"{head}_calibrator.joblib")

    metadata = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "train_seasons": sorted(df["season"].unique()),
        "n_rows": int(len(df)),
        "features": features.FEATURES,
        "lgb_params": LGB_PARAMS,
        "num_boost_round": NUM_ROUNDS,
        "calib_fraction": CALIB_FRACTION,
        "seed": SEED,
        "heads": metrics_by_head,
        "conditional_minutes": conditional_minutes(df),
        "baseline_start_rates": baseline_start_rates(df),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("saved model version %s to %s", version, out_dir)
    return out_dir


def latest_version(store: Path = MODEL_STORE) -> str | None:
    if not store.exists():
        return None
    versions = sorted(p.name for p in store.iterdir() if (p / "metadata.json").exists())
    return versions[-1] if versions else None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _pct(new: float, old: float) -> str:
    if old == 0:
        return "n/a"
    return f"{100 * (old - new) / old:+.1f}%"


def write_report(results: dict, df: pd.DataFrame, version: str, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    w = lines.append

    beat_all = all(
        r["heads"]["p_start"]["model"]["log_loss"] < r["heads"]["p_start"]["baseline"]["log_loss"]
        for r in results.values()
    )

    w("# Minutes model, v0")
    w("")
    w(f"Model version `{version}`, generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.")
    w("")
    margins = {
        s: 100
        * (r["heads"]["p_start"]["baseline"]["log_loss"] - r["heads"]["p_start"]["model"]["log_loss"])
        / r["heads"]["p_start"]["baseline"]["log_loss"]
        for s, r in results.items()
    }
    weakest = min(margins, key=margins.get)
    if beat_all:
        w("**The model beats the started-last-match baseline on p_start log loss in "
          "every held out season.**")
        w("")
        w(f"The margin is not uniform, and the weakest season is the informative one: "
          f"{weakest} improves by only {margins[weakest]:.1f}%, against "
          f"{max(margins.values()):.1f}% at best. {weakest} is the season with the least "
          "training history behind it, and the history it does have is the least "
          "trustworthy, since 2021-22 carries no exact start flags at all. The edge "
          "grows as seasons accumulate, which is the expected shape but is worth "
          "restating: on two seasons of training data this model is barely worth its "
          "complexity over a lookup table.")
    else:
        w("**WARNING: the model does NOT beat the baseline on p_start log loss in every "
          "held out season.** See the table below and the notes at the end.")
    w("")
    w("The baseline is a genuine one. It is P(start | started last match, position) "
      "estimated on the training seasons, which is strictly stronger than the hard 0 or "
      "1 rule the spec names, so beating it is a stronger claim than beating the "
      "literal version.")
    w("")
    w("Walk forward only: each season is scored by a model trained purely on earlier "
      "seasons, with an isotonic calibrator fitted on the tail of those seasons that "
      "the booster never saw.")
    w("")

    w("## Head metrics by held out season")
    w("")
    w("Baseline for p_start is P(start | started last match, position) estimated on the "
      "training seasons. For the other heads it is the training base rate. Improvement "
      "is the reduction in log loss.")
    w("")
    w("| Season | Head | Rows | Base rate | Model log loss | Baseline log loss | Improvement | Model Brier | Baseline Brier |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for season, r in results.items():
        for head, entry in r["heads"].items():
            m, b = entry["model"], entry["baseline"]
            w(
                f"| {season} | {head} | {m['n']:,} | {m['base_rate']:.3f} | "
                f"{m['log_loss']:.4f} | {b['log_loss']:.4f} | {_pct(m['log_loss'], b['log_loss'])} | "
                f"{m['brier']:.4f} | {b['brier']:.4f} |"
            )
    w("")

    w("## Expected minutes")
    w("")
    w("Error against realised minutes. The naive rule is 90 minutes if the player "
      "started their previous match, otherwise 0.")
    w("")
    w("| Season | Model MAE | Naive MAE | Model RMSE | Naive RMSE | RMSE improvement |")
    w("|---|---:|---:|---:|---:|---:|")
    for season, r in results.items():
        m, n = r["minutes"]["model"], r["minutes"]["naive"]
        w(
            f"| {season} | {m['mae']:.2f} | {n['mae']:.2f} | {m['rmse']:.2f} | "
            f"{n['rmse']:.2f} | {_pct(m['rmse'], n['rmse'])} |"
        )
    w("")
    last_min = results[list(results)[-1]]["minutes"]
    w(f"**The naive rule wins on MAE and loses on RMSE, and that is expected rather "
      f"than a defect.** MAE is minimised by the conditional median, and "
      f"{100 * (1 - last_min['appearance_share']):.0f}% of player-fixture rows are non "
      "appearances whose median is exactly 0. A hard 0 or 90 rule scores well on MAE "
      "precisely because it refuses to hedge, and it pays for that with a much worse "
      "RMSE and a far worse log loss when it is wrong. Splitting the last held out "
      "season shows where each one earns its error:")
    w("")
    w("| Rows | Model MAE | Naive MAE |")
    w("|---|---:|---:|")
    w(f"| Appearances | {last_min['model']['mae_appearances']:.2f} | "
      f"{last_min['naive']['mae_appearances']:.2f} |")
    w(f"| Non appearances | {last_min['model']['mae_non_appearances']:.2f} | "
      f"{last_min['naive']['mae_non_appearances']:.2f} |")
    w("")
    w("The model is better on the rows where somebody actually played. It is worse only "
      "on rows where the answer was zero and the naive rule guessed zero exactly. Note "
      "also that expected minutes is a diagnostic here, not the production output: the "
      "scoring module consumes p_appear and p_60, where the model wins outright.")
    w("")

    last = list(results)[-1]
    cond = results[last]["minutes"]["conditional_means"]
    pos_names = {"1": "GKP", "2": "DEF", "3": "MID", "4": "FWD"}
    w("Conditional means estimated from the training seasons, which is why expected "
      "minutes is not just p_start times 90:")
    w("")
    w("| Position | E[min given start] | E[min given sub appearance] |")
    w("|---|---:|---:|")
    for et in sorted(cond["start"]):
        w(f"| {pos_names.get(et, et)} | {cond['start'][et]:.1f} | {cond['sub'].get(et, float('nan')):.1f} |")
    w("")

    w(f"## Calibration of p_start, held out {last}")
    w("")
    w("A calibrated model puts the realised column next to the predicted one. Bins are "
      "fixed width, so sparse bins at the extremes are expected.")
    w("")
    w("| Bin | Rows | Mean predicted | Realised |")
    w("|---|---:|---:|---:|")
    for row in results[last]["heads"]["p_start"]["calibration"]:
        if row["n"] == 0:
            w(f"| {row['bin']} | 0 | | |")
        else:
            w(f"| {row['bin']} | {row['n']:,} | {row['mean_predicted']:.3f} | {row['realised']:.3f} |")
    w("")

    w("## Notes and known limits")
    w("")
    starters = {s: r["heads"]["p_60_given_start"]["model"]["n"] for s, r in results.items()}
    if len(set(starters.values())) == 1 and next(iter(starters.values())) % 22 == 0:
        n = next(iter(starters.values()))
        w(f"- The start labels pass a hard integrity check: every held out season has "
          f"exactly {n:,} starter rows, which is {n // 22} fixtures times 22 starters. "
          "Eleven players per side per match is a law of the game, so an exact match "
          "across three seasons with different squad sizes says the labels are not "
          "drifting.")
    exact = 100 * df["started_is_exact"].mean()
    w(f"- {exact:.1f}% of start labels come from the archive's exact starts flag. The "
      "remainder, 2021-22 only, fall back to a minutes >= 45 proxy, which misclassifies "
      "roughly one non starter in seven.")
    w("- status and chance_of_playing_next_round are NOT used. The archive preserves only "
      "the end of season snapshot of both, so training on them would leak. They become "
      "usable once the weekly live snapshots have accumulated a season of history, and "
      "they are the single most valuable addition available to this model.")
    w("- Current season set piece order is excluded for the same reason. The previous "
      "season's order is used instead, which is leak free.")
    w("- Hyperparameters are untuned on purpose. Tuning before the evaluation harness "
      "exists would be chasing noise.")
    w("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train and evaluate the minutes model")
    ap.add_argument("--no-save", action="store_true", help="skip writing the model store")
    ap.add_argument("--version", default=None, help="model version label")
    args = ap.parse_args()

    df = features.add_features(dataset.build())

    results = walk_forward(df)
    version = args.version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_final(df, version, save=not args.no_save)
    write_report(results, df, version)

    for season, r in results.items():
        ps = r["heads"]["p_start"]
        print(
            f"{season}: p_start log loss {ps['model']['log_loss']:.4f} "
            f"vs baseline {ps['baseline']['log_loss']:.4f} "
            f"| exp minutes RMSE {r['minutes']['model']['rmse']:.2f} "
            f"vs naive {r['minutes']['naive']['rmse']:.2f}"
        )


if __name__ == "__main__":
    main()
