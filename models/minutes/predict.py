"""Predict minutes for an upcoming gameweek.

Writes data/predictions/{season}/gw{g}/minutes.parquet with one row per
player per fixture, so a double gameweek produces two rows and nothing
downstream has to guess.

The upcoming rows are appended to the full history and pushed through the
same feature builder used in training. That is deliberate. Every feature is
a shift(1) within player, so an unplayed row can only draw on matches that
have already kicked off, and train and predict cannot drift apart because
there is only one implementation.

History carries across seasons on player_code, which is what makes a
gameweek 1 prediction possible at all: on the first weekend of a season the
only evidence about a player is what they did in the previous one.

Usage:
    python -m models.minutes.predict --season 2026-27 --gw 1
    python -m models.minutes.predict --season 2025-26 --gw 20 --version 20260820T191553Z
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from ingest.curate import CURATED_ROOT
from models.minutes import dataset, features
from models.minutes.train import (
    HEADS,
    MODEL_STORE,
    PROB_CEIL,
    PROB_FLOOR,
    expected_minutes,
    latest_version,
)

log = logging.getLogger(__name__)

PREDICTIONS_ROOT = Path("data/predictions")

OUTPUT_COLUMNS = [
    "season",
    "gw",
    "player_id",
    "player_code",
    "fixture_id",
    "element_type",
    "team_id",
    "opponent_team",
    "was_home",
    "p_start",
    "p_60",
    "p_sub",
    "p_appear",
    "exp_minutes",
    # kept for debugging: p_60 is this times p_start
    "p_60_given_start",
]


def load_model(version: str | None = None, store: Path = MODEL_STORE) -> dict:
    version = version or latest_version(store)
    if version is None:
        raise FileNotFoundError(
            f"no trained model in {store}. Run python -m models.minutes.train first."
        )
    path = store / version
    metadata = json.loads((path / "metadata.json").read_text())
    heads = {
        head: (
            lgb.Booster(model_file=str(path / f"{head}.txt")),
            joblib.load(path / f"{head}_calibrator.joblib"),
        )
        for head in HEADS
    }
    log.info("loaded minutes model %s (trained on %s)", version, metadata["train_seasons"])
    return {"version": version, "metadata": metadata, "heads": heads}


def predict_gameweek(
    season: str,
    gw: int,
    model: dict,
    curated_root: Path = CURATED_ROOT,
) -> pd.DataFrame:
    """Score every player fixture in the target gameweek."""
    upcoming = dataset.build_upcoming(season, gw, curated_root)

    history = dataset.build(curated_root=curated_root)
    # Anything at or after the target gameweek in the target season is the
    # future as far as this prediction is concerned, including results that
    # already exist on disk when backfilling a past gameweek.
    history = history[~((history["season"] == season) & (history["gw"] >= gw))]
    log.info(
        "predicting %s gw%d: %d upcoming rows from %d historical rows",
        season,
        gw,
        len(upcoming),
        len(history),
    )

    combined = pd.concat([history, upcoming], ignore_index=True)
    featured = features.add_features(combined, curated_root=curated_root)

    target = featured[
        (featured["season"] == season) & (featured["gw"] == gw) & featured["minutes"].isna()
    ].copy()
    if target.empty:
        raise ValueError(f"no upcoming rows survived feature building for {season} gw{gw}")

    def _head(name: str):
        booster, calibrator = model["heads"][name]
        raw = calibrator.predict(booster.predict(features.feature_matrix(target)))
        # Isotonic regression saturates at exactly 0 and 1 in its end bins.
        # No footballer is certain to start, so the output is kept strictly
        # inside the open interval, matching the clipping the metrics use.
        return raw.clip(PROB_FLOOR, PROB_CEIL)

    target["p_start"] = _head("p_start")
    target["p_60_given_start"] = _head("p_60_given_start")
    p_sub_given_no_start = _head("p_sub")

    target["p_sub"] = p_sub_given_no_start * (1 - target["p_start"])
    target["p_60"] = target["p_start"] * target["p_60_given_start"]
    target["p_appear"] = target["p_start"] + target["p_sub"]
    target["exp_minutes"] = expected_minutes(
        target["p_start"].to_numpy(),
        target["p_sub"].to_numpy(),
        target["element_type"],
        model["metadata"]["conditional_minutes"],
    )

    return target[OUTPUT_COLUMNS].sort_values(
        ["p_start", "player_code"], ascending=[False, True]
    ).reset_index(drop=True)


def write_predictions(df: pd.DataFrame, season: str, gw: int) -> Path:
    out_dir = PREDICTIONS_ROOT / season / f"gw{gw}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "minutes.parquet"
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Player names carry accents and the Windows console defaults to cp1252,
    # which cannot encode them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Predict minutes for an upcoming gameweek")
    ap.add_argument("--season", required=True)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--version", default=None, help="model version, defaults to newest")
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    args = ap.parse_args()

    model = load_model(args.version)
    df = predict_gameweek(args.season, args.gw, model)
    path = write_predictions(df, args.season, args.gw)

    players = pd.read_parquet(CURATED_ROOT / args.season / "players.parquet")
    named = df.merge(
        players[["player_id", "web_name", "team_short"]], on="player_id", how="left"
    )
    show = named.head(args.top)[
        ["web_name", "team_short", "element_type", "p_start", "p_60", "p_sub", "exp_minutes"]
    ]
    print(f"\n{path}\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
