"""Expected points for every player in a gameweek.

One command combining the minutes model with the v0 rate estimators through
`scoring/rules_2026_27.py`. No points arithmetic happens here. This module
only assembles inputs, hands them to the scoring module and adds up the
result, which is what keeps a rule change a one file diff.

Composition is per player per FIXTURE and only then summed to the gameweek.
That ordering is not cosmetic. Every threshold in FPL is per match, so a
player with two fixtures has two independent shots at an appearance, a
clean sheet and a defensive contribution, and aggregating the inputs first
would quietly halve a double gameweek.

Every component is kept in the output next to `ep_total`. Components are how
this gets debugged and how error attribution works later, and they cost
almost nothing to store.

Two modes:

  live        one upcoming gameweek from the current season's curated data,
              using the saved minutes model
  historical  a whole past season replayed gameweek by gameweek, with the
              minutes model trained only on earlier seasons

Both share one code path. Every trailing input is a backward looking window
evaluated as of the gameweek deadline, so a row for gameweek g is identical
whether or not gameweek g+1 exists on disk. There is a test for that.

Usage:
    python -m models.compose --season 2026-27 --gw 1
    python -m models.compose --season 2025-26 --historical
    python -m models.compose --season 2025-26 --historical --from-gw 6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.curate import CURATED_ROOT
from models import rates
from models.minutes import dataset, features, predict, train
from scoring.rules_2026_27 import ELEMENT_TYPE_TO_POSITION, expected_points

log = logging.getLogger(__name__)

PREDICTIONS_ROOT = Path("data/predictions")

# Components kept alongside ep_total. Anything the scoring module consumes
# is preserved so a surprising ep_total can always be taken apart.
COMPONENT_COLUMNS = [
    "p_start",
    "p_60",
    "p_sub",
    "p_appear",
    "exp_minutes",
    "exp_goals",
    "exp_assists",
    "p_clean_sheet",
    "exp_goals_conceded",
    "exp_saves",
    "p_defcon",
    "exp_bonus",
    "exp_cards",
    "opp_defence_adj",
    "opp_attack_adj",
    "own_defence_adj",
    "defcon_available",
]

FIXTURE_KEYS = ["season", "gw", "player_id", "player_code", "fixture_id", "element_type", "team_id"]


def compose_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Expected points per player per fixture, via the scoring module."""
    records = df.to_dict("records")
    ep = np.empty(len(records))
    for i, row in enumerate(records):
        ep[i] = expected_points(
            ELEMENT_TYPE_TO_POSITION[int(row["element_type"])],
            p_appear=float(row["p_appear"]),
            p_60plus=float(row["p_60"]),
            exp_goals=float(row["exp_goals"]),
            exp_assists=float(row["exp_assists"]),
            p_clean_sheet=float(row["p_clean_sheet"]),
            exp_goals_conceded=float(row["exp_goals_conceded"]),
            exp_saves=float(row["exp_saves"]),
            p_defcon=float(row["p_defcon"]),
            exp_bonus=float(row["exp_bonus"]),
            exp_cards=float(row["exp_cards"]),
        )
    out = df.copy()
    out["ep_total"] = ep
    return out


def aggregate_gameweek(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Sum fixture level expectations into one row per player.

    Expectations are additive across fixtures, so a double gameweek sums.
    Probabilities are summed too, which is intentional: p_appear over two
    fixtures is the expected number of appearances, not a probability, and
    that is exactly what the additive scoring terms consume.
    """
    if fixtures.empty:
        return fixtures

    sum_cols = [c for c in COMPONENT_COLUMNS if c != "defcon_available"] + ["ep_total"]
    agg = {c: "sum" for c in sum_cols}
    grouped = fixtures.groupby(
        ["season", "gw", "player_id", "player_code", "element_type", "team_id"], as_index=False
    ).agg({**agg, "fixture_id": "count", "defcon_available": "max"})
    return grouped.rename(columns={"fixture_id": "n_fixtures"}).sort_values(
        "ep_total", ascending=False
    ).reset_index(drop=True)


def attach_minutes(target: pd.DataFrame, minutes_model: dict) -> pd.DataFrame:
    """Add the minutes model's outputs to a set of rated rows."""
    target = target.copy()
    p_start = train.predict_head(*minutes_model["p_start"], target).clip(
        train.PROB_FLOOR, train.PROB_CEIL
    )
    p_60_given_start = train.predict_head(*minutes_model["p_60_given_start"], target).clip(
        train.PROB_FLOOR, train.PROB_CEIL
    )
    p_sub_given_no_start = train.predict_head(*minutes_model["p_sub"], target).clip(
        train.PROB_FLOOR, train.PROB_CEIL
    )

    target["p_start"] = p_start
    target["p_sub"] = p_sub_given_no_start * (1 - p_start)
    target["p_60"] = p_start * p_60_given_start
    target["p_appear"] = target["p_start"] + target["p_sub"]
    target["exp_minutes"] = train.expected_minutes(
        target["p_start"].to_numpy(),
        target["p_sub"].to_numpy(),
        target["element_type"],
        minutes_model["conditional_minutes"],
    )

    return rates.apply_minutes(target)


def run_gameweek(
    season: str, gw: int, minutes_model: dict, rated: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compose one gameweek from an already rated frame."""
    target = rated[(rated["season"] == season) & (rated["gw"] == gw)]
    if target.empty:
        raise ValueError(f"no rows for {season} gw{gw}")
    fixtures = compose_fixtures(attach_minutes(target, minutes_model))
    keep = FIXTURE_KEYS + COMPONENT_COLUMNS + ["ep_total"]
    fixtures = fixtures[[c for c in keep if c in fixtures.columns]]
    return fixtures, aggregate_gameweek(fixtures)


def write_gameweek(gameweek: pd.DataFrame, fixtures: pd.DataFrame, season: str, gw: int) -> Path:
    out_dir = PREDICTIONS_ROOT / season / f"gw{gw}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "expected_points.parquet"
    gameweek.to_parquet(path, index=False)
    fixtures.to_parquet(out_dir / "expected_points_fixtures.parquet", index=False)
    return path


# --------------------------------------------------------------------------
# Frame construction for the two modes
# --------------------------------------------------------------------------


def build_full_frame(
    season: str, gw: int | None, mode: str, curated_root: Path = CURATED_ROOT
) -> pd.DataFrame:
    """Every played row plus, in live mode, the unplayed target gameweek."""
    history = dataset.build(curated_root=curated_root)
    if mode == "live":
        upcoming = dataset.build_upcoming(season, gw, curated_root)
        # A live gameweek has not happened, so anything already stored for it
        # or later is the future and is dropped rather than trusted.
        history = history[~((history["season"] == season) & (history["gw"] >= gw))]
        frame = pd.concat([history, upcoming], ignore_index=True)
    else:
        frame = history
    return features.add_features(frame, curated_root=curated_root)


def load_minutes_model(
    season: str, mode: str, full: pd.DataFrame, version: str | None = None
) -> dict:
    """Saved model for live mode, walk forward model for historical mode."""
    if mode == "live":
        saved = predict.load_model(version)
        return {
            **saved["heads"],
            "conditional_minutes": saved["metadata"]["conditional_minutes"],
            "train_seasons": saved["metadata"]["train_seasons"],
        }
    model = train.walk_forward_model_for(full, season)
    log.info("historical mode: minutes model trained on %s", model["train_seasons"])
    return model


def run(
    season: str,
    gw: int | None = None,
    historical: bool = False,
    from_gw: int = 1,
    version: str | None = None,
    curated_root: Path = CURATED_ROOT,
    team_source: str = rates.DEFAULT_TEAM_SOURCE,
) -> dict[int, pd.DataFrame]:
    mode = "historical" if historical else "live"
    full = build_full_frame(season, gw, mode, curated_root)
    minutes_model = load_minutes_model(season, mode, full, version)

    # Priors come from a window that closes before the target, so replaying a
    # season cannot let a later gameweek inform an earlier one. In historical
    # mode that window is every earlier season, matching how the minutes
    # model is trained. Trailing form is backward looking regardless, so the
    # rates are computed once over the whole frame rather than per gameweek.
    history = (
        full[full["season"] < season] if historical else full[full["minutes"].notna()]
    )
    rated = rates.add_rates(
        full, history=history, curated_root=curated_root, team_source=team_source
    )

    if historical:
        target_gws = sorted(
            int(g)
            for g in full.loc[full["season"] == season, "gw"].dropna().unique()
            if g >= from_gw
        )
    else:
        target_gws = [gw]

    results: dict[int, pd.DataFrame] = {}
    for target in target_gws:
        fixtures, gameweek = run_gameweek(season, target, minutes_model, rated)
        write_gameweek(gameweek, fixtures, season, target)
        results[target] = gameweek
        log.info(
            "%s gw%d: %d players, mean ep %.2f, max ep %.2f",
            season,
            target,
            len(gameweek),
            gameweek["ep_total"].mean(),
            gameweek["ep_total"].max(),
        )
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Expected points for a gameweek")
    ap.add_argument("--season", required=True)
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--historical", action="store_true", help="replay a whole past season")
    ap.add_argument("--from-gw", type=int, default=1, help="historical mode start gameweek")
    ap.add_argument("--version", default=None, help="minutes model version for live mode")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument(
        "--team-source",
        default=rates.DEFAULT_TEAM_SOURCE,
        choices=sorted(rates.TEAM_SOURCE_CHAIN),
        help="where team goal expectations and clean sheets come from",
    )
    args = ap.parse_args()

    if not args.historical and args.gw is None:
        raise SystemExit("pass --gw for live mode, or --historical for a whole season")

    results = run(
        args.season, args.gw, args.historical, args.from_gw, args.version,
        team_source=args.team_source,
    )

    last_gw = max(results)
    players = pd.read_parquet(CURATED_ROOT / args.season / "players.parquet")
    named = results[last_gw].merge(
        players[["player_id", "web_name", "team_short"]], on="player_id", how="left"
    )
    pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    named["pos"] = named["element_type"].map(pos)
    show = named.head(args.top)[
        ["web_name", "team_short", "pos", "n_fixtures", "exp_minutes", "exp_goals",
         "p_clean_sheet", "exp_bonus", "ep_total"]
    ]
    print(f"\n{args.season} gw{last_gw}, top {args.top} by expected points\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
