"""Dixon-Coles team strength, fitted walk forward.

Replaces the v0 trailing clean sheet and concession rates with a proper
goals model. Two teams, an attack and a defence parameter each, a home
advantage, and a correlation correction for low scoring games where the
independent Poisson is known to be wrong. penaltyblog does the fitting.

Time decay weights recent matches more, with xi = 0.0018 per day. That is
the literature standard value, roughly a half life of a year, and it is
NOT tuned here. Tuning a decay constant against 2025-26 would spend the
frozen evaluation season on a hyperparameter, which is exactly what the
gating policy exists to prevent. If it ever needs choosing between
candidates, choose on 2021-22 through 2024-25 log loss alone.

Walk forward means what it says: the fit used to predict gameweek g sees
only matches that kicked off before the first fixture of gameweek g,
including earlier seasons, decayed. A full season is 38 refits and takes a
few seconds, so there is no reason to cut the corner of fitting once.

Promoted teams
--------------
penaltyblog raises ValueError on a team it has not seen, so promoted sides
are wrapped rather than left to crash. They enter with the mean attack and
defence of the three weakest teams in the current fit. That is not a guess:
fitting 2024-25 puts Southampton, Leicester and Ipswich as the three
weakest sides by attack minus defence, and those were precisely that
season's three promoted clubs. Promoted teams perform like the relegation
tier until they prove otherwise, and after a handful of matches the fit has
real evidence and the prior stops mattering.

Usage:
    python -m models.team.dixon_coles --season 2025-26
    python -m models.team.dixon_coles --all
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.curate import CURATED_ROOT

log = logging.getLogger(__name__)

# Per day decay. Literature standard, deliberately untuned.
XI_DECAY = 0.0018
MAX_GOALS = 15
PROMOTED_TIER_SIZE = 3
MIN_TRAINING_MATCHES = 60

OUTPUT_NAME = "team_model.parquet"
OUTPUT_COLUMNS = [
    "season", "gw", "fixture_id", "home_name", "away_name",
    "home_xg", "away_xg", "p_home_cs", "p_away_cs",
    "p_home_win", "p_draw", "p_away_win",
    "home_is_promoted", "away_is_promoted", "n_training_matches",
]


def load_matches(seasons: list[str], curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Every completed match across seasons, keyed by club name.

    Club name rather than team_id, because FPL renumbers teams every August
    and a model fitted across seasons has to know that Arsenal is Arsenal.
    """
    frames = []
    for season in seasons:
        fixtures_path = curated_root / season / "fixtures.parquet"
        players_path = curated_root / season / "players.parquet"
        if not fixtures_path.exists() or not players_path.exists():
            continue
        fixtures = pd.read_parquet(fixtures_path)
        names = pd.read_parquet(players_path)[["team_id", "team_name"]].drop_duplicates()
        df = fixtures.merge(
            names.rename(columns={"team_id": "team_h", "team_name": "home_name"}), on="team_h", how="left"
        ).merge(
            names.rename(columns={"team_id": "team_a", "team_name": "away_name"}), on="team_a", how="left"
        )
        frames.append(
            df[[
                "season", "gw", "fixture_id", "kickoff_time",
                "home_name", "away_name", "team_h_score", "team_a_score",
            ]]
        )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True)
    return out.sort_values("kickoff_time", kind="stable").reset_index(drop=True)


def fit(matches: pd.DataFrame, xi: float = XI_DECAY):
    """Fit Dixon-Coles to completed matches, weighting recent ones more."""
    from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights

    played = matches.dropna(subset=["team_h_score", "team_a_score"])
    weights = dixon_coles_weights(played["kickoff_time"].dt.date, xi)
    model = DixonColesGoalModel(
        played["team_h_score"].astype(int),
        played["team_a_score"].astype(int),
        played["home_name"],
        played["away_name"],
        weights=weights,
    )
    model.fit()
    return model, len(played)


def promoted_prior(params: dict) -> tuple[float, float]:
    """Attack and defence for a club the fit has never seen.

    The mean of the weakest tier in the current fit, where weakest is by
    attack minus defence.
    """
    attack = {k[len("attack_"):]: v for k, v in params.items() if k.startswith("attack_")}
    defence = {k[len("defence_"):]: v for k, v in params.items() if k.startswith("defence_")}
    if not attack:
        return 0.0, 0.0
    strength = {t: attack[t] - defence.get(t, 0.0) for t in attack}
    weakest = sorted(strength, key=strength.get)[:PROMOTED_TIER_SIZE]
    return (
        float(np.mean([attack[t] for t in weakest])),
        float(np.mean([defence[t] for t in weakest])),
    )


def predict_fixture(params: dict, home: str, away: str) -> dict:
    """Goal expectations and outcome probabilities for one fixture.

    Lambdas are computed from the parameters directly rather than through
    the library's lookup, because that is what allows a promoted side to be
    substituted in without the fit having seen it.
    """
    from penaltyblog.models import create_dixon_coles_grid

    prior_attack, prior_defence = promoted_prior(params)
    home_new = f"attack_{home}" not in params
    away_new = f"attack_{away}" not in params

    attack_home = params.get(f"attack_{home}", prior_attack)
    defence_home = params.get(f"defence_{home}", prior_defence)
    attack_away = params.get(f"attack_{away}", prior_attack)
    defence_away = params.get(f"defence_{away}", prior_defence)

    home_lambda = float(np.exp(attack_home + defence_away + params["home_advantage"]))
    away_lambda = float(np.exp(attack_away + defence_home))

    # The low score correction is only defined for a range of rho that
    # depends on the lambdas, and a fit on thin or degenerate data can land
    # outside it. Falling back to an uncorrected grid is much better than
    # raising: the correction is what is unreliable in that case, not the
    # goal expectations.
    try:
        grid_obj = create_dixon_coles_grid(home_lambda, away_lambda, params["rho"], MAX_GOALS)
    except ValueError as exc:
        log.warning(
            "rho=%.3f out of bounds for %s v %s (%s), using an uncorrected grid",
            params["rho"], home, away, exc,
        )
        grid_obj = create_dixon_coles_grid(home_lambda, away_lambda, 0.0, MAX_GOALS)

    grid = np.asarray(grid_obj.grid, dtype=float)
    grid = grid / grid.sum()

    return {
        "home_xg": home_lambda,
        "away_xg": away_lambda,
        # A clean sheet is the opponent's marginal probability of zero, taken
        # from the corrected grid rather than a raw Poisson zero, because the
        # correction is largest exactly at 0-0 and 1-0.
        "p_home_cs": float(grid[:, 0].sum()),
        "p_away_cs": float(grid[0, :].sum()),
        "p_home_win": float(np.tril(grid, -1).sum()),
        "p_draw": float(np.trace(grid)),
        "p_away_win": float(np.triu(grid, 1).sum()),
        "home_is_promoted": home_new,
        "away_is_promoted": away_new,
    }


def walk_forward(
    season: str,
    seasons: list[str] | None = None,
    curated_root: Path = CURATED_ROOT,
    xi: float = XI_DECAY,
) -> pd.DataFrame:
    """Refit before every gameweek, predict that gameweek, never look ahead."""
    if seasons is None:
        seasons = sorted({p.parent.name for p in curated_root.glob("*/fixtures.parquet")})
    matches = load_matches(seasons, curated_root)
    if matches.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target = matches[matches["season"] == season]
    rows: list[dict] = []
    for gw in sorted(target["gw"].dropna().unique()):
        fixtures = target[target["gw"] == gw]
        cutoff = fixtures["kickoff_time"].min()
        history = matches[matches["kickoff_time"] < cutoff].dropna(
            subset=["team_h_score", "team_a_score"]
        )
        if len(history) < MIN_TRAINING_MATCHES:
            log.info("%s gw%d: only %d prior matches, skipping", season, int(gw), len(history))
            continue

        model, n_train = fit(history, xi)
        params = model.get_params()
        for fx in fixtures.to_dict("records"):
            prediction = predict_fixture(params, fx["home_name"], fx["away_name"])
            rows.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "fixture_id": fx["fixture_id"],
                    "home_name": fx["home_name"],
                    "away_name": fx["away_name"],
                    "n_training_matches": n_train,
                    **prediction,
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def run_season(season: str, curated_root: Path = CURATED_ROOT, xi: float = XI_DECAY) -> Path:
    started = time.perf_counter()
    df = walk_forward(season, curated_root=curated_root, xi=xi)
    path = curated_root / season / OUTPUT_NAME
    df.to_parquet(path, index=False)
    log.info(
        "%s: %d fixtures modelled in %.1fs (%d gameweeks) -> %s",
        season, len(df), time.perf_counter() - started, df["gw"].nunique(), path,
    )
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fit Dixon-Coles walk forward")
    ap.add_argument("--season", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--xi", type=float, default=XI_DECAY)
    args = ap.parse_args()

    if args.all:
        seasons = sorted({p.parent.name for p in CURATED_ROOT.glob("*/fixtures.parquet")})
    elif args.season:
        seasons = [args.season]
    else:
        raise SystemExit("pass --season or --all")

    for season in seasons:
        run_season(season, xi=args.xi)


if __name__ == "__main__":
    main()
