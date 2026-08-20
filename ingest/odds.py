"""Bookmaker odds from football-data.co.uk, de-vigged and joined to fixtures.

The betting market is the external bar. It aggregates more information than
this project will ever have, it is continuously corrected by people with
money at stake, and it is free. If a trained model cannot beat odds from
before the deadline, that is the finding, not an embarrassment.

Source: football-data.co.uk publishes one CSV per season with per match
results and odds from a dozen books. Only the CLOSING average columns are
used, AvgC*, because closing odds are the sharpest published price and the
opening ones move before kickoff. Columns are stable across seasons and
have no missing values in 2021-22 through 2025-26.

De-vigging uses Shin's method via penaltyblog. Bookmaker prices sum to more
than 1, and the excess has to be removed before they are probabilities.
Basic normalisation spreads the margin proportionally, which is known to
overprice favourites, and Shin instead models the margin as protection
against insider trading and removes proportionally more from longshots.
That matters here because longshot distortion is worst exactly where FPL
cares most, on the clean sheet and heavy-win end of the distribution.

Team goal expectations are recovered by fitting an independent Poisson pair
to the de-vigged 1X2 and over/under 2.5 markets. Independence is wrong,
goals are correlated, and Dixon-Coles in task 2 is what fixes it. It is
good enough to establish a bar.

Usage:
    python -m ingest.odds --all
    python -m ingest.odds --seasons 2025-26
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

from ingest.curate import CURATED_ROOT

log = logging.getLogger(__name__)

EXTERNAL_ROOT = Path("data/external/football-data")
CLUB_NAME_MAP = Path("mapping/club_names.csv")
BASE = "https://www.football-data.co.uk/mmz4281"
USER_AGENT = "fpl-research/0.1 (personal project)"
TIMEOUT = 60

# football-data numbers its seasons by the two calendar years involved.
SEASON_CODES = {
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}
DEFAULT_SEASONS = list(SEASON_CODES)

# Closing average across books. Closing is the sharpest published price.
ODDS_1X2 = {"home": "AvgCH", "draw": "AvgCD", "away": "AvgCA"}
ODDS_OU = {"over": "AvgC>2.5", "under": "AvgC<2.5"}
OU_LINE = 2.5

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *ODDS_1X2.values(), *ODDS_OU.values()]

# Goal expectations are searched in a sane football range. Nothing in the
# Premier League has a true expectation outside this.
GOALS_BOUNDS = (0.15, 5.0)
MAX_GOALS = 10


def load_club_map(source: str = "football-data", path: Path = CLUB_NAME_MAP) -> dict[str, str]:
    """External club spellings to the FPL team_name they correspond to."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["source"] == source]
    return dict(zip(df["external_name"], df["fpl_name"]))


def ensure_csv(season: str, root: Path = EXTERNAL_ROOT) -> Path:
    """Download a season's CSV unless it is already cached."""
    code = SEASON_CODES.get(season)
    if code is None:
        raise ValueError(f"no football-data season code for {season}")
    path = root / season / "E0.csv"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{code}/E0.csv"
    log.info("GET %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    log.info("cached %s (%d bytes)", path, path.stat().st_size)
    return path


def load_raw(season: str, root: Path = EXTERNAL_ROOT) -> pd.DataFrame:
    """One season of match odds, with club names normalised to FPL spellings."""
    path = ensure_csv(season, root)
    # football-data ships latin-1, not utf-8
    df = pd.read_csv(path, encoding="latin-1")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{season} odds CSV missing expected columns: {missing}")

    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    club_map = load_club_map()
    df["home_name"] = df["HomeTeam"].replace(club_map)
    df["away_name"] = df["AwayTeam"].replace(club_map)
    df["match_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    df["season"] = season
    return df


# --------------------------------------------------------------------------
# De-vigging and goal expectations
# --------------------------------------------------------------------------


def devig(odds: list[float] | np.ndarray, method: str = "shin") -> tuple[np.ndarray, float]:
    """Turn bookmaker prices into probabilities. Returns probabilities and margin."""
    from penaltyblog.implied import ImpliedMethod, calculate_implied

    result = calculate_implied(list(odds), method=ImpliedMethod(method))
    return np.asarray(result.probabilities, dtype=float), float(result.margin)


def _outcome_probabilities(home_xg: float, away_xg: float) -> tuple[float, float, float, float]:
    """P(home win), P(draw), P(away win) and P(over 2.5) under independent Poisson."""
    h = poisson.pmf(np.arange(MAX_GOALS + 1), home_xg)
    a = poisson.pmf(np.arange(MAX_GOALS + 1), away_xg)
    grid = np.outer(h, a)
    home = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    away = float(np.triu(grid, 1).sum())
    totals = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    over = float(grid[totals > OU_LINE].sum())
    return home, draw, away, over


def goal_expectations(p_home: float, p_draw: float, p_away: float, p_over: float) -> tuple[float, float]:
    """Recover team goal expectations that best reproduce the de-vigged markets.

    Fits an independent Poisson pair to four target probabilities. The 1X2
    market fixes the balance between the sides and the over/under market
    fixes the total, so together they identify both expectations.
    """
    targets = np.array([p_home, p_draw, p_away, p_over])

    def loss(params):
        home_xg, away_xg = np.exp(params)
        got = np.array(_outcome_probabilities(home_xg, away_xg))
        return float(((got - targets) ** 2).sum())

    best = minimize(loss, x0=np.log([1.5, 1.2]), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 500})
    home_xg, away_xg = np.exp(best.x)
    lo, hi = GOALS_BOUNDS
    return float(np.clip(home_xg, lo, hi)), float(np.clip(away_xg, lo, hi))


def build_match_odds(season: str, root: Path = EXTERNAL_ROOT) -> pd.DataFrame:
    """De-vigged probabilities and goal expectations per match."""
    raw = load_raw(season, root)
    rows = []
    for r in raw.to_dict("records"):
        try:
            probs_1x2, margin_1x2 = devig([r[ODDS_1X2["home"]], r[ODDS_1X2["draw"]], r[ODDS_1X2["away"]]])
            probs_ou, margin_ou = devig([r[ODDS_OU["over"]], r[ODDS_OU["under"]]])
        except Exception as exc:  # a malformed price should not kill a season
            log.warning("%s %s v %s: could not de-vig (%s)", season, r["home_name"], r["away_name"], exc)
            continue
        home_xg, away_xg = goal_expectations(*probs_1x2, probs_ou[0])
        rows.append(
            {
                "season": season,
                "match_date": r["match_date"],
                "home_name": r["home_name"],
                "away_name": r["away_name"],
                "p_home_win": probs_1x2[0],
                "p_draw": probs_1x2[1],
                "p_away_win": probs_1x2[2],
                "p_over_2_5": probs_ou[0],
                "margin_1x2": margin_1x2,
                "margin_ou": margin_ou,
                "home_xg": home_xg,
                "away_xg": away_xg,
                # A clean sheet is the opponent failing to score at all
                "p_home_cs": float(np.exp(-away_xg)),
                "p_away_cs": float(np.exp(-home_xg)),
                "home_goals": r["FTHG"],
                "away_goals": r["FTAG"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Joining to fixtures
# --------------------------------------------------------------------------


def team_names(season: str, curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    players = pd.read_parquet(curated_root / season / "players.parquet")
    return players[["team_id", "team_name"]].drop_duplicates()


def join_to_fixtures(
    odds: pd.DataFrame, season: str, curated_root: Path = CURATED_ROOT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach fixture_id to each priced match. Returns joined rows and misses.

    Matched on club names plus kickoff date rather than on date alone,
    because a postponed fixture keeps its teams but moves its date, and
    matched on both sides' names so a double header cannot cross over.
    """
    fixtures = pd.read_parquet(curated_root / season / "fixtures.parquet")
    names = team_names(season, curated_root)
    fx = (
        fixtures.merge(names.rename(columns={"team_id": "team_h", "team_name": "home_name"}), on="team_h", how="left")
        .merge(names.rename(columns={"team_id": "team_a", "team_name": "away_name"}), on="team_a", how="left")
    )
    fx["match_date"] = pd.to_datetime(fx["kickoff_time"], utc=True).dt.date

    joined = fx.merge(
        odds, on=["season", "home_name", "away_name", "match_date"], how="left", validate="one_to_one"
    )
    misses = joined[joined["home_xg"].isna()]

    if not misses.empty:
        # A fixture played a day either side of the listed date is the same
        # match, so a near miss is repaired rather than dropped.
        repaired = _repair_by_nearby_date(misses, odds)
        if not repaired.empty:
            joined = joined.set_index("fixture_id")
            joined.update(repaired.set_index("fixture_id"))
            joined = joined.reset_index()
            misses = joined[joined["home_xg"].isna()]

    return joined, misses


def _repair_by_nearby_date(misses: pd.DataFrame, odds: pd.DataFrame, days: int = 1) -> pd.DataFrame:
    """Rejoin unmatched fixtures allowing the date to differ by a day."""
    out = []
    for row in misses.to_dict("records"):
        candidates = odds[
            (odds["home_name"] == row["home_name"]) & (odds["away_name"] == row["away_name"])
        ]
        if candidates.empty:
            continue
        gap = candidates["match_date"].apply(lambda d: abs((d - row["match_date"]).days))
        best = candidates[gap <= days]
        if len(best) == 1:
            merged = {**row}
            for col in best.columns:
                if col not in ("season", "home_name", "away_name", "match_date"):
                    merged[col] = best.iloc[0][col]
            out.append(merged)
    return pd.DataFrame(out)


ODDS_COLUMNS = [
    "season", "gw", "fixture_id", "home_name", "away_name",
    "p_home_win", "p_draw", "p_away_win", "p_over_2_5",
    "home_xg", "away_xg", "p_home_cs", "p_away_cs",
    "margin_1x2", "margin_ou",
]


def run_season(season: str, root: Path = EXTERNAL_ROOT, curated_root: Path = CURATED_ROOT) -> dict:
    odds = build_match_odds(season, root)
    joined, misses = join_to_fixtures(odds, season, curated_root)

    coverage = 1.0 - len(misses) / len(joined) if len(joined) else 0.0
    out_path = curated_root / season / "fixture_odds.parquet"
    joined[ODDS_COLUMNS].to_parquet(out_path, index=False)

    log.info(
        "%s: %d fixtures, %.1f%% priced, mean 1X2 margin %.3f -> %s",
        season, len(joined), 100 * coverage, joined["margin_1x2"].mean(), out_path,
    )
    if not misses.empty:
        for m in misses.to_dict("records"):
            log.warning(
                "  unpriced: %s v %s on %s", m["home_name"], m["away_name"], m["match_date"]
            )
    return {"season": season, "fixtures": len(joined), "coverage": coverage,
            "misses": misses[["home_name", "away_name", "match_date"]].to_dict("records"),
            "path": out_path}


def run(seasons: list[str], root: Path = EXTERNAL_ROOT, curated_root: Path = CURATED_ROOT) -> list[dict]:
    return [run_season(s, root, curated_root) for s in seasons]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest and de-vig bookmaker odds")
    ap.add_argument("--seasons", nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help=f"ingest {DEFAULT_SEASONS}")
    args = ap.parse_args()

    seasons = DEFAULT_SEASONS if args.all else args.seasons
    if not seasons:
        raise SystemExit("pass --seasons or --all")

    for result in run(seasons):
        print(f"{result['season']}: {100 * result['coverage']:.1f}% of {result['fixtures']} fixtures priced")


if __name__ == "__main__":
    main()
