"""Append-only record of how each candidate did in the real 2026-27 season.

2025-26 is frozen, and every run against it spends a little of its value.
This log is the antidote. The live season is the only evaluation data that
nothing was ever tuned on, cannot be re-run, and accumulates one honest
gameweek a week.

The discipline that makes it worth anything: a row may only be written from
a predictions file that existed BEFORE the deadline. Predictions
regenerated after the fact would use snapshots taken after team news broke,
which is the exact leak this whole project is built to avoid, and it would
be undetectable a year from now. So the timestamp of the prediction file is
recorded alongside the result, and a file modified after its deadline is
refused rather than logged.

No analysis here. This is a paper trail, and it is deliberately dull. The
report regenerates from the log and says nothing a table cannot.

Usage:
    python -m eval.live_log --gw 1
    python -m eval.live_log --gw 1 --candidates compose_v0 last5
    python -m eval.live_log --report
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingest.curate import CURATED_ROOT
from optimise import milp
from scoring.replay import realised_gameweek_points

log = logging.getLogger(__name__)

LIVE_SEASON = "2026-27"
LOG_PATH = Path("data/eval/live_2026-27.parquet")
REPORT_PATH = Path("experiments/reports/live_2026-27.md")
PREDICTIONS_ROOT = Path("data/predictions")

# Candidates tracked live. odds_only and naive_minutes_ep are omitted on
# purpose: this log records what was actually decided each week, and only
# these have a prediction file written before the deadline.
LIVE_CANDIDATES = ["compose_v0", "last5", "price", "ep_next"]

LOG_COLUMNS = [
    "season",
    "gw",
    "candidate",
    "xi_points",
    "captain_points",
    "total_points",
    "prediction_file",
    "prediction_written_utc",
    "logged_utc",
]


class DeadlineError(RuntimeError):
    """Raised when a prediction file cannot be trusted as pre deadline."""


def deadline(season: str, gw: int, curated_root: Path = CURATED_ROOT) -> pd.Timestamp:
    """First kickoff of the gameweek, the moment a team stops being changeable."""
    fixtures = pd.read_parquet(curated_root / season / "fixtures.parquet")
    rows = fixtures[fixtures["gw"] == gw]
    if rows.empty:
        raise ValueError(f"no fixtures for {season} gw{gw}")
    return pd.to_datetime(rows["kickoff_time"].min(), utc=True)


def prediction_file(season: str, gw: int, candidate: str) -> Path:
    """Where a candidate's pre deadline expected points are expected to live."""
    name = "expected_points.parquet" if candidate == "compose_v0" else f"{candidate}.parquet"
    return PREDICTIONS_ROOT / season / f"gw{gw}" / name


def check_pre_deadline(path: Path, season: str, gw: int, curated_root: Path = CURATED_ROOT) -> pd.Timestamp:
    """Refuse a prediction file that was written after the gameweek started."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, nothing was predicted for this gameweek")
    written = pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
    cutoff = deadline(season, gw, curated_root)
    if written > cutoff:
        raise DeadlineError(
            f"{path} was written {written:%Y-%m-%d %H:%M} UTC, after the {season} gw{gw} "
            f"deadline of {cutoff:%Y-%m-%d %H:%M} UTC. A prediction made after kickoff is "
            "worthless as evidence. Regenerate it before the deadline or leave the "
            "gameweek unlogged."
        )
    return written


def score_candidate(
    season: str, gw: int, candidate: str, curated_root: Path = CURATED_ROOT
) -> dict:
    """Optimise a fresh pick from the stored prediction and score what it did."""
    path = prediction_file(season, gw, candidate)
    written = check_pre_deadline(path, season, gw, curated_root)

    ep = pd.read_parquet(path)
    pool = milp.pool_from_ep(ep, season, gw, curated_root)
    solution = milp.solve(pool, [gw])

    realised = realised_gameweek_points(season, gw, curated_root)[["player_id", "realised"]]
    squad = solution.squad.merge(realised, on="player_id", how="left")
    squad["realised"] = squad["realised"].fillna(0.0)

    xi_points = float(squad.loc[squad["in_xi"], "realised"].sum())
    captain_points = float(squad.loc[squad["is_captain"], "realised"].sum())
    return {
        "season": season,
        "gw": gw,
        "candidate": candidate,
        "xi_points": xi_points,
        "captain_points": captain_points,
        "total_points": xi_points + captain_points,
        "prediction_file": str(path),
        "prediction_written_utc": written,
        "logged_utc": pd.Timestamp.now(tz="UTC"),
    }


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_parquet(path)


def append(rows: list[dict], path: Path = LOG_PATH) -> Path:
    """Append rows, refusing to rewrite a gameweek that is already recorded."""
    existing = load_log(path)
    new = pd.DataFrame(rows)
    if not existing.empty:
        already = set(zip(existing["gw"], existing["candidate"]))
        clash = [r for r in rows if (r["gw"], r["candidate"]) in already]
        if clash:
            raise ValueError(
                f"already logged: {[(c['gw'], c['candidate']) for c in clash]}. The log is "
                "append only, so delete the row deliberately if it really must change."
            )
    out = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    path.parent.mkdir(parents=True, exist_ok=True)
    out[LOG_COLUMNS].to_parquet(path, index=False)
    log.info("appended %d rows to %s (%d total)", len(new), path, len(out))
    return path


def write_report(path: Path = LOG_PATH, report_path: Path = REPORT_PATH) -> Path:
    """Regenerate the report from the log. No analysis, just the record."""
    df = load_log(path)
    lines: list[str] = []
    w = lines.append
    w(f"# Live tracking, {LIVE_SEASON}")
    w("")
    w("Realised points of each candidate's optimised fresh pick, gameweek by gameweek, "
      "from predictions written before the deadline. This is the only evaluation data in "
      "the project that nothing has been tuned against, and it cannot be re-run. Treat it "
      "as the honest test and the frozen 2025-26 numbers as the development one.")
    w("")

    if df.empty:
        w("No gameweeks logged yet. The season has not started.")
        w("")
        w("To log one, after the gameweek has finished and been snapshotted:")
        w("")
        w("```")
        w("python -m eval.live_log --gw 1")
        w("```")
        w("")
        w("Predictions must already exist from before the deadline. The logger refuses a "
          "file written after the first kickoff.")
    else:
        w(f"{df['gw'].nunique()} gameweeks logged, {len(df)} rows.")
        w("")
        w("| GW | " + " | ".join(sorted(df["candidate"].unique())) + " |")
        w("|---:|" + "|".join(["---:"] * df["candidate"].nunique()) + "|")
        pivot = df.pivot_table(index="gw", columns="candidate", values="total_points")
        for gw, row in pivot.iterrows():
            cells = " | ".join(
                f"{row[c]:.0f}" if pd.notna(row.get(c)) else "" for c in sorted(df["candidate"].unique())
            )
            w(f"| {int(gw)} | {cells} |")
        w("")
        w("Running means:")
        w("")
        w("| Candidate | Gameweeks | Mean points | Mean captain |")
        w("|---|---:|---:|---:|")
        for name, grp in df.groupby("candidate"):
            w(
                f"| {name} | {len(grp)} | {grp['total_points'].mean():.2f} | "
                f"{grp['captain_points'].mean():.2f} |"
            )
        w("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", report_path)
    return report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Append a finished gameweek to the live log")
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--season", default=LIVE_SEASON)
    ap.add_argument("--candidates", nargs="+", default=LIVE_CANDIDATES)
    ap.add_argument("--report", action="store_true", help="regenerate the report only")
    args = ap.parse_args()

    if not args.report:
        if args.gw is None:
            raise SystemExit("pass --gw to log a gameweek, or --report to regenerate")
        rows = []
        for candidate in args.candidates:
            try:
                rows.append(score_candidate(args.season, args.gw, candidate))
            except (FileNotFoundError, DeadlineError) as exc:
                log.warning("skipping %s: %s", candidate, exc)
        if rows:
            append(rows)
        else:
            log.warning("nothing logged for gw%d", args.gw)

    print(f"wrote {write_report()}")


if __name__ == "__main__":
    main()
