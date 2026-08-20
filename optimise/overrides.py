"""Manual overrides: lock a player in, ban one out, force the armband.

An override is the human overruling the model, usually because they know
something it does not. A press conference, a rumoured rotation, a hunch
about a cup game. That makes overrides the most likely source of an
impossible problem, so everything here is about failing with a sentence
that names the clash rather than a solver status code.

The structural checks in this module need no solver. They catch the clashes
that can be proved from counting alone, which is most of them, and they are
what lets the failure message say "four locked defenders come from Arsenal"
instead of "kInfeasible".

File format, all lists of FPL player codes:

    lock: [123456]          must appear in the starting eleven
    ban: [654321]           must not appear in the squad at all
    force_captain: 123456   must wear the armband

Codes rather than names, because names are ambiguous and are renumbered
per season while codes are stable. An absent file means no overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

OVERRIDES_PATH = Path("overrides.yaml")
EMPTY: dict = {"lock": [], "ban": [], "force_captain": None}


def load_overrides(path: Path = OVERRIDES_PATH) -> dict:
    """Read overrides from YAML. A missing file means no overrides."""
    import yaml

    if not path.exists():
        return dict(EMPTY)
    raw = yaml.safe_load(path.read_text()) or {}
    overrides = {
        "lock": [int(c) for c in (raw.get("lock") or [])],
        "ban": [int(c) for c in (raw.get("ban") or [])],
        "force_captain": int(raw["force_captain"]) if raw.get("force_captain") else None,
    }
    both = set(overrides["lock"]) & set(overrides["ban"])
    if both:
        raise ValueError(f"player codes are both locked and banned: {sorted(both)}")
    if overrides["lock"] or overrides["ban"] or overrides["force_captain"]:
        log.info(
            "overrides: %d locked, %d banned, captain %s",
            len(overrides["lock"]),
            len(overrides["ban"]),
            overrides["force_captain"] or "free",
        )
    return overrides


def structural_conflict(
    pool: pd.DataFrame,
    overrides: dict,
    quota: dict,
    names: dict,
    xi_size: int,
    xi_exact_gkp: int,
    max_per_club: int,
    budget: float | None,
) -> str | None:
    """Name the clash an override causes, or None if counting cannot prove one.

    Ordered from most specific to least, so the message points at the thing
    a human would have to change.
    """
    by_code = pool.set_index(pool["player_code"].astype(int))
    lock = [int(c) for c in (overrides.get("lock") or [])]
    ban = [int(c) for c in (overrides.get("ban") or [])]
    forced = overrides.get("force_captain")

    referenced = lock + ban + ([int(forced)] if forced else [])
    unknown = sorted({c for c in referenced if c not in by_code.index})
    if unknown:
        return f"override references player codes not in the pool: {unknown}"

    if forced and int(forced) in ban:
        return f"force_captain {forced} is also in the ban list"

    if len(lock) > xi_size:
        return f"{len(lock)} players are locked into an eleven of {xi_size}"

    if lock:
        locked = by_code.loc[lock]
        counts = locked["element_type"].value_counts().to_dict()
        # A locked keeper surplus is the tighter of the two limits, so it is
        # reported against the eleven rather than the squad.
        if counts.get(1, 0) > xi_exact_gkp:
            return (
                f"{counts[1]} keepers are locked into the eleven, which allows exactly "
                f"{xi_exact_gkp}"
            )
        for et, allowed in quota.items():
            if counts.get(et, 0) > allowed:
                return (
                    f"{counts[et]} locked {names[et]} exceeds the squad quota of {allowed}"
                )
        club_counts = locked["team_id"].value_counts()
        over = club_counts[club_counts > max_per_club]
        if not over.empty:
            club = pool[pool["team_id"] == over.index[0]]["team_short"].iloc[0]
            return (
                f"{int(over.iloc[0])} locked players come from {club}, over the limit of "
                f"{max_per_club} per club"
            )
        if budget is not None and float(locked["price"].sum()) > budget:
            return (
                f"locked players alone cost {locked['price'].sum():.1f}m, over the "
                f"{budget:.1f}m budget"
            )

    if ban:
        available = pool[~pool["player_code"].astype(int).isin(ban)]
        for et, needed in quota.items():
            have = int((available["element_type"] == et).sum())
            if have < needed:
                return (
                    f"only {have} {names[et]} remain after the ban list, and a squad needs "
                    f"{needed}"
                )

    return None
