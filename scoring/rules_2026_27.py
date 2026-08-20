"""Single source of truth for FPL 2026/27 scoring.

Every points calculation in this project goes through here. When the rules
change, this file is the only thing that changes.

Verified against the official 2026/27 rules as of August 2026. Key points:
  - Goal values by position: GKP 10, DEF 6, MID 5, FWD 4
  - Defensive contributions (introduced 2025/26, unchanged 2026/27):
      DEF   +2 at 10 CBIT   (clearances, blocks, interceptions, tackles)
      MID   +2 at 12 CBIRT  (CBIT plus ball recoveries)
      FWD   +2 at 12 CBIRT
      capped at +2 per player per match, not per threshold multiple
  - The 60 minute threshold excludes stoppage time
  - BPS changed for 2026/27 but bonus points themselves are still 1/2/3,
    so bonus enters this module as an input, not something it derives

This module deliberately does NOT model BPS. Bonus is predicted upstream
and passed in. Keep it that way, otherwise a BPS rule change forces a
rewrite of the scoring core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SEASON = "2026-27"


class Position(str, Enum):
    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


# FPL element_type in the API maps to position like this
ELEMENT_TYPE_TO_POSITION = {
    1: Position.GKP,
    2: Position.DEF,
    3: Position.MID,
    4: Position.FWD,
}

GOAL_POINTS = {
    Position.GKP: 10,
    Position.DEF: 6,
    Position.MID: 5,
    Position.FWD: 4,
}

CLEAN_SHEET_POINTS = {
    Position.GKP: 4,
    Position.DEF: 4,
    Position.MID: 1,
    Position.FWD: 0,
}

ASSIST_POINTS = 3
APPEARANCE_SHORT = 1  # played 1 to 59 minutes
APPEARANCE_LONG = 2  # played 60 or more minutes
MINUTES_THRESHOLD = 60

SAVES_PER_POINT = 3
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2
GOALS_CONCEDED_PER_PENALTY = 2  # -1 point per 2 conceded, GKP and DEF only

DEFCON_POINTS = 2
DEFCON_THRESHOLD = {
    Position.GKP: None,  # goalkeepers do not earn defensive contribution points
    Position.DEF: 10,  # CBIT
    Position.MID: 12,  # CBIRT
    Position.FWD: 12,  # CBIRT
}


@dataclass
class MatchStats:
    """Realised or simulated stats for one player in one match.

    For a double gameweek, score each fixture separately and sum. Do not
    aggregate the raw stats first, because thresholds (60 minutes, DefCon,
    clean sheet) are per match.
    """

    minutes: int = 0
    goals: int = 0
    assists: int = 0
    goals_conceded: int = 0  # while the player was on the pitch
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    bonus: int = 0
    # Defensive contribution inputs. Pass the count for the player's own
    # metric: CBIT for defenders, CBIRT for midfielders and forwards.
    defensive_actions: int = 0


@dataclass
class PointsBreakdown:
    """Component decomposition of a player's score.

    Keep the breakdown rather than just the total. It is what makes the
    component model debuggable and what you attribute prediction error to.
    """

    appearance: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheet: int = 0
    goals_conceded: int = 0
    saves: int = 0
    penalties: int = 0
    cards: int = 0
    own_goals: int = 0
    defcon: int = 0
    bonus: int = 0

    @property
    def total(self) -> int:
        return (
            self.appearance
            + self.goals
            + self.assists
            + self.clean_sheet
            + self.goals_conceded
            + self.saves
            + self.penalties
            + self.cards
            + self.own_goals
            + self.defcon
            + self.bonus
        )

    def as_dict(self) -> dict[str, int]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["total"] = self.total
        return d


def score_match(stats: MatchStats, position: Position | str | int) -> PointsBreakdown:
    """Score one player in one match under 2026/27 rules."""
    pos = _coerce_position(position)
    b = PointsBreakdown()

    if stats.minutes <= 0:
        # A player who does not appear scores nothing, except that cards and
        # own goals cannot happen without minutes. Bonus cannot either.
        return b

    b.appearance = (
        APPEARANCE_LONG if stats.minutes >= MINUTES_THRESHOLD else APPEARANCE_SHORT
    )
    b.goals = stats.goals * GOAL_POINTS[pos]
    b.assists = stats.assists * ASSIST_POINTS

    if stats.minutes >= MINUTES_THRESHOLD and stats.goals_conceded == 0:
        b.clean_sheet = CLEAN_SHEET_POINTS[pos]

    if pos in (Position.GKP, Position.DEF):
        b.goals_conceded = -(stats.goals_conceded // GOALS_CONCEDED_PER_PENALTY)

    if pos is Position.GKP:
        b.saves = stats.saves // SAVES_PER_POINT

    b.penalties = (
        stats.penalties_saved * PENALTY_SAVE_POINTS
        + stats.penalties_missed * PENALTY_MISS_POINTS
    )
    b.cards = (
        stats.yellow_cards * YELLOW_CARD_POINTS + stats.red_cards * RED_CARD_POINTS
    )
    b.own_goals = stats.own_goals * OWN_GOAL_POINTS

    threshold = DEFCON_THRESHOLD[pos]
    if threshold is not None and stats.defensive_actions >= threshold:
        b.defcon = DEFCON_POINTS

    b.bonus = stats.bonus
    return b


def score_gameweek(
    fixtures: list[MatchStats], position: Position | str | int
) -> PointsBreakdown:
    """Score a player across all fixtures in a gameweek (handles DGWs)."""
    totals = PointsBreakdown()
    for stats in fixtures:
        part = score_match(stats, position)
        for f in totals.__dataclass_fields__:
            setattr(totals, f, getattr(totals, f) + getattr(part, f))
    return totals


def _coerce_position(position: Position | str | int) -> Position:
    if isinstance(position, Position):
        return position
    if isinstance(position, int):
        return ELEMENT_TYPE_TO_POSITION[position]
    return Position(str(position).upper())


# --------------------------------------------------------------------------
# Expected points composition
# --------------------------------------------------------------------------
# This is the seam the component models plug into. Each argument is an
# expectation or probability produced by a separate model. Keeping the
# composition here means a rule change is a one file diff.


def expected_points(
    position: Position | str | int,
    *,
    p_appear: float,
    p_60plus: float,
    exp_goals: float,
    exp_assists: float,
    p_clean_sheet: float,
    exp_goals_conceded: float = 0.0,
    exp_saves: float = 0.0,
    p_defcon: float = 0.0,
    exp_bonus: float = 0.0,
    exp_cards: float = 0.0,
) -> float:
    """Compose component predictions into expected FPL points.

    Args are already minutes adjusted where relevant, i.e. exp_goals is the
    expected goals for this fixture given the player's expected minutes, not
    a per 90 rate. p_clean_sheet should also be conditioned on playing 60+.
    """
    pos = _coerce_position(position)

    ep = p_appear * APPEARANCE_SHORT + p_60plus * (APPEARANCE_LONG - APPEARANCE_SHORT)
    ep += exp_goals * GOAL_POINTS[pos]
    ep += exp_assists * ASSIST_POINTS
    ep += p_60plus * p_clean_sheet * CLEAN_SHEET_POINTS[pos]

    if pos in (Position.GKP, Position.DEF):
        ep -= exp_goals_conceded / GOALS_CONCEDED_PER_PENALTY
    if pos is Position.GKP:
        ep += exp_saves / SAVES_PER_POINT

    if DEFCON_THRESHOLD[pos] is not None:
        ep += p_defcon * DEFCON_POINTS

    ep += exp_bonus
    ep -= exp_cards
    return ep
