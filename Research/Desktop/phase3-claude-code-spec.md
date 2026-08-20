# Phase 3 build spec: expected points, optimiser, baselines (v2)

Revised after phase 2 completion. Supersedes any earlier phase 3 spec.
Work through the three tasks IN ORDER, one per session. Do not start a task
until the previous one's acceptance criteria pass.

## Prerequisites (verify before starting)

Phase 2 is complete: five seasons curated with the `starts`, `was_home` and
`opponent_team` columns, `experiments/reports/exp1_minutes_attribution.md`
exists, `fpl/models/minutes/` produces calibrated p_start, p_60, p_sub and
exp_minutes, walk-forward beats the corrected lookup baseline on p_start
log loss in every held-out season, and the suite passes (67 tests at
handoff). `models_store/` binaries are gitignored with metadata.json kept.
If any of that is missing, stop and say so.

## Hard rules (permanent)

- All scoring goes through `fpl.scoring.rules_2026_27`. No exceptions.
- No feature or input may see the future. Historical rows may only use
  information available before that gameweek's deadline. Phase 2 set the
  precedent: end-of-season archive snapshots of status,
  chance_of_playing_next_round and set-piece order are NOT valid historical
  features. Follow it.
- Time-ordered evaluation only.
- Every task ships pytest tests, suite stays green, everything seeded.
- Negative or unflattering results go in the FIRST LINE of the relevant
  report, not a footnote. Phase 2 did this correctly. Keep doing it.
- No em dashes in comments or docs.

## Findings from phase 2 that bind this phase

1. Expected minutes IS load-bearing here. The composition uses two
   different minutes quantities and they must not be conflated:
   - exp_minutes (the model's conditional-mean estimate) scales per-90
     rates into per-fixture expectations. Its RMSE win over the naive 0/90
     rule is the relevant property. Its MAE loss is irrelevant for this
     use, do not "fix" it.
   - p_appear and p_60 feed the appearance and clean-sheet terms of
     `expected_points` directly.
2. Minutes error dominates rate error by about 3.5x (corrected exp1), and
   the trained model captures about 20 percent of the minutes gap. The v0
   rate models below are therefore deliberately crude. Do not gold-plate
   them.
3. Every prediction row carries `n_prior_matches` (matches with >0 minutes
   for that player_code before this gameweek). The harness slices on it.

---

## Task 1: expected points pipeline (`fpl/models/rates.py` and `fpl/models/compose.py`)

Goal: one command producing expected points for every player for a target
gameweek by combining the minutes model with simple v0 rate models through
the scoring module. Crude rates, permanent interface.

`fpl/models/rates.py`, all per player per fixture:
1. exp_goals_p90, exp_assists_p90: trailing per-90 rates over the last 10
   matches with >=30 minutes, shrunk with a prior weight of ~600 minutes.
   Shrinkage target: the mean of the player's position x price-tier cell,
   not the bare position mean. Price tiers: quartiles of current price
   within position, computed from the season's players table. This is the
   cheap version of hierarchical priors and it is what makes new and
   low-data players sane (a 7.5m unknown shrinks toward premium behaviour,
   a 4.5m unknown toward budget behaviour).
2. Opponent adjustment for attacking rates: opponent trailing
   goals-conceded rate over last 10 divided by league average, clipped to
   [0.6, 1.6]. Use the `opponent_team` and `was_home` columns, apply a
   home/away multiplier estimated from league-wide scoring splits.
3. p_clean_sheet: team trailing clean sheet rate over last 10, blended
   50/50 with the team's home-or-away split rate per `was_home`, adjusted
   by opponent trailing scoring rate, clipped to [0.02, 0.75].
4. exp_goals_conceded: opponent trailing scoring rate times own defensive
   adjustment, for GKP/DEF concession points.
5. exp_saves (GKP only): trailing saves per 90 times exp_minutes/90.
6. p_defcon: trailing rate of hitting the player's own threshold (10 CBIT
   DEF, 12 CBIRT MID/FWD) over last 10, shrunk toward the position x
   price-tier mean. Meaningful from 2025-26 data only; earlier seasons get
   0 and defcon_available=false.
7. exp_bonus: trailing bonus per match over last 10, shrunk toward 0.
8. exp_cards: position-level base rate.
9. Set-piece uplift: a flat additive bump to exp_goals_p90 for penalty
   takers (order 1). Historical rows use PREVIOUS season's order (the
   leak-free convention phase 2 established). Live rows use current
   bootstrap order. Carry a `sp_source` column ("prev_season" or "live")
   so the harness can measure whether the stale historical signal helps or
   hurts. Size the bump from league-wide penalties per team-match times
   conversion, roughly +0.08 per 90, named constant, documented.
Every constant in a named CONSTANTS block with a one-line rationale. No
tuning against 2025-26, it is the frozen evaluation season.

`fpl/models/compose.py`:
- Joins minutes predictions with rates, calls
  `rules_2026_27.expected_points` per fixture using the minutes quantities
  as specified in finding 1 above, sums fixtures within gameweek
  (DGW-safe), writes
  `data/predictions/{season}/gw{g}/expected_points.parquet` keeping every
  component column plus ep_total and n_prior_matches.
- CLI: `--season 2026-27 --gw 4` for live, `--historical --season 2025-26`
  to run a whole past season gameweek by gameweek on prior information
  only.

Acceptance criteria:
- Historical mode over 2025-26 under 10 minutes.
- A test hand-computes EP for one constructed player and matches compose,
  including the DGW sum and the penalty bump.
- Leakage test: gw g EP identical whether or not gw g+1 exists in curated.
- A test asserts a player with n_prior_matches=0 gets the position x
  price-tier prior, not NaN and not the global mean.
- `experiments/reports/ep_v0_sanity.md`: top 20 by EP for one historical
  gameweek, plus the top 5 among players with n_prior_matches < 5. If
  either list is mad, the report says so in the first line.

## Task 2: single-GW MILP optimiser (`fpl/optimise/milp.py`)

Goal: given expected points and a squad state, the optimal legal 15, XI,
captain and vice, transfers priced in. Single gameweek only. Build the
model construction as one function taking a horizon list even though only
length 1 is supported, so phase 6 extends rather than rewrites.

Use highspy directly. If the API is awkward, PuLP with the HiGHS backend
is acceptable, note the choice in the module docstring.

Modes:
1. `fresh`: no existing squad, 100.0m budget, best 15/XI/C/VC. Used by the
   harness and for wildcards.
2. `transfer`: existing squad plus bank plus free_transfers (1 to 5) from
   `team_state.yaml` (player codes, purchase_price, selling_price, bank,
   free_transfers). Human supplies selling prices, store purchase_price
   for a later phase to compute the 50 percent sell-on rule. Each transfer
   beyond FT costs 4 points in the objective.

Constraints, exactly:
- Squad 2/5/5/3, max 3 per club, budget per mode.
- XI: 11 of 15, exactly 1 GKP, >=3 DEF, >=1 FWD.
- Captain and vice: one each, both in XI, different players.
- Objective: XI ep_total + captain ep_total again
  + BENCH_WEIGHT * bench ep_total - 4 * paid_transfers, with
  BENCH_WEIGHT = 0.1 and a comment that phase 6 replaces it with autosub
  probabilities.
- `overrides.yaml`: lock (in XI), ban (out of squad), force_captain.
  Infeasible overrides fail loud naming the binding constraint.

Output: readable table (name, pos, team, price, EP, C/VC, transfers and
hit cost) and `data/decisions/{season}/gw{g}/squad.json`.

Acceptance criteria:
- Under 5 seconds on the full player pool.
- Synthetic-pool tests: position counts, club limit binds, budget binds,
  captain is max-EP in XI when unforced, ban excludes, lock includes,
  transfer mode declines a 1-EP upgrade at -4 but takes a 5-EP upgrade.
- Determinism: identical inputs, identical squad, two runs. If HiGHS
  multithreading breaks this, pin threads=1 and note it.

## Task 3: baselines and the evaluation harness (`fpl/baselines/` and `fpl/eval/harness.py`)

Goal: the referee. Realised points of the optimised team, candidates
compared like for like. Every future model change passes through here.

Baselines, each emitting compose.py's output schema (ep_total plus
n_prior_matches, components where they exist):
1. `last5.py`: mean points over last 5 appearances via the scoring module
   per rule_regime, 0 if none.
2. `price.py`: ep_total = current price.
3. `ep_next.py`: FPL's ep_next from live snapshots only. Historical
   seasons return an empty frame with a logged explanation. Do not fake it
   from the archive, that column leaks (phase 2 established this for xP,
   same applies).
4. `naive_minutes_ep.py`: exp1 variant B (naive 0/90 minutes plus trailing
   rates), frozen as the reference the trained system must beat.

`fpl/eval/harness.py`, mode "weekly fresh pick":
- Held-out season 2025-26, gameweeks 6 to 38: build each candidate's EP on
  prior information only, run the optimiser in fresh mode, score XI plus
  captain against realised points via the scoring module on
  player_fixture (DGW-safe).
- Name it clearly as a model comparison, not a season simulation. The
  sequential backtest with transfer state is phase 5.
- Diagnostics per candidate: mean within-position Spearman of EP vs
  realised points, captain precision at 1 (chosen captain in the top 5
  realised scorers among that squad's players).
- Paired comparison: per-gameweek realised deltas between candidates,
  mean delta, sign test p-value.
- Low-data slice: all headline metrics recomputed on the subset of
  selected players with n_prior_matches < 5, reported separately. This is
  the number that tells us how badly new players are handled.
- Output `experiments/reports/eval_{date}.md`, one table: candidate, mean
  weekly realised points, delta vs last5, sign test p, Spearman, captain
  precision, low-data Spearman. First line states which candidate won,
  or that the trained system lost, plainly.

Acceptance criteria:
- compose-v0, last5, price and naive_minutes_ep on 2025-26 end to end in
  under 30 minutes.
- A test runs 3 synthetic gameweeks and checks the paired delta and sign
  test arithmetic by hand.
- If compose-v0 does not beat last5 or does not beat naive_minutes_ep,
  the first line of the report says so.

## Out of scope

Multi-GW MILP, chips, banked FT state, autosub probabilities beyond
BENCH_WEIGHT, odds data, Understat joins, Dixon-Coles, price modelling,
sequential simulation, league-adjusted history for new signings. Finish
early: strengthen tests and reports, do not start new modules.
