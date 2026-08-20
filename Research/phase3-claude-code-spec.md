# Phase 3 build spec: expected points, optimiser, baselines

Work through the three tasks IN ORDER, one per session. Do not start a task
until the previous one's acceptance criteria pass.

## Prerequisites (verify before starting)

Phase 2 is complete: `fpl/ingest/history.py` has curated seasons 2021-22
through 2025-26, `experiments/reports/exp1_minutes_attribution.md` exists,
and `fpl/models/minutes/` produces calibrated p_start, p_60, p_sub and
exp_minutes via predict.py. All tests green. If any of that is missing,
stop and say so.

## Hard rules (same as phase 2, permanent)

- All scoring goes through `fpl.scoring.rules_2026_27`. No exceptions.
- No feature or input may see the future. Rolling inputs use gameweeks < g.
- Time-ordered evaluation only.
- Every task ships pytest tests, suite stays green, everything seeded.
- No em dashes in comments or docs.

---

## Task 1: expected points pipeline (`fpl/models/rates.py` and `fpl/models/compose.py`)

Goal: one command that produces expected points for every player for a
target gameweek, by combining the minutes model with simple v0 rate models
through the scoring module. This productionises what exp1 prototyped. The
rate models are deliberately crude, they get replaced by proper Poisson and
Dixon-Coles models in a later phase. The INTERFACE is what matters now.

`fpl/models/rates.py`, v0 rate estimators, all per player per fixture:
1. exp_goals_p90, exp_assists_p90: shrunk trailing per-90 rates. Use the
   last 10 matches with >=30 minutes, shrink toward the position mean with a
   prior weight of ~600 minutes (empirical Bayes flavour, tune nothing yet).
   Multiply by exp_minutes/90 and an opponent adjustment: opponent trailing
   goals-conceded rate divided by league average, clipped to [0.6, 1.6].
2. p_clean_sheet: team trailing clean sheet rate over last 10, blended
   50/50 with a home/away split rate, adjusted by opponent trailing
   goals-scored rate the same way, clipped to [0.02, 0.75].
3. exp_goals_conceded: opponent trailing scoring rate times own defensive
   adjustment, for GKP/DEF concession points.
4. exp_saves (GKP only): trailing saves per 90 times exp_minutes/90.
5. p_defcon: trailing rate of hitting the player's own threshold (10 CBIT
   for DEF, 12 CBIRT for MID/FWD) over the last 10 matches, shrunk toward
   position mean. Only meaningful from 2025-26 data onward, else 0 and a
   `defcon_available` flag set false.
6. exp_bonus: trailing bonus per match over last 10, shrunk toward 0.
7. exp_cards: position-level base rate, do not model per player yet.
Document every constant in the docstring. These are placeholders with sane
behaviour, not science.

`fpl/models/compose.py`:
- Joins minutes predictions with rate outputs, calls
  `rules_2026_27.expected_points` per player per fixture, sums fixtures
  within the gameweek (DGW-safe), writes
  `data/predictions/{season}/gw{g}/expected_points.parquet` with every
  component column kept (exp_goals, p_clean_sheet, ...) plus `ep_total`.
  Keep components, they are how you debug and how error attribution works.
- CLI: `python -m fpl.models.compose --season 2026-27 --gw 4` for live, and
  `--historical` mode that runs a whole past season gameweek by gameweek
  using only information available before each gameweek.

Acceptance criteria:
- Historical mode over 2025-26 completes in under 10 minutes.
- A test hand-computes EP for one constructed player and matches compose.
- A leakage test: player's gw g EP is identical whether or not gw g+1 data
  exists in the curated tables.
- Sanity report `experiments/reports/ep_v0_sanity.md`: top 20 players by EP
  for one historical gameweek. If the list is obviously mad (bench fodder in
  the top 10), say so in the report rather than papering over it.

## Task 2: single-GW MILP optimiser (`fpl/optimise/milp.py`)

Goal: given expected points and a squad state, produce the optimal legal
15, XI, captain and vice, with transfers priced in. Single gameweek only.
Multi-GW, chips and banked-FT state machines are a later phase, do not
build them now, but do not paint them into a corner either: keep the model
construction in one function that takes a horizon list even if only length
1 is supported.

Use highspy directly (HiGHS). If the API is awkward, PuLP with the HiGHS
backend is acceptable, note the choice in the module docstring.

Two modes:
1. `fresh`: no existing squad, 100.0m budget, pick the best 15/XI/C/VC.
   This is the wildcard/preseason mode and the model-evaluation mode.
2. `transfer`: existing squad plus bank plus free_transfers (int 1 to 5).
   Decide transfers, each beyond FT costs 4 points inside the objective.
   Squad state comes from `team_state.yaml` at repo root, format:
   player codes with purchase_price and selling_price, bank, free_transfers.
   Selling prices are supplied by the human, do not compute the 50 percent
   sell-on rule yet, but store purchase_price so a later phase can.

Constraints (encode exactly):
- Squad: 2 GKP, 5 DEF, 5 MID, 3 FWD, max 3 per club, budget from mode.
- XI: 11 of the 15, exactly 1 GKP, >=3 DEF, >=1 FWD.
- Captain and vice: exactly one each, both in XI, not the same player.
- Objective: sum over XI of ep_total + captain's ep_total again
  + 0.1 * sum over bench of ep_total (crude bench weight placeholder,
  named constant BENCH_WEIGHT with a comment that phase 4 replaces it with
  autosub probabilities) - 4 * paid_transfers.
- Overrides from `overrides.yaml`: lock (must be in XI), ban (must not be
  in squad), force_captain. Fail loud if an override is infeasible, print
  WHICH constraint made it infeasible.

Output: solved squad printed as a readable table (name, pos, team, price,
EP, C/VC markers, transfers in/out and hit cost) and written to
`data/decisions/{season}/gw{g}/squad.json`.

Acceptance criteria:
- Solve time under 5 seconds on the full ~700 player pool.
- Tests on a tiny synthetic pool (~30 players) assert: position counts,
  club limit binds when it should, budget binds, captain has max EP among
  XI when unforced, a ban removes a player, a lock forces one in, and in
  transfer mode a 1-EP upgrade does not justify a -4 hit but a 5-EP
  upgrade does.
- Determinism: same inputs give same squad across two runs.

## Task 3: baselines and the evaluation harness (`fpl/baselines/` and `fpl/eval/harness.py`)

Goal: the machinery that answers "is my model actually better" using the
end-to-end metric: realised points of the optimised team. This harness is
the referee for every future model change.

Baselines, each producing the SAME schema as compose.py output
(player_id, player_code, gw, ep_total, minimal components where possible):
1. `last5.py`: mean total points over last 5 appearances, 0 for no
   appearances. Recompute points through the scoring module from components
   per the rule_regime, do not trust stored totals across regimes.
2. `price.py`: ep_total = current price (pure ranking signal).
3. `ep_next.py`: FPL's own ep_next. Live snapshots only. For historical
   seasons it is not reliably archived, return an empty frame with a
   logged explanation rather than faking it.
4. `naive_minutes_ep.py`: the exp1 variant B system (naive minutes plus
   trailing rates) as a frozen reference.

`fpl/eval/harness.py`, evaluation mode "weekly fresh pick":
- For a held-out season (default 2025-26), for each gameweek from gw6:
  build each candidate's EP using only prior information, run the optimiser
  in `fresh` mode, score the resulting XI plus captain against realised
  points computed through the scoring module (DGW-safe, use player_fixture).
- This is NOT a realistic season simulation (no transfer continuity), it is
  a clean model comparison on identical decisions. Name it clearly in code
  and report. The sequential backtest with transfer state is phase 4.
- Also compute two diagnostics per candidate: Spearman of EP vs realised
  points within position per gameweek (mean across gameweeks), and
  captain precision at 1 (how often the chosen captain was in the top 5
  realised scorers among owned-eligible players).
- Paired comparison: report per-gameweek realised point deltas between
  candidates, mean delta, and a sign test p-value across gameweeks. No
  fancy stats yet, just honest uncertainty.
- Output `experiments/reports/eval_{date}.md` with one table: candidate,
  mean weekly realised points, delta vs last5, sign test p, Spearman,
  captain precision.

Acceptance criteria:
- Harness runs compose-v0, last5, price and naive_minutes_ep on 2025-26
  end to end in under 30 minutes.
- The report exists and states plainly which candidate won.
- A test runs the harness on 3 synthetic gameweeks and checks the paired
  delta arithmetic by hand.
- If compose-v0 does NOT beat last5, the report must say so in the first
  line. That result is informative, not embarrassing, and phase 4
  prioritisation depends on it.

## Out of scope for all tasks

Multi-GW MILP, chips, banked FT state, autosub probabilities beyond the
BENCH_WEIGHT placeholder, odds data, Understat joins, Dixon-Coles, price
change modelling, sequential season simulation. If you finish early,
strengthen tests and reports.
