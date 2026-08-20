# CLAUDE.md addendum: Phase 2 build spec

Work through the three tasks below IN ORDER, one per session. Do not start
task 2 until task 1's acceptance criteria pass, and so on.

## Project context

Component-based FPL points prediction and MILP squad optimisation for 2026/27.
Repo state: `fpl/scoring/rules_2026_27.py` (the ONLY place scoring logic may
live), `fpl/ingest/snapshot.py` (immutable raw API snapshots) and
`fpl/ingest/curate.py` (snapshots -> players, fixtures, player_fixture,
player_gw Parquet under `data/curated/{season}/`) are done and tested.
25 tests pass. Read all three files plus README.md before writing anything.

## Hard rules for every task

- Never compute FPL points anywhere except through `fpl.scoring.rules_2026_27`.
- Never let a feature see the future. Every rolling feature is shifted so a
  row for gameweek g uses only data from gameweeks < g. Write a test proving
  it for at least one feature.
- Time-ordered splits only. No random train/test splits anywhere, ever.
- Every task ships with pytest tests and `python -m pytest tests/ -q` green.
- Deterministic: seed everything, no network calls inside tests.
- Style: no em dashes in comments or docs, follow existing module patterns.

---

## Task 1: historical ingest (`fpl/ingest/history.py`)

Goal: seasons 2021-22 through 2025-26 from the vaastav archive, curated into
the SAME table shapes curate.py produces, so downstream code cannot tell
historical and live data apart.

Source: https://github.com/vaastav/Fantasy-Premier-League under
`data/{season}/`. Use `merged_gw.csv` (player x gameweek rows),
`players_raw.csv` (season player list with `code`), `fixtures.csv` and
`teams.csv`. Download only the files needed (raw.githubusercontent.com),
cache under `data/external/vaastav/{season}/`, never re-download if present.

Requirements:
1. Output `data/curated/{season}/player_gw.parquet`, `players.parquet` and
   `fixtures.parquet` matching curate.py's column names and dtypes. Add a
   `source` column: "vaastav" vs "live".
2. Player identity: FPL element IDs are NOT stable across seasons. Use the
   stable `code` field from players_raw.csv as `player_code` on every table,
   keep the per-season `player_id` too. Build
   `data/curated/player_index.parquet` mapping code -> per-season ids, names,
   positions.
3. Drop the `xP` column entirely. It is scraped after the gameweek and leaks.
   Do not shift it, drop it, and leave a comment saying why.
4. Column names drift between seasons in the archive. Write an explicit
   per-season rename map, fail loud on unexpected missing columns, log new
   ones. Note `defensive_contribution` style columns exist only from 2025-26.
5. Add a `rule_regime` column per season (e.g. "pre_defcon" for <=2024-25,
   "defcon_v1" for 2025-26). Downstream needs to know points were earned
   under different rules.
6. Positions: map element_type per season. Beware position reclassifications
   between seasons for the same code, keep position per season not global.
7. Validation module `fpl/ingest/validate.py`: reconcile each player's
   summed goals/assists/minutes from player_gw against the season totals in
   players_raw.csv (or cleaned_players.csv). Tolerance zero. Emit a report of
   mismatches, fail the run if more than 1 percent of players mismatch.
8. DGW handling: merged_gw.csv has one row per player per FIXTURE (check
   this per season, it changed around 2019-20). Where per-fixture rows exist,
   also emit player_fixture.parquet. Where they do not, emit player_gw only
   and set n_fixtures from the fixture table.
9. CLI: `python -m fpl.ingest.history --seasons 2021-22 2022-23 ...` and
   `--all` for the default five.

Acceptance criteria:
- Five seasons curated, validation report shows >=99 percent reconciliation.
- A test loads 2025-26 player_gw, picks a known DGW, asserts n_fixtures == 2.
- A test proves live and historical player_gw have identical column sets.
- Total runtime under 5 minutes from warm cache.

## Task 2: experiment 1, minutes error attribution (`experiments/exp1_minutes_attribution.py`)

Goal: quantify how much of expected points error comes from minutes vs
per-90 rates, using season 2025-26. This number decides how much effort the
minutes model deserves. It is a script plus a short markdown report, not
production code.

Method:
1. For every player-gameweek in 2025-26 from gw6 onward, build three
   expected points estimates via `fpl.scoring.rules_2026_27.expected_points`:
   - A "oracle minutes": true minutes converted to p_appear/p_60plus (1/0),
     per-90 rates from trailing per-90 stats over the previous 5 matches
     played (goals, assists, defensive actions), team-level trailing clean
     sheet rate for p_clean_sheet, trailing bonus mean for exp_bonus.
   - B "naive minutes": same rates, but minutes from a naive rule
     (p_start = started last match, exp minutes 90 if so else 0).
   - C "oracle rates": minutes as in B... no, invert: true per-match outcomes
     for rates (actual goals/assists/defcon that match) with naive minutes.
   So A isolates rate error, C isolates minutes error, B is the full naive
   system. Decompose: (MAE_B - MAE_A) is what perfect minutes buys,
   (MAE_B - MAE_C) is what perfect rates buy.
2. Score against actual gameweek points from player_gw (recompute from
   components through the scoring module, do not trust stored total_points
   blindly, log any disagreement rate).
3. Report MAE and RMSE per position and overall, plus the two deltas, into
   `experiments/reports/exp1_minutes_attribution.md` with a 10 line summary
   table and a 3 sentence conclusion.

Acceptance criteria: script runs end to end from curated data in under 2
minutes, report generated, a test covers the estimate construction for one
hand-built player case.

## Task 3: minutes model (`fpl/models/minutes/`)

Goal: calibrated probabilities p_start, p_60plus_given_start and
p_sub_appearance per player per upcoming fixture, plus expected minutes.
This is the highest-value model in the project. Take it seriously.

Structure:
- `fpl/models/minutes/dataset.py`: build the training frame from curated
  player_gw/player_fixture across all seasons. One row per player per
  fixture. Labels: started (minutes > 0 and in starting XI if available,
  else minutes >= 45 as proxy, document the choice), played_60, sub_appear.
- `fpl/models/minutes/features.py`: all features shifted/lagged. Minimum
  set: started last 1/3/5 matches (rates), minutes in last 1/3/5, matches
  since last start, days since previous fixture, price and price rank within
  team-position, status and chance_of_playing_next_round where present
  (historical availability is patchy, encode missing explicitly),
  set piece order flags, team fixture congestion (fixtures in next 7 days),
  new-signing flag (no prior matches for code), season gameweek number.
- `fpl/models/minutes/train.py`: LightGBM classifiers for the three heads.
  Walk-forward evaluation: train on seasons up to S-1, evaluate on season S,
  for S in {2023-24, 2024-25, 2025-26}. Calibrate with isotonic regression
  fitted on a tail split of training data. Save models + calibrators +
  feature list to `models_store/minutes/{version}/` with a metadata json
  (git sha, train seasons, metrics).
- `fpl/models/minutes/predict.py`: CLI producing
  `data/predictions/{season}/gw{g}/minutes.parquet` for an upcoming gameweek
  from curated live data, columns: player_id, player_code, fixture_id,
  p_start, p_60, p_sub, exp_minutes.
- Expected minutes: p_start * E[min|start] + p_sub * E[min|sub]. Estimate
  the two conditional means empirically per position from training data,
  do not hardcode 90.

Evaluation (write to `experiments/reports/minutes_model.md`):
- Log loss and Brier for each head vs the baseline "started last match".
- Calibration table (10 bins) for p_start. The model MUST be calibrated:
  predicted 0.7 bucket should start about 70 percent of the time.
- MAE of expected minutes vs actual minutes, vs the naive rule.
- Beat the baseline on log loss for p_start on every held-out season or
  say loudly in the report that it does not and why.

Acceptance criteria:
- All tests green including a leakage test: shuffle a future gameweek's
  outcomes and assert train-time features for earlier gameweeks unchanged.
- predict.py runs against current live curated data and produces sane output
  (spot check: a nailed premium keeper has p_start > 0.9).
- Walk-forward beats the naive baseline on p_start log loss.

## Out of scope for all three tasks

No attack/clean sheet/bonus models, no optimiser work, no odds, no Understat
joins, no ID mapping beyond the vaastav player code. Those come later. If
you finish early, improve tests and the reports, do not start new modules.
