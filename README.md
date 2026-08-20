# FPL ML + Optimisation System

Component-based FPL points prediction and MILP squad optimisation for 2026/27.

## Non-negotiables

Three rules that, if broken, waste months of work.

1. **Snapshot from day one.** The FPL API overwrites itself. Data you do not capture
   at the time is gone forever. Run `ingest.snapshot` before you write a single
   model. It costs about 2 MB a week.
2. **All scoring goes through `scoring/rules_2026_27.py`.** Never inline a points
   calculation anywhere else. Rules change every August and you want that to be a
   one file diff.
3. **The metric is realised optimised-team points, not RMSE.** A model with worse
   RMSE that picks better captains wins. RMSE is a diagnostic only. This is not
   theory: in the first evaluation, `last5` had the *best* rank correlation of any
   candidate and scored 17 points a week fewer than the winner. Run
   `python -m eval.harness` before believing any model is better.

## Build order

Do not start with the model. The spine is:

```
snapshot -> scoring module -> curated player_gw -> minutes model -> single-GW MILP
```

Nothing else gets built until that runs end to end. **The spine now runs end
to end.** What is missing from here is not plumbing, it is better models and
an evaluation harness to prove they are better.

```bash
python -m ingest.snapshot --season 2026-27      # capture, run on a schedule
python -m ingest.history --all                  # 2021-22 to 2025-26, ~8s warm
python -m ingest.curate --season 2026-27        # live snapshots to Parquet
python -m models.minutes.train --version v0     # walk forward, ~35s
python -m models.minutes.predict --season 2026-27 --gw 1
python -m models.compose --season 2026-27 --gw 1        # expected points
python -m models.compose --season 2025-26 --historical  # replay a season, ~10s
python -m optimise.milp --season 2026-27 --gw 1          # best legal 15/XI/C
python -m optimise.milp --season 2026-27 --gw 2 --mode transfer
python -m eval.harness                          # the referee, ~65s
python -m experiments.exp1_minutes_attribution  # why minutes came first
python -m experiments.ep_v0_sanity              # is the EP list mad
```

For transfer mode, copy `team_state.example.yaml` to `team_state.yaml` and fill
in your squad, or generate a starting one with
`--save-state team_state.yaml`. To overrule the model, copy
`overrides.example.yaml` to `overrides.yaml`. An override that makes a legal
squad impossible fails with the constraint it broke, for example "2 keepers are
locked into the eleven, which allows exactly 1".

## First fortnight

**Week 1: get data flowing** (done)

- [x] Snapshot the live API, immutable, schema validated
- [x] Curate snapshots into `players`, `fixtures`, `player_fixture`, `player_gw`
- [x] Historical seasons 2021-22 to 2025-26 from the vaastav archive, curated into
      identical shapes so downstream cannot tell them apart. `xP` is dropped, not
      shifted, because it is scraped after the gameweek and leaks
- [x] Reconciliation against archive season totals, tolerance zero, over 99.8%
      clean on every season

**Week 2: the minutes model** (done)

- [x] `p_start`, `p_60_given_start` and `p_sub` as LightGBM classifiers
- [x] Isotonic calibration, fitted on a chronological tail the booster never saw
- [x] Walk forward evaluation, beats the started-last-match baseline on every
      held out season
- [x] Expected minutes from estimated conditional means, not a hardcoded 90

Two features the plan called for are deliberately absent, and the reason is worth
knowing: `status`, `chance_of_playing_next_round` and current-season set-piece
order survive in the archive only as an end-of-season snapshot, so training on
them would leak the future into every earlier gameweek. The previous season's
set-piece order is used instead, which is leak free. Once the weekly snapshots
have accumulated a season of per-gameweek availability, these become the single
biggest available upgrade. See `experiments/reports/minutes_model.md`.

## Then, in order

3. **Pre-build experiment 1: error attribution.** Done, see
   `experiments/reports/exp1_minutes_attribution.md`. Perfect minutes knowledge is
   worth 0.279 MAE per player-fixture against 0.079 for perfect rate knowledge, so
   minutes dominate and the build order holds. The model captures 20% of that gap.
3b. **Expected points v0.** Done. `models/rates.py` holds deliberately crude
   trailing rate estimators and `models/compose.py` puts them through the scoring
   module with the minutes model. A whole season replays in about 10 seconds. The
   known weakness is compression at the top of the ranking, written up in
   `experiments/reports/ep_v0_sanity.md`: shrinkage is too aggressive for elite
   players, so premiums are under-ranked against cheap nailed defenders. Item 6
   is what fixes it.
4. **Single-GW MILP** via `highspy`. Done. `optimise/milp.py` picks the best legal
   15, XI, captain and vice in about 0.1 seconds on the full pool, in either
   `fresh` or `transfer` mode, with hits priced inside the objective. Multi-GW,
   chips and banked-FT state are not built, but the model construction already
   takes a horizon list so they attach rather than require a rewrite.
4b. **Baselines and the evaluation harness.** Done. `eval/harness.py` is the referee
   for every future change: each candidate builds EP from prior information only,
   runs the same optimiser, and is scored on the realised points of the team it
   picked. First result, 2025-26 gameweeks 6 to 38, in
   `experiments/reports/eval_2026-08-20.md`:

   | Candidate | Mean pts/GW | vs last5 | Sign test p |
   |---|---:|---:|---:|
   | compose-v0 | 54.36 | +17.33 | < 0.001 |
   | naive_minutes_ep | 46.73 | +9.70 | 0.002 |
   | last5 | 37.03 | reference | |
   | price | 29.88 | -7.15 | 0.377 |

   compose-v0 wins, and beats the frozen pre-model system by 7.6 points a week,
   which is what the minutes model plus the rate estimators are worth. Note this
   is a weekly fresh pick, not a season simulation: no transfer continuity, no
   autosubs, no chips. Phase 4 adds those and the numbers will change.
5. **Pre-build experiment 2: odds-only baseline.** Done, and it is the most important
   result so far. De-vigged football-data.co.uk closing prices, Shin's method, 100%
   fixture coverage across five seasons, no ML anywhere. It scores **54.39 points a
   gameweek against compose-v0's 54.36**: a paired delta of -0.03 across a 16-16-1
   sign record, p = 1.000. The trained pipeline does not beat the market, and it does
   not lose to it either. See `experiments/reports/odds_baseline.md` and
   `eval_2026-08-20.md`.

   Read carefully, that is not a verdict on the models. It is the harness saying it
   cannot separate them, consistent with the phase 3 finding that this measurement
   barely moves under within-tier reordering. compose-v0 does beat last5 in 29 of 33
   gameweeks against the odds baseline's 24, which is the more informative signal.
   The sequential backtest in phase 5 is what settles it.
5b. **Team strength: the market, with Dixon-Coles behind it.** Done, and it broke the
   tie. The de-vigged market beat Dixon-Coles on match outcome log loss in all four
   pre-freeze seasons (0.9417 vs 0.9824), and Dixon-Coles beat the trailing rates in
   all of them, so compose now takes clean sheets and concessions from the market and
   falls back to DC where no odds exist. The gated run moved the mean from **54.36 to
   57.58** points a gameweek, and compose now clears the odds-only baseline it
   previously tied. See `experiments/reports/team_model.md`.

   The architecture this settles: the market owns team strength and cannot see who
   starts; this project owns a calibrated minutes model and will not out-predict
   closing prices. Feed the first into the second. Phase 4's attack model takes its
   team expectation from whichever source wins, not from Dixon-Coles specifically.

   **Operational catch:** football-data publishes closing odds only for matches
   already played, so a live gameweek has no market coverage and DC currently carries
   the entire 2026-27 season. A live odds feed is now load-bearing, not optional.
6. Component models (attack Poisson, DefCon threshold classifier, bonus).
7. Walk-forward backtest engine with information-set replay.

## Layout

```
ingest/     snapshot.py, curate.py, history.py, validate.py, odds.py (all done)
mapping/    id maps, overrides.csv, join validators
features/   feature builders, rolling windows, set-piece flags
scoring/    rules_2026_27.py (done) - the only place scoring lives
            replay.py - recompute realised points from components
models/     minutes/ (done) rates.py compose.py team/dixon_coles.py (done)
            attack/ defcon/ bonus/ still to build
optimise/   milp.py (done) overrides.py (done) chips.py
backtest/   walk-forward engine, paired comparisons, resampling
eval/       harness.py (done) - the referee. live_log.py - the honest test
baselines/  last5, price, ep_next, naive_minutes_ep, odds_only (all done)
ops/        weekly scheduler, staleness/schema checks, alerting
cli.py
experiments/        one-off studies, not production code
  reports/          generated markdown, committed so results are reviewable
team_state.yaml     your squad, bank and free transfers (see the .example)
overrides.yaml      manual locks, bans, forced captain (see the .example)
data/
  raw/              immutable API snapshots, never edited
  external/         vaastav archive cache, downloaded once
  curated/          typed and joined, historical and live indistinguishable
  predictions/      {season}/gw{n}/minutes.parquet, expected_points.parquet
  decisions/        {season}/gw{n}/squad.json
models_store/       versioned model artefacts plus metadata json
```

## Stack

- Python 3.11+, polars or pandas, pyarrow, DuckDB for ad-hoc queries over Parquet
- LightGBM primary
- `penaltyblog` for Dixon-Coles, bivariate Poisson, odds de-vigging and scrapers.
  Do not hand-roll Dixon-Coles
- `highspy` for MILP. HiGHS, not CBC. It is roughly 2x faster and actively developed
- MLflow with a SQLite backend, or just a `runs/` directory of Parquet

## Reference points

- **OpenFPL** (arXiv:2508.09992, MIT licensed) rivals commercial FPL Review on
  high-return players but loses on low-return ones, because it uses the API
  availability tags instead of expected minutes projections. That gap is your
  opening, and it is why the minutes model comes first.
- **sertalpbilal/FPL-Optimization-Tools** is the reference MILP formulation. Read
  its multi-period solver before writing your own.
- **theFPLkiwi** publishes free expected points and expected minutes in
  fplreview-compatible format. Useful as a benchmark.

## Rules encoded (2026/27, verified August 2026)

15 players (2/5/5/3), 100.0m budget, max 3 per club. Bank up to 5 free transfers,
and banked transfers survive a Wildcard or Free Hit. Eight chips, two each of
Wildcard, Free Hit, Triple Captain, Bench Boost, first set expiring at the GW19
deadline. Goals: GKP 10, DEF 6, MID 5, FWD 4. DefCon: DEF +2 at 10 CBIT, MID/FWD
+2 at 12 CBIRT, capped at 2 per match. BPS reworked for 2026/27 (tackled-player
penalty removed, CBI now 1 BPS per 3 actions, GK saves rescored), which makes
DefCon and bonus more separable than last season.
