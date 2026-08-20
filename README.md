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
   RMSE that picks better captains wins. RMSE is a diagnostic only.

## Build order

Do not start with the model. The spine is:

```
snapshot -> scoring module -> curated player_gw -> minutes model -> single-GW MILP
```

Nothing else gets built until that runs end to end. Everything up to and
including expected points is done. The MILP is next, and it is the last
piece of the spine.

```bash
python -m ingest.snapshot --season 2026-27      # capture, run on a schedule
python -m ingest.history --all                  # 2021-22 to 2025-26, ~8s warm
python -m ingest.curate --season 2026-27        # live snapshots to Parquet
python -m models.minutes.train --version v0     # walk forward, ~35s
python -m models.minutes.predict --season 2026-27 --gw 1
python -m models.compose --season 2026-27 --gw 1        # expected points
python -m models.compose --season 2025-26 --historical  # replay a season, ~10s
python -m experiments.exp1_minutes_attribution  # why minutes came first
python -m experiments.ep_v0_sanity              # is the EP list mad
```

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
4. **Single-GW MILP** via `highspy`. Legal squad, XI, captain. Not multi-GW yet.
5. **Pre-build experiment 2: odds-only baseline.** Bookmaker match and scorer odds
   plus a naive minutes rule, no ML. This is the bar. If your ML cannot beat it,
   stop and rethink the modelling plan.
6. Component models (attack Poisson, Dixon-Coles clean sheets via `penaltyblog`,
   DefCon threshold classifier, bonus).
7. Walk-forward backtest engine with information-set replay.

## Layout

```
ingest/     snapshot.py, curate.py, history.py, validate.py (all done)
mapping/    id maps, overrides.csv, join validators
features/   feature builders, rolling windows, set-piece flags
scoring/    rules_2026_27.py (done) - the only place scoring lives
models/     minutes/ (done) rates.py (v0) compose.py (done)
            attack/ team/ defcon/ bonus/ replace rates.py later
optimise/   milp.py, chips.py, overrides.py
backtest/   walk-forward engine, paired comparisons, resampling
eval/       metrics, points-to-rank, top10k study
baselines/  ep_next, price, odds-only, last5, openfpl adapter
ops/        weekly scheduler, staleness/schema checks, alerting
cli.py
experiments/        one-off studies, not production code
  reports/          generated markdown, committed so results are reviewable
data/
  raw/              immutable API snapshots, never edited
  external/         vaastav archive cache, downloaded once
  curated/          typed and joined, historical and live indistinguishable
  predictions/      {season}/gw{n}/minutes.parquet
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
