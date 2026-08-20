# FPL ML + Optimisation System

Component-based FPL points prediction and MILP squad optimisation for 2026/27.

## Non-negotiables

Three rules that, if broken, waste months of work.

1. **Snapshot from day one.** The FPL API overwrites itself. Data you do not capture
   at the time is gone forever. Run `fpl.ingest.snapshot` before you write a single
   model. It costs about 2 MB a week.
2. **All scoring goes through `fpl/scoring/rules_2026_27.py`.** Never inline a points
   calculation anywhere else. Rules change every August and you want that to be a
   one file diff.
3. **The metric is realised optimised-team points, not RMSE.** A model with worse
   RMSE that picks better captains wins. RMSE is a diagnostic only.

## Build order

Do not start with the model. The spine is:

```
snapshot -> scoring module -> curated player_gw -> minutes model -> single-GW MILP
```

Nothing else gets built until that runs end to end.

## First fortnight

**Week 1: get data flowing**

- [ ] `uv sync`, confirm `pytest` passes
- [ ] Run `python -m fpl.ingest.snapshot --season 2026-27` and eyeball the output JSON
- [ ] Schedule it: cron Friday evening, Saturday pre-deadline, Tuesday. Or a GitHub
      Action on the same cadence. Do not overthink this
- [ ] Clone `vaastav/Fantasy-Premier-League` for historical seasons. Note the
      documented trap: the `xP` column is scraped after the gameweek and leaks.
      `shift(1)` within player or drop it
- [ ] Write `fpl/ingest/curate.py`: bootstrap + live snapshots -> tidy `player_gw`
      Parquet partitioned by season and gw

**Week 2: the minutes model**

- [ ] Build `p_start`, `p_60plus` and `p_sub_appearance` as LightGBM classifiers
- [ ] Calibrate them (isotonic). Uncalibrated probabilities will wreck your EV maths
- [ ] Features: rolling starts (1/3/5), minutes trend, `status` and
      `chance_of_playing_next_round` from the API, days rest, price tier as a
      nailedness proxy, set-piece order fields
- [ ] Baseline to beat: "started last game implies starts this game". If you cannot
      beat that, your features are wrong

Set-piece order is already in bootstrap (`penalties_order`,
`corners_and_indirect_freekicks_order`, `direct_freekicks_order`). It is free and
it is the highest signal-per-hour feature in the whole project. Use it in week 2,
not v0.3.

## Then, in order

3. **Pre-build experiment 1: error attribution.** Swap true minutes vs modelled
   minutes into points composition on one archived season. Confirms minutes
   dominates before you invest in per-90 modelling. Two evenings.
4. **Single-GW MILP** via `highspy`. Legal squad, XI, captain. Not multi-GW yet.
5. **Pre-build experiment 2: odds-only baseline.** Bookmaker match and scorer odds
   plus a naive minutes rule, no ML. This is the bar. If your ML cannot beat it,
   stop and rethink the modelling plan.
6. Component models (attack Poisson, Dixon-Coles clean sheets via `penaltyblog`,
   DefCon threshold classifier, bonus).
7. Walk-forward backtest engine with information-set replay.

## Layout

```
fpl/
  ingest/     snapshot.py (done), curate.py, schema checks
  scoring/    rules_2026_27.py (done) - the only place scoring lives
  models/     minutes/ attack/ team/ defcon/ bonus/ compose.py
  optimise/   milp.py, chips.py, overrides.py
  backtest/   walk-forward engine, paired comparisons, resampling
  eval/       metrics, points-to-rank, top10k study
  baselines/  ep_next, price, odds-only, last5, openfpl adapter
data/
  raw/        immutable API snapshots, never edited
  curated/    typed and joined
  features/   model-ready
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
