# Phase 4 build spec: real rate models and the external bar

Five tasks, one per session, IN ORDER. Task order is deliberate: the
external bar is established before the models it judges, and the team
model exists before the attack model that allocates its goals.

## Prerequisites (verify before starting)

Phase 3 frozen: v0 rates with the evidence-weighted opponent adjustment,
single-GW HiGHS optimiser, evaluation harness with two runs on record
(mean 54.364, sign record 29-4 vs last5 on 2025-26), 134 tests green.
Curated tables carry starts, was_home, opponent_team, n_prior_matches.
Team-level joins across seasons key on club name, never team_id.
If any of that is missing, stop and say so.

## Hard rules (permanent, unchanged)

- All scoring through `fpl.scoring.rules_2026_27`.
- No feature or input sees the future. End-of-season archive snapshots of
  mutable fields are not historical features.
- Time-ordered evaluation only. Everything seeded, suite stays green.
- Negative results in the FIRST LINE of the relevant report.
- No em dashes in comments or docs.

## Harness gating policy (new, binds every task)

2025-26 is the frozen evaluation season and its value degrades with every
run against it. Therefore:
1. Each component earns AT MOST ONE harness run, taken only after its own
   internal validation passes. Design decisions are made on internal
   metrics (log loss, calibration, reconciliation), never by iterating
   against harness output.
2. A component is swapped into compose alone, compared paired against the
   pre-swap system, and accepted or rejected on that one run. Rejected
   components are kept behind a flag with the result documented, not
   deleted and not retried with tweaks.
3. Primary readout is the per-gameweek sign record and mean paired delta.
   The phase 3 finding stands: the harness mean is nearly flat under
   within-tier player reordering, so a flat mean with an improved sign
   record is a positive result, and a flat everything is "no measurable
   effect at this resolution", which phase 5's sequential backtest will
   re-test. Say which one happened.
4. In parallel, live 2026-27 tracking starts now (task 0 below). The live
   season is the only evaluation data nothing was ever tuned on. It
   accumulates one gameweek a week and becomes the honest test.

## Task 0: live tracking (do this in the task 1 session, 30 minutes)

`fpl/eval/live_log.py`: after each real gameweek, append one row per
candidate (compose, last5, ep_next live, price) to
`data/eval/live_2026-27.parquet`: gw, candidate, optimised fresh-pick XI
realised points, captain realised points, timestamp of the prediction file
used. Predictions must be the ones generated BEFORE the deadline from
snapshots, never regenerated after. A tiny report
`experiments/reports/live_2026-27.md` regenerates from the log. No
analysis yet, just an append-only paper trail.

## Task 1: odds ingest and the odds-only baseline (`fpl/ingest/odds.py`, `fpl/baselines/odds_only.py`)

Goal: the external bar. If trained components cannot beat bookmaker-derived
expectations, that is the headline finding of the phase.

Ingest:
- Historical: football-data.co.uk EPL CSVs, seasons 2021-22 through
  2025-26. Free, per-match 1X2 and over/under closing odds from several
  books. penaltyblog has a scraper, use it if convenient, else plain CSV
  download cached under `data/external/football-data/`. Join to fixtures
  on club name and kickoff date, with a name-mapping overrides file
  (football-data spells clubs differently, e.g. "Man United").
- De-vig with penaltyblog's implied probability tools, note the method
  chosen (power or Shin preferred over basic normalisation, one line why).
- Live (optional, behind a flag): The Odds API free tier for current
  match odds. Do not block the task on it, historical is what the
  baseline needs.

Baseline `odds_only.py`, emitting the compose schema:
- Team goal expectations from de-vigged 1X2 plus over/under via
  penaltyblog inversion, p_clean_sheet from opponent goal expectation
  through a Poisson zero.
- Minutes: the frozen naive 0/90 rule, deliberately, so the baseline is
  pure odds plus no ML.
- Attacking: allocate team exp goals to players by a crude fixed share by
  position and within-team price rank, constants documented. No scorer
  odds historically, say so in the module docstring.
- Defenders and keepers get odds-implied CS and concessions, which is
  where this baseline should be strong.

Validation before its harness run: odds-implied match outcome log loss vs
the realised results as a floor check, and calibration of its p_clean_sheet
in 10 bins.

Acceptance criteria:
- Join coverage >=99 percent of fixtures per season, misses listed.
- One harness run under the gating policy, report first line states
  whether compose-v0 beats the odds-only baseline. Either answer is a
  result. If compose-v0 loses on defensive components' home turf (CS-heavy
  positions), that is expected and phase 4's job is to fix it.

## Task 2: Dixon-Coles team model (`fpl/models/team/dixon_coles.py`)

Goal: replace trailing clean sheet and concession rates with a proper
team goals model. This feeds p_clean_sheet, exp_goals_conceded and the
team goal expectations task 4 allocates.

- penaltyblog's Dixon-Coles with time decay, fitted on match results
  keyed by club name, refit weekly in walk-forward fashion (fit at each
  gameweek uses only prior matches, including prior seasons with decay).
  Promoted teams enter with the fitted promoted-side prior, document how
  penaltyblog handles unseen teams and wrap it if it does not.
- Decay half-life: pick ONE value from the literature-standard range
  (penaltyblog default or xi around 0.0018 per day), document it, do not
  tune it against 2025-26. If you must choose between two candidates,
  choose on 2021-22 through 2024-25 log loss only.
- Outputs per fixture: home/away goal expectations, score matrix,
  p_clean_sheet both sides, p_win/draw/loss.
- Internal validation: match outcome log loss vs (a) the trailing-rate
  v0 proxy and (b) the de-vigged odds from task 1, per season. Beating
  the bookmaker is not expected, getting within a few percent is. CS
  calibration in 10 bins must track.
- Swap into compose behind a flag: p_clean_sheet and exp_goals_conceded
  only. One harness run, gated.

Acceptance criteria: walk-forward fit for a full season under 5 minutes,
calibration table in the report, one gated harness run with the paired
delta and sign record, first line states accept or reject.

## Task 3: Understat ingest and ID mapping (`fpl/ingest/understat.py`, `fpl/mapping/`)

Goal: player xG, xA, npxG and shots per match, joined to player_code.
This is plumbing with a known 20-30 percent time share, budget for it.

- Ingest via understatapi or penaltyblog scraper, seasons 2021-22
  through current, cached under `data/external/understat/`, per-match
  player rows.
- Mapping: seed from ChrisMusson/FPL-ID-Map and vaastav id_dict.csv,
  fuzzy-match the residual with rapidfuzz, everything unresolved goes to
  `fpl/mapping/overrides.csv` for the human. Output
  `data/curated/understat_map.parquet`: understat_id, player_code,
  confidence, source.
- Validation is mandatory and blocking: per player per season, summed
  Understat goals vs FPL goals must reconcile exactly for >=99 percent of
  players with >=90 minutes. Every mismatch listed with both names. Own
  goals and definition quirks explained in the validator's docstring, not
  hand-waved.
- Emit `data/curated/{season}/player_match_xg.parquet` joined on
  player_code and fixture (club name and date), DGW-safe.

Acceptance criteria: coverage and reconciliation numbers per season in a
report, unresolved players enumerated, zero silent drops. No harness run,
this task produces data, not predictions.

## Task 4: Poisson attack model (`fpl/models/attack/`)

Goal: replace trailing per-90 goal and assist rates with allocated,
xG-based expectations. This is the phase's main event.

Structure:
- Player share model: each player's share of his team's npxG and xA over
  a recency-decayed window (reuse the decayed-evidence machinery from the
  opponent adjustment fix, same half-life family), shrunk with empirical
  Bayes toward position x price-tier cells exactly as v0 does.
- Allocation: exp_goals for a fixture = team goal expectation from
  Dixon-Coles (task 2) x player share x finishing adjustment (player
  goals vs npxG over the decayed window, shrunk hard toward 1, cap the
  adjustment to [0.85, 1.15]). Assists analogous via xA share.
- Set-piece handling: penalty takers get the penalty component modelled
  separately (team penalty rate x conversion), replacing the v0 flat
  bump. Historical order uses the previous-season convention, live uses
  bootstrap order, sp_source carried through.
- Minutes scaling and composition unchanged: exp_minutes scales rates,
  p_appear and p_60 feed appearance and CS terms.
- Internal validation: per-position log score of realised goals under the
  predicted Poisson vs the v0 trailing rates, on 2021-22 through 2024-25
  only. Also the GW20-style sanity table: Haaland, Palmer and peers must
  sit where domain sense says.
- One gated harness run swapping exp_goals and exp_assists only.

Acceptance criteria: internal log score beats v0 on the pre-freeze
seasons, sanity table in the report, one gated harness run, first line
states accept or reject and whether the effect cleared the resolution
floor.

## Task 5: DefCon classifier and bonus v1 (`fpl/models/defcon/`, `fpl/models/bonus/`)

Smallest task, and the one to cut if the phase overruns.

- DefCon: LightGBM classifier for P(threshold hit) per player per match.
  Features: decayed defensive-action rates, position, opponent possession
  proxy (Dixon-Coles strength as a stand-in), was_home, exp_minutes.
  Training data 2025-26 only (the single DefCon season) plus current
  snapshots, so keep the model small and heavily regularised, and say in
  the report that one season of training data caps confidence. Calibrate.
  Internal bar: beat the v0 trailing threshold rate on log loss,
  walk-forward within 2025-26 is acceptable given the data constraint,
  documented as a weaker protocol.
- Bonus v1: expected bonus from a BPS proxy built from predicted
  components under the 2026/27 BPS weights in a small module
  `fpl/models/bonus/bps_2026_27.py` (separate from the points rules
  module, same single-source principle). Rank the three bonus slots per
  fixture via simulation or a Plackett-Luce style approximation, keep it
  simple, document the approximation error against realised 2021-24
  bonus.
- One gated harness run for the pair together (they are small and both
  touch the same tail of EP, a joint swap is acceptable, note it).

Acceptance criteria: both internal bars stated and met or missed in the
first line, calibration table for DefCon, one gated harness run.

## Out of scope

Sequential backtest (phase 5), multi-GW MILP and autosub weighting
(phase 6), chips and EO (phase 7), scorer odds history, news ingestion,
price modelling, correlated match simulation. Finish early: strengthen
validation and reports, do not start new modules.
