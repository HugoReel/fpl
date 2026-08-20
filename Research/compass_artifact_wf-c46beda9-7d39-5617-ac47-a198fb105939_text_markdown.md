# FPL ML + Optimisation System: Research and Initial Build Plan (2026/27)

## TL;DR
- Build component-based, not points-regression-based: model expected minutes first, decompose points into appearance + attacking (Poisson) + team clean sheet (Dixon-Coles) + DefCon, then compose through the current 2026/27 scoring function. This survives rule drift and gives you distributions for free.
- The 2026/27 rules are confirmed and stable: 15-man squad, 100.0m budget, max 3 per club, bank up to 5 free transfers, 8 chips (two each of Wildcard/Free Hit/Triple Captain/Bench Boost split across halves at GW19), DefCon points unchanged, BPS tweaked, projected bonus live after 20 minutes. No assistant manager chip.
- Your bar to beat is an odds-only baseline and FPL's own ep_next, then OpenFPL (open-source, MIT-licensed, rivals commercial FPL Review on high-return players). End-to-end success = realised points of the optimised team, benchmarked to a top-100k finish in walk-forward backtests, not RMSE.

## Key Findings
- Expected minutes is the single biggest error driver and belongs in v0.1. OpenFPL, the leading open-source model (Daniel Groos, arXiv:2508.09992, 29 Jul 2025), explicitly identifies its one weakness versus commercial FPL Review as low-return prediction, because it "dispenses with proprietary 'expected minutes' projections, using instead the categorical availability tags provided by the FPL API." That is the gap you close cheaply.
- OpenFPL is MIT-licensed, runnable off the shelf (pip install, run play.ipynb), and beats commercial FPL Review on Tickers and Haulers (the high-return players that drive rank) while losing on Zeros and Blanks. Use it as a reference architecture and a hard baseline, not something to reinvent. FPL is played by more than 11 million people, so the commercial models it rivals are genuinely strong.
- Data plumbing is the real cost. FPL-to-Understat ID mapping and snapshotting the mutable API are where solo builders lose time. Community mapping tables exist (ChrisMusson/FPL-ID-Map, vaastav id_dict.csv), so do not hand-roll from scratch.
- HiGHS is the correct open-source solver now, not CBC. On the MIPLIB2017 benchmark HiGHS solved 24 instances to CBC's 5 and is roughly 2 to 2.5x faster on scaled geometric mean, and it is actively developed. CBC and GLPK are effectively uncompetitive on non-trivial MILP. Access HiGHS through highspy directly or via PuLP.
- penaltyblog gives you Dixon-Coles, bivariate Poisson, odds de-vigging and scrapers (FBRef, Understat, Club Elo) in one pip install. Do not implement Dixon-Coles by hand.

## Details

### 1. System overview and objectives

You are building an end-to-end system that ingests FPL and football data weekly, predicts each player's points distribution for the next N gameweeks, optimises the 15-man squad and starting XI under FPL constraints, plans transfers across a rolling horizon with banked free transfers, decides chip timing, and backtests all of it on historical seasons without lookahead.

Design principle that overrides everything: the model selection metric is the realised points of the optimised team in walk-forward backtests, not per-player RMSE. A model with worse RMSE that picks better captains and finds better differentials can win. RMSE is a diagnostic, not the objective.

Explicit success criteria, staged:
- Tier 0 (must hit or the project is pointless): beat a naive "last 5 games mean points" baseline on realised optimised-team points across backtested seasons. This is OpenFPL's own baseline, so it is a fair floor.
- Tier 1 (the real bar): beat an odds-only baseline (bookmaker match odds and scorer odds plus a naive minutes rule, no ML) on realised optimised-team points. If you cannot beat odds-only, your ML is adding nothing.
- Tier 2: match or beat FPL's own ep_next field and OpenFPL as ranking signals, measured by Spearman within position and precision-at-k for captaincy.
- Tier 3 (end-to-end): in walk-forward simulation the optimised season total maps to a top-100k finish equivalent, with error bars from resampled outcomes. Top-10k historically requires roughly 2,300 to 2,450 points, a gameweek average around 60 to 65; top-100k sits below that. Use points-to-rank tables to convert, and note thresholds drift year to year with entrant count.

Be honest that variance dominates. Season standard deviation is tens of points, so a single backtested season cannot separate skill from luck. Every accept/reject decision goes through paired comparisons and resampled error bars (see section 6).

### 2. FPL rules 2026/27 as verified (the ground truth the system encodes)

Squad and budget:
- 15 players: 2 GKP, 5 DEF, 5 MID, 3 FWD. Max 3 players per real club. Budget 100.0m.
- Starting XI: 1 GKP, at least 3 DEF, at least 1 FWD, 11 players total. Valid formations include 3-4-3, 3-5-2, 4-4-2, 4-3-3, 5-3-2, 5-4-1, 4-5-1 (any split satisfying min 3 DEF, min 1 FWD, exactly 1 GKP).
- Captain doubles, vice-captain deputises if the captain plays 0 minutes.

Transfers:
- 1 free transfer per gameweek, bank up to a maximum of 5. This rule dates from 2024/25 and the Premier League has confirmed it is unchanged for 2026/27 ("You will still be able to roll up to five free transfers"). Extra transfers cost -4 points each.
- If you play a Wildcard or Free Hit with banked transfers, you keep those banked transfers for the following gameweek (no longer reset to 1). This is a recent and important change for the planner.
- No extra AFCON transfers this season because AFCON is not held until June/July 2027.

Chips (confirmed 2026/27):
- Eight chips total: two each of Wildcard, Free Hit, Triple Captain, Bench Boost.
- The first set must be played before the GW19 deadline (13:30 GMT, Saturday 2 January) and cannot be carried into the second half; the same chips refresh for GW20 onward.
- Only one chip per gameweek. Free Hit cannot be played in GW1, and if you use the first Free Hit in GW19 you cannot use the second in GW20.
- No Assistant Manager chip (introduced mid-2024/25, scrapped for 2025/26 and still gone in 2026/27). No new chips for 2026/27.

Scoring, including 2025/26 and 2026/27 changes:
- Defensive contributions (DefCon), unchanged for 2026/27: DEF earn +2 for reaching 10 combined clearances, blocks, interceptions, tackles (CBIT) in a match. MID and FWD earn +2 for reaching 12 CBIRT (CBIT plus ball recoveries). Capped at +2 per player per match; it is a threshold bonus, not per-action. New API fields defensive_contributions and clearances_blocks_interceptions exist. This is a material, stable point source (for example Marcos Senesi banked 50 DefCon points in 2025/26).
- Assist definition simplified from 2025/26.
- Penalty goal BPS normalised: 12 BPS for scoring a penalty regardless of position (previously 24/18/12 for FWD/MID/DEF).
- BPS tweaks for 2026/27 to reduce overlap with DefCon: players no longer lose 1 BPS when tackled; CBI now earns 1 BPS per three actions instead of per two; goalkeeper save BPS reworked (3 BPS for a save from inside the box, 2 BPS for "any other save" replacing the old outside-the-box category, 1 BPS for saving a big chance, 8 BPS for a penalty save which is 7 plus 1 big-chance-saved). Net effect: DefCon magnets are now less rewarded in bonus, so DefCon and bonus are more separable, and goalkeepers, full-backs and attackers gain bonus upside.
- Live projected bonus points appear after 20 minutes of each match in 2026/27; mini-league standings and overall rank update live.
- Gameweek "lockdown" (scores final) moves to 9am UK the day after the final match of the gameweek, later than the old one-hour-after-final-whistle cutoff. Late DefCon/bonus corrections now count, which matters for backtest fidelity when replaying older seasons scored under the old cutoff.

Price change mechanics:
- Prices change once daily overnight (around 2:00 to 3:00am UK), frozen during live gameweeks. Max 0.1m change per night; community consensus is a max 0.3m move per gameweek.
- Driven by net transfers (in minus out) against a hidden ownership-scaled threshold. Community rule of thumb is roughly 140k net transfers for a mid-priced player, lower for cheap/low-owned players, higher for premiums. FPL now publishes an official Price Change Predictor for 2026/27. Selling price carries the standard 50% sell-on fee on profit.

### 3. Data architecture

Sources and current status (August 2026):
- Official FPL API: bootstrap-static (players, teams, prices, ownership, ep_next, status, set-piece fields), fixtures, element-summary/{id}, event/{gw}/live. The authoritative live source. Mutable, so snapshot it.
- vaastav/Fantasy-Premier-League: the canonical historical archive, active, now includes a 2026-27 folder. Provides cleaned_players, per-gw merged_gw.csv, an understat/ directory and id_dict.csv mapping Understat to FPL IDs. Documented caveat in its own DATA_DICTIONARY: the xP column is scraped from ep_this after the gameweek and may reflect post-match info, so shift(1) within each player or drop it before using as a feature.
- olbauday/FPL-Core-Insights and the related FPL-Elo-Insights: newer curated datasets fusing FPL API with Opta-like match stats and ClubElo ratings, aligned on official FPL IDs, including 2026/27. Good complement for team strength.
- Understat: free xG, xA, xGChain, xGBuildup, key passes, npxG. Access via the understatapi Python client or penaltyblog scrapers. This is the main free advanced-stat source and the one OpenFPL relies on.
- penaltyblog scrapers: FBRef, Understat, Club Elo, football-data.co.uk in one library, plus odds de-vigging.
- ID mapping: ChrisMusson/FPL-ID-Map (FPL IDs to Understat and other sites), vaastav id_dict.csv, and parmacalcio1913/players-matcher (TF-IDF fuzzy matcher) as tooling.
- Set piece and penalty takers: FPL's own official set-piece fields in bootstrap now expose pecking orders. Community lists at Fantasy Football Scout, RotoWire, allaboutfpl, Full90, footieguide, OneFPL. Scrape the official field as ground truth and use community lists to fill gaps.
- Top 10k effective ownership: LiveFPL (plan.livefpl.net/top10k and /EO), fotprem, Fantasy Football Pundit. Top-10k EO is usually available within about 30 minutes of the deadline, broader top-100k a few hours later. Known pre-deadline for your own decision, so no leakage when used as a live signal.
- Points-to-rank tables: derivable from vaastav season totals; community references (FPL Oracle, GiveMeSport) give top-1k/10k/100k thresholds.
- Odds: The Odds API (free tier 500 requests/day, EPL match odds in decimal), with scorer/BTTS markets on paid tiers; The Odds API also starts at 30 USD/month for hobbyists. Check The Chance publishes free odds-derived FPL points and clean-sheet probabilities. Betting scorer odds are the cleanest single input for attacking rates.
- Commercial benchmarks (for reference/validation, mostly paid): FPL Review Massive Data Model (live for 26/27, with a free model too, and previously shown to be more accurate than other leading services), Fantasy Football Fix, Fantasy Football Scout RMT projections. theFPLkiwi publishes free expected points and expected minutes projections on GitHub in fplreview-compatible format, which is gold for a solo dev.

Snapshotting design (do this from day one, near-zero cost, critical):
- Every scheduled run, pull bootstrap-static, fixtures, and event/{gw}/live and write immutable timestamped Parquet to a raw/ zone: raw/{season}/{gw}/{endpoint}_{iso8601}.parquet. Never mutate. These snapshots become your future backtest ground truth because the live API overwrites itself.
- Layer a curated/ zone (typed, joined, ID-mapped) and a features/ zone (model-ready) on top. Partition by season and gw.
- Store the exact information set available at each deadline so backtests can replay "as known at the time."

Unified schema (minimum viable):
- player_gw: player_id, season, gw, team_id, opponent_id, was_home, minutes, starts, goals, assists, xg, xa, npxg, shots, key_passes, cbit, cbirt, defensive_contributions, saves, goals_conceded, yellow, red, bonus, bps, total_points, price, selected_by_pct, transfers_in, transfers_out, status, chance_of_playing, ep_next.
- team_gw: team_id, season, gw, xg_for, xga, elo, form, home/away splits.
- fixtures: fixture_id, season, gw, home_team, away_team, kickoff, fdr_home, fdr_away, finished, is_dgw, is_bgw, reschedule_reveal_ts.
- meta: set_piece_order (pens, corners, direct fk) per player, injury/suspension flags with as-of timestamps.
- ownership: player_id, gw, overall_own, top10k_own, top10k_captain_pct, EO.

ID mapping approach: start from ChrisMusson/FPL-ID-Map and vaastav id_dict.csv, fill gaps with a fuzzy matcher (rapidfuzz or players-matcher TF-IDF), then keep a hand-maintained overrides CSV for the residual. Validate every join by reconciling season goal totals between FPL and Understat per player; mismatches flag bad maps. Treat any -1 ID as unmatched and exclude from cross-system joins.

### 4. Modelling plan

Minutes model first (v0.1, first-class citizen). Expected minutes dominates expected-points error, and it is exactly where the best open-source model is beaten by commercial services. Model, per player per gw:
- P(start), P(60+ minutes | start), P(any appearance | not start i.e. sub cameo), and from these expected minutes.
- Compose: E[minutes] = P(start) * E[min | start] + P(sub) * E[min | sub]. Use these to weight all per-90 rates and to compute appearance points (1 point for playing, 2 for 60+).
- Features: recent starts rolling (1/3/5), was in last XI, minutes trend, rotation flags, status/chance_of_playing from API, fixture congestion (days rest, midweek European/cup game), price tier as a proxy for nailedness, manager rotation tendency, pre-season minutes tracker signal early in season. Set-piece and penalty duty also proxy nailedness.
- Start with gradient boosting classifiers (LightGBM) for P(start) and P(60+), calibrate probabilities (isotonic or Platt). Add a manual override file for known team news.

Component decomposition (target architecture by v0.2). Instead of regressing total_points directly, model the pieces and compose through the 2026/27 scoring function:
- Appearance points from the minutes model.
- Attacking events: Poisson-style per-90 rates for goals and assists per player, shrunk toward position/role priors, scaled by expected minutes and adjusted by opponent strength and home/away. Blend player xG/xA rates with team-level attack and bookmaker scorer odds where available.
- Team clean sheet probability: Dixon-Coles goals model (via penaltyblog) or odds-implied clean sheet probability. Feeds GK and DEF clean-sheet points and conceding penalties, plus save points for GK.
- DefCon points: model P(hit CBIT/CBIRT threshold) per player per match as a calibrated classifier on rolling defensive-action rates, times +2. This is now a material, stable point source, especially for centre-backs and defensive midfielders, and it interacts with the 2026/27 BPS change (CBI now worth less BPS, so DefCon and bonus are more separable).
- Bonus: model expected bonus from projected BPS components under the 2026/27 BPS scheme. Approximate early, refine later.
- Compose all pieces through a scoring module that encodes 2026/27 rules exactly. Because scoring is applied at the end, rule changes only touch that module, and you get a points distribution (via simulation) rather than a point estimate.

Note that OpenFPL itself uses direct regression, not decomposition, and still rivals FPL Review. Decomposition is the more robust choice for you because it survives rule drift, exposes distributions for captaincy and chip EV, and lets you inject odds cleanly. Prove it beats direct regression on realised optimised-team points in pre-build experiment 3 before committing.

Scoring rule drift decision (make it in v0.1): historical points 2016 to 2025 were earned under different rules (no DefCon pre-2025/26, different BPS, different penalty BPS, different GK save scheme). Go component-based and always apply the current scoring function to component predictions. For backtests that need historical "truth," recompute historical points under each season's actual rules from event data rather than trusting stored totals across rule regimes. Document any season where component data is incomplete.

Features summary (beyond minutes):
- Rolling form windows (short 1/3/5 and long 10/38 matches), mirroring OpenFPL's multi-horizon mean aggregation (it represents each historical feature over 1, 3, 5, 10 and 38 match horizons).
- Team strength (Elo, xG/xGA), opponent strength, home/away, FDR, plus Understat PPDA and deep-completion metrics as OpenFPL does.
- Position-specific stat sets (GK saves and clean sheets; DEF clean sheets and DefCon; MID/FWD attacking and DefCon). OpenFPL uses one feature set for GK, one shared by DEF/MID/FWD and one for the now-defunct AM chip; you only need the first two.
- Set-piece and penalty flags: cheap, high-signal, in v0.1. Penalty duty materially shifts goal expectation; corner/free-kick duty shifts assist and defender-goal expectation.
- Promoted-team handling via priors (no PL history, so shrink to promoted-team baselines).
- Rotation/congestion features.

Validation protocol:
- Time-based only. Train on early seasons, validate on later. Expanding window intra-season.
- Walk-forward: freeze hyperparameters on pre-backtest seasons, then retrain weights on an expanding window at each season boundary during the backtest. Never tune hyperparameters on a season you will report backtest results for. OpenFPL is a concrete template: hyperparameters tuned via K-Best Search over 5 cross-validation folds (26 teams allocated across folds), trained 2020-21 to 2023-24, evaluated prospectively on 2024-25 data that was "not gathered until after completing development."
- Metrics: RMSE/MAE as diagnostics; Spearman within position and precision-at-k for captaincy as ranking checks; realised optimised-team points as the decision metric.

Hard baselines (must be in place in v0.2 before you add richer features):
- FPL's own ep_next.
- Player price as a ranking signal.
- Bookmaker odds-derived expected points.
- "Last 5" mean points (OpenFPL's own baseline).
- OpenFPL predictions (MIT-licensed, drop-in) and, where obtainable, FPL Review-class public projections (theFPLkiwi free set is fplreview-format).
Every feature addition is measured against these. If it does not beat them on realised optimised-team points with error bars, it does not ship.

Concrete reference target from OpenFPL (arXiv:2508.09992, Table 4, one gameweek ahead, RMSE with MAE in parentheses): Tickers Last5 2.136(1.645) / FPL Review 1.594(1.227) / OpenFPL 1.517(1.127); Haulers Last5 5.613(4.709) / FPL Review 5.172(4.381) / OpenFPL 5.142(4.317); Zeros FPL Review 0.689 vs OpenFPL 0.818; Blanks FPL Review 1.189 vs OpenFPL 1.291. OpenFPL beats the commercial model on the high-return players that drive rank and loses on non-playing/low-return players precisely because it lacks expected minutes. The lesson for you: attack the minutes and low-return gap and you can exceed the best open model.

### 5. Optimisation plan

Solver choice: HiGHS via highspy (or through PuLP). HiGHS is the strongest open-source MILP solver now, roughly 2 to 2.5x faster than CBC on scaled benchmark geometric mean and solving far more instances (24 vs 5 of the MIPLIB2017 benchmark set), and it is actively maintained; CBC and GLPK are otherwise uncompetitive. Single-GW solves are sub-second, multi-GW horizons a few seconds. OR-Tools is a fine alternative. The sertalpbilal/FPL-Optimization-Tools reference stack already uses HiGHS via highspy, so you are in good company. Keep the model in a solver-agnostic layer so you can swap.

Core MILP (single GW first, then multi-GW):
- Decision variables: squad membership x_p (15), starting XI y_p, captain c_p, vice-captain v_p, transfers in/out per gw, chip-usage binaries per gw.
- Constraints: 2/5/5/3 by position, max 3 per club, budget with dynamic prices and the 50% sell-on fee, XI formation constraints (1 GK, >=3 DEF, >=1 FWD, 11 total), captain and vice exactly one each and in XI.
- Objective: maximise expected points over horizon minus 4 per extra transfer, with a per-gw discount factor on future gameweeks to reflect forecast decay. A decay base around 0.84 (as used in the sertalpbilal multi-period solver) is a sane starting point; tune it.

Banked FT state 0 to 5: add an integer state variable ft_g per gameweek with transition ft_{g+1} = min(5, ft_g - used_g + 1), and cost extra transfers beyond ft_g at -4. Add a terminal-value bonus per banked FT at the horizon end so the optimiser does not burn transfers wastefully (flexibility value). Re-solve on a rolling horizon each week. This is the single most important departure from naive one-GW solving.

Autosub and bench weighting (v0.3): weight bench players' expected points by their autosub entry probability, a function of starters' P(0 minutes) and bench order. Linearise for MILP with fixed per-slot coefficients (bench GK, bench slots 1/2/3), a standard community trick. Vice-captain EV = captain_points * P(captain plays) + vice_points * P(captain no-show) * P(vice plays). This makes the optimiser value nailed benches and sensible VC picks correctly.

Chip optimisation (v1.0): do NOT put monolithic full-season chip decision variables in one MILP. Two-stage instead:
- Stage 1: enumerate candidate chip gameweeks from DGW/BGW structure and fixture swings (Bench Boost and Triple Captain in DGWs, Free Hit in your worst BGW, Wildcards around fixture turns).
- Stage 2: for each candidate scenario, solve a medium-horizon MILP and compare realised expected value. Pick the best scenario. This keeps the problem tractable and interpretable, and respects the GW19 half-season split constraint.

Risk and EO objective (v0.3): add top-10k effective ownership as a live input. Two use modes: (a) as a decision-layer signal feeding a mini-league-vs-overall-rank risk objective with a parameter lambda that penalises variance or rewards EO-differential depending on whether you are chasing or protecting rank; (b) never as a training feature for the points model (low value, skip). For a mini-league you often want to minimise EO-differential to the template; for overall rank climbing from behind you want deliberate differentials. Expose lambda as a knob.

Human override constraints (v0.3, operationally essential): lock (force in XI), ban (force out), and manual expected-minutes overrides. Apply these as hard constraints/overrides after model output and before solve, so you can inject team news the model has not seen.

### 6. Backtesting and evaluation

Walk-forward protocol: hyperparameters frozen on pre-backtest seasons; weights retrained on expanding window at season boundaries; no tuning on reported seasons. Replay each gameweek using only the information set snapshotted as available at that deadline.

DGW/BGW lookahead handling: assuming fixtures are known from season start inflates chip EV, because DGWs/BGWs are announced mid-season when cups resolve. Reveal reschedules with a fixed lag (about 4 gameweeks) in backtests, or run both regimes (full-knowledge and lagged) and report the gap so you know how much of your chip performance is hindsight.

Evaluation variance and the acceptance gate: a single-trajectory backtest cannot separate skill from luck (season SD is tens of points). Therefore:
- Paired comparisons: same seasons, same information sets, change exactly one component, measure the delta.
- Resampled outcome simulations: simulate match outcomes and player events many times to put error bars on season totals and on the paired delta.
- A change is accepted only if it improves realised optimised-team points with error bars that clear zero, not on a single number.

Points-to-rank mapping: convert backtested season totals to rank equivalents using historical points-to-rank tables (top 10k roughly 2,300 to 2,450; top 100k lower). Report the finish-equivalent, not just raw points, and caveat that thresholds move with entrant count.

Historical injury info is mostly not reconstructable "as known at the time." Use conservative proxies (0 minutes last GW after previously starting implies doubt), vaastav archived status flags where present, and document backtests as an upper bound on achievable performance.

Correlation (post-v1.0, but architect for it now): simulate at match level (draw the scoreline, then allocate player events conditional on it) rather than sampling players independently. This matters for captain/TC/bench-boost EV and any risk objective because your own players' returns are correlated within a match. Keep the scoring/composition layer simulation-ready so you can slot this in later.

Top-10k behaviour validation study (a weekend of work, high value as a sanity bound): one-off descriptive study of past top-10k managers' hit frequency, chip timing windows, team-value evolution and bench spend, sourced from LiveFPL and crawling picks for entry IDs off the overall standings. Use it to bound sane optimiser behaviour (if your optimiser wants 15 hits a season, something is wrong). Use top 10k to 50k, not top 1k, to avoid survivorship bias (top 1k is selected on variance/luck).

### 7. Pre-build experiments (do these before heavy building)

1. Error attribution: minutes vs per-90 rates. On one archived season, swap true minutes vs modelled minutes into the points composition and measure how much error each explains. Effort: 2 to 3 evenings. Expected result: minutes dominates, justifying the minutes-first design. OpenFPL's own results predict this outcome.
2. Odds-only baseline. Build bookmaker match and scorer odds plus a naive minutes rule, no ML, and score it on realised optimised-team points. Effort: a weekend. This is the bar the stats model must beat; if it cannot, stop and rethink.
3. Direct regression vs component decomposition. Score both on realised optimised-team points, not RMSE. Effort: 3 to 4 evenings once data is flowing. Decides your modelling backbone.
4. Rolling re-solve at horizons 1/3/6 with and without FT banking. Quantify the value of multi-GW planning and banked transfers. Effort: 2 to 3 evenings once the MILP exists.
5. Quantify DGW lookahead cost. Run the backtest with full fixture knowledge vs 4-GW lagged reveal and report the chip-EV gap. Effort: 2 evenings. Tells you how much to trust chip results.

### 8. Tech stack and repo structure

Stack: Python 3.11+, pandas/polars, LightGBM (primary) with XGBoost/CatBoost optional, scikit-learn, penaltyblog (Dixon-Coles, bivariate Poisson, odds de-vig, scrapers), highspy or PuLP for MILP, Parquet via pyarrow for storage, DuckDB for ad-hoc querying over Parquet. Experiment tracking: self-hosted MLflow backed by SQLite (Apache-2.0, free, runs on one machine, and the recommended cost-to-capability choice for a solo dev running under 50 experiments a month); a structured runs/ directory of Parquet is an even lighter fallback. Scheduling: cron or GitHub Actions for the weekly pipeline. uv for env/dependency management (also what the reference FPL solver uses).

Repo layout:
```
fpl/
  ingest/        # API pulls, snapshotting to raw/ Parquet, schema validation
  data/          # raw/ curated/ features/ zones (gitignored or DVC-tracked)
  mapping/       # id maps, overrides.csv, join validators
  features/      # feature builders, rolling windows, set-piece flags
  models/
    minutes/     # P(start), P(60+), expected minutes
    attack/      # goal/assist Poisson rates
    team/        # dixon_coles, clean sheet
    defcon/      # threshold classifiers
    bonus/       # BPS composition (2026/27 scheme)
    compose.py   # applies 2026/27 scoring to components -> points distribution
  scoring/       # rules_2026_27.py (single source of truth for scoring)
  optimise/
    milp.py      # squad + XI + captain + transfers + banked FT state
    chips.py     # two-stage scenario enumeration
    overrides.py # lock/ban/xmin overrides
  backtest/      # walk-forward engine, info-set replay, paired comparisons, resampling
  eval/          # metrics, points-to-rank, top10k study
  baselines/     # ep_next, price, odds-only, last5, openfpl adapter
  ops/           # weekly scheduler, staleness/schema checks, alerting
  cli.py
```
Keep scoring/rules_2026_27.py as the only place scoring logic lives, so a rule change is a one-file diff. Wrap OpenFPL as one of the adapters under baselines/ (its MIT license permits this).

### 9. Phased roadmap (realistic for evenings/weekends alongside a full-time job)

v0.1 Minimal predictor plus single-GW optimiser (4 to 6 weeks). Deliverables: raw API snapshotting live from day one; schema validation on ingest; minutes model (P(start), P(60+), expected minutes); set-piece/penalty flags; a simple per-90 attacking model composed through scoring/rules_2026_27.py; single-GW MILP via HiGHS producing a legal squad and XI; scoring-rule-drift decision made and documented (component-based, current scoring applied at end). Definition of done: an end-to-end run picks a legal GW team from live data and writes a snapshot; the minutes model is calibrated and beats a naive start rule.

v0.2 Richer features, backtesting, hard baselines, decomposition (6 to 8 weeks). Deliverables: all hard baselines wired (ep_next, price, odds-only, last5, OpenFPL adapter); component decomposition (attack Poisson + Dixon-Coles clean sheets + DefCon + bonus) as the primary model; corrected FT rules (bank up to 5, chip-keeps-banked-FTs); walk-forward backtest engine with info-set replay; ID mapping validated via goal-total reconciliation. Definition of done: the backtest runs a full historical season without lookahead, and at least one feature/model change has passed the paired-comparison gate.

v0.3 Transfer planning depth, bench value, risk (6 to 8 weeks). Deliverables: multi-GW MILP with banked-FT state 0 to 5, discounting and terminal FT value; autosub bench weighting and VC EV; lock/ban and expected-minutes overrides; top-10k EO input and a risk objective with tunable lambda; weekly operational pipeline around deadlines (team news Friday, deadline Saturday morning) with staleness checks. Definition of done: the system produces a defensible multi-week transfer plan you would actually follow, and EO-aware risk toggling demonstrably changes picks.

v1.0 Chips, evaluation rigor, polish (8 to 12 weeks). Deliverables: two-stage chip scenario enumeration and comparison respecting the GW19 half-season split; paired-comparison evaluation gate formalised with resampled error bars; DGW-lookahead-cost experiment folded into reporting; points-to-rank finish-equivalent reporting; top-10k behaviour validation study. Definition of done: a full-season walk-forward backtest reports a finish-equivalent with error bars, chip timing is chosen by the two-stage method, and results clear the odds-only bar with confidence. Defer news/sentiment ingestion and full match-level correlation simulation past v1.0 (the architecture already allows correlation).

### 10. Risks and open questions
- Minutes is hard and high-variance; if your minutes model is weak the whole system underperforms. Mitigate with the manual override file and pre-season minutes signals.
- API schema drift every season will break ingest; schema validation on ingest is not optional.
- ID mapping rot as players transfer; keep the overrides file and goal-total validator running.
- Odds coverage for scorer/BTTS markets on free tiers is limited; you may need a paid Odds API tier (from 30 USD/month) or Check The Chance scraping for full attacking-rate priors.
- Backtest fidelity is capped by non-reconstructable historical team news and by the GW19/lockdown-timing and DefCon rule-regime differences across seasons; always report backtests as upper bounds.
- Chip EV is the most lookahead-sensitive result; trust it least and always report the lagged-reveal gap.
- OpenFPL is a v1 preprint (not peer-reviewed) and does not publish an exact training-sample count; treat its numbers as a strong reference, not gospel.
- Variance means you can do everything right and still have a mediocre season; judge the process by paired deltas and error bars, not one trajectory.