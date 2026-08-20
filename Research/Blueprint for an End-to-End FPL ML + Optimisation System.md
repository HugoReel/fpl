# Blueprint for an End-to-End FPL ML + Optimisation System

## Executive Overview

This report outlines a research and implementation blueprint for an end-to-end Fantasy Premier League (FPL) system that forecasts player points, optimises squads and starting XIs, plans transfers over a season, and schedules chip usage (wildcard, free hit, bench boost, triple captain). The focus is a single technically literate developer working primarily in Python with basic machine learning and integer programming knowledge.[^1][^2]

The blueprint is organised into problem/formalisation, data, feature engineering, modelling, optimisation (single- and multi-gameweek), backtesting, compute/stack, and learning from existing projects, followed by a phased roadmap and resource list.[^3][^4][^5]

## 1. FPL Rules and Optimisation Problem

### 1.1 Core FPL rules relevant to modelling

Key constraints and mechanics of the official Fantasy Premier League game include:[^6][^7][^1]

- **Squad structure**: 15 players – 2 goalkeepers (GKP), 5 defenders (DEF), 5 midfielders (MID), 3 forwards (FWD), with a maximum of 3 players from any one Premier League club.
- **Starting XI and bench**: Each gameweek (GW) you select 11 starters in a valid formation (e.g. 3–4–3, 4–4–2, 3–5–2) respecting a minimum of 1 GKP, 3 DEF, 2 MID, 1 FWD; 4 remaining players are on the bench in a ranked bench order.
- **Captaincy**: One captain (C) whose points are doubled and one vice-captain (VC) who takes over if the captain does not play at all.
- **Budget**: Squads are constrained by a total budget (e.g. 100.0m) and player prices that can change across the season, based roughly on ownership and transfers.[^1]
- **Transfers**: Each GW you receive a certain number of free transfers (usually 1, occasionally 2); unused free transfers can roll once, usually up to a maximum of 2 in the standard game, while some updates allow saving more in some formats; each additional transfer beyond the free allowance costs −4 points.[^2][^1]
- **Scoring**: Points depend on position and match events.[^7]
  - Appearance (1 point for up to 60 minutes, 2 points for 60+ minutes),
  - Goals (GKP: 10, DEF: 6, MID: 5, FWD: 4),
  - Assists (3),
  - Clean sheets (GKP/DEF: 4, MID: 1),
  - Goals conceded (GKP/DEF: −1 per 2 goals conceded),
  - Defensive contributions bonuses (e.g. 2 points for defenders reaching specific defensive contribution thresholds),
  - Save, penalty save and penalty miss points,
  - Cards and own goals (negative points),
  - Bonus points (1–3) allocated via an Opta-based Bonus Points System.
- **Chips/boosters**: Wildcard (unlimited transfers that week, no hits), Bench Boost (bench points count), Triple Captain (triple captain’s score), Free Hit (unlimited transfers for one week with reversion after).[^8][^2][^1]
- **Recent rule updates**: As of 2026/27, there are live projected bonus updates and some specific BPS changes (e.g. new BPS scheme for goalkeepers, CBIs, penalty save BPS).[^9][^8][^6]

These rules define both the state space (squad, budget, chips) and the feasible actions (transfers, captaincy, chip plays) for optimisation.

### 1.2 Translating rules into an optimisation problem

At a high level, the system’s decision-making per planning horizon can be seen as a mixed-integer optimisation problem:

- **Decision variables** (per gameweek and sometimes per player):
  - Binary ownership variable: whether a player is in the 15-man squad.
  - Binary lineup variable: whether a player is in the starting XI.
  - Captain and vice-captain variables: which players receive captaincy.
  - Transfer variables: whether a player is transferred in or out between GWs.
  - Chip variables: whether a chip (wildcard, bench boost, triple captain, free hit) is played in a specific GW.[^4][^5][^3]

- **Constraints**:
  - Squad size and composition: 15 total, 2/5/5/3 by position, ≤3 from any club.
  - Formation and starting XI: exactly 11 starters, formation constraints, 1 GKP in XI.
  - Budget: sum of player prices ≤ budget; prices can be updated week-by-week.
  - Transfer limits: free transfers per GW, hits cost points, and chip-specific allowances (e.g. wildcard lifts transfer limits).
  - Chip use limits: each chip at most once (or once per half-season for formats with two sets), with mutual exclusivity per GW.

- **Objective function**:
  - Maximise expected total FPL points over a planning horizon, including:
    - Expected points from starting XI and bench (when Bench Boost is active),
    - Captaincy multiplier (double or triple points),
    - Minus transfer hits.
  - Optional risk-aware formulations (e.g. penalise variance or downside risk) can be added.

The core optimisation is to choose ownership, starting XI, captaincy and transfers under these constraints to maximise expected or risk-adjusted points over one or several GWs.[^3][^4]

## 2. Data Requirements and Sources

### 2.1 Player-level data per gameweek

For robust modelling, the system requires detailed per-player, per-gameweek data:[^10][^11]

- Minutes played, starting vs substitute, and 60+ minute flags.
- Goals, assists, shots, shots on target, xG, xA, key passes.
- Clean sheets, goals conceded where relevant.
- Defensive metrics (tackles, interceptions, blocks, CBIs) and BPS-related stats.
- Cards (yellow/red), own goals, penalty events (taken, scored, missed, saved).
- FPL-specific fields: position, price at GW, ownership, total points.
- Bonus points awarded.
- Injury and suspension indicators, if available.

### 2.2 Team-level data

Team-level statistics help capture global strength and fixture difficulty:[^12][^11]

- Goals for and against, xG and xGA.
- Shot counts for and against, big chances, shots in box.
- Form measures (e.g. last N games rolling metrics).
- Home/away splits.
- Tactical changes and manager data where available.

### 2.3 Fixture data

For forward-looking predictions and planning, fixture information is essential:[^13][^1]

- Full fixture list per season, with gameweek, home team, away team.
- Double and blank gameweeks.
- Official FPL “fixture difficulty rating” (FDR) per team, per match.
- Optional: betting odds (match outcome, goals, anytime scorer) to build market-implied expectations.

### 2.4 Meta data

Meta data supports better availability and rotation modelling:[^13][^12]

- Injury reports and expected return dates.
- Suspension status and accumulated bookings.
- Rotation risk proxies (e.g. minutes congestion, European competitions).
- Manager tendencies, especially for rotation.

### 2.5 Public data sources

Practical sources include:

- **Official FPL API (undocumented but stable)**: Public endpoints provide bootstrap data, fixtures, per-player summaries and live GW stats.[^13]
  - `bootstrap-static/` for players, teams, and global settings.
  - `fixtures/` and `fixtures/?event={gw}` for fixture lists and FDR.[^13]
  - `element-summary/{player_id}/` for player history and upcoming fixtures.
  - `event/{gw}/live/` for per-GW player stats.
  - These endpoints are suitable for both current-season data collection and, with some work or third-party archives, historical reconstruction.

- **Community-maintained APIs / wrappers**:
  - Python, JS and other language wrappers (e.g. open-source `fpl-api` and related projects) simplify interacting with the official endpoints.[^14][^15]
  - Pros: easier integration and typed responses; cons: may lag behind official changes.

- **Historical open datasets**:
  - GitHub repositories that archive per-player, per-gameweek data across multiple seasons (for example, long-running repositories that generate CSVs from FPL data).[^11]
  - Kaggle datasets that provide multi-season FPL player data from 2016–2024 with per-gameweek statistics.[^10]
  - These provide 8+ seasons of data at a granularity of player–GW, ideal for training and backtesting.

- **Existing ML/optimisation projects**:
  - Prediction-focused repos that build ML models on FPL data.[^16][^17]
  - Optimisation-focused repos (linear programming team optimisers) that assume expected points as input.[^5][^3]
  - OpenFPL (academic/open-source forecasting method) providing models and code for competitive FPL forecasting.[^18]

- **External stats providers**:
  - Sites that publish xG/xA and advanced metrics (e.g. Understat-type data) or FPL-specific projection services; some open projects integrate these with FPL modelling.[^19][^20][^12]

### 2.6 Data source characteristics

- **Seasons covered**:
  - FPL-based open datasets commonly cover between 6 and 10 recent seasons (e.g. 2016–2024). This is sufficient for training models that need thousands of player–GW samples and for backtesting across different environments.[^11][^10]

- **Granularity**:
  - Player–GW level for most FPL archives.
  - Match-level for xG/xA and advanced stats from external providers.
  - Team–GW level for aggregated team stats.

- **Pros/cons and quality**:
  - FPL stats: directly aligned with scoring system, but limited in some advanced metrics.
  - External advanced stats: richer football data but may require joining to FPL players and can have inconsistencies.
  - Community archives: convenient but may have gaps or need validation each season.

### 2.7 Proposed datasets

- **Minimal v1 dataset (proof of concept)**:
  - 3–5 seasons of FPL player–GW data, including minutes, goals, assists, clean sheets, cards, bonus, and prices.[^10][^11]
  - Basic team form: goals for/against by team and simple FDR from fixtures.
  - No external xG/xA in v1; rely on simple features like recent goals and minutes.
  - Use this to build a first expected points model and simple one-week optimiser.

- **Extended v2 dataset (advanced model)**:
  - 6–10 seasons of player–GW data with linked xG/xA and detailed advanced stats.[^12][^10]
  - Betting odds for upcoming matches to derive team and player-level expectations.
  - Rich meta data on injuries, suspensions, and rotation.
  - This supports more advanced probabilistic models and multi-week planning.

## 3. Target Definition and Feature Engineering

### 3.1 Prediction targets

Suitable targets include:

- **Expected FPL points per player per upcoming GW**: a regression target representing the mean points expectation, possibly decomposed into attacking, defensive and bonus components.[^16][^18]
- **Probability of starting / playing 60+ minutes**: a classification target capturing availability and rotation risk; this can be used to adjust point expectations.[^18]
- **Optional position-specific models**: separate models for GKP, DEF, MID, FWD reflecting different scoring profiles and relevant features.[^19]

These targets align with how optimisers consume inputs—given expected points and probabilities, they select players and lineups.

### 3.2 Feature sets

Rich feature families can include:

- **Player form and usage**:
  - Rolling averages over last 4–6 GWs of minutes, goals, assists, shots, xG/xA, key passes.[^12]
  - Moving averages of FPL points, with and without penalties and bonus.
  - Share of team xG/xA or involvement rate.

- **Team strength and context**:
  - Team-level rolling xG/xGA and goals for/against.[^12]
  - Attack vs defence splits.
  - Home/away dummy, fixture congestion (days since last match), travel.

- **Opponent-related features**:
  - Opponent defensive strength (goals/xGA conceded),
  - Opponent attacking strength for defensive clean sheet probabilities,
  - Fixture difficulty ratings and bookmaker odds.

- **Position-specific features**:
  - For defenders: CBIs, clearances, block counts, team defensive solidity.
  - For goalkeepers: saves, save percentage, shots on target faced.
  - For forwards/mids: xG, xA, shots in box, big chances.

- **Rotation and risk**:
  - Historical minutes pattern (start/sub/bench),
  - Participation in European or domestic cups,
  - Injury history, flagged status.

- **Interactions**:
  - Strong attack vs weak defence interaction indicators (e.g. top-attack team vs bottom-defence team).

### 3.3 Feature construction tactics

- **Lagged features and rolling aggregates**:
  - Create lagged stats (1, 2, 3 GWs ago) and rolling windows (last 4/6/8 GWs) using time-ordered data.
  - Ensure features only use information up to the prediction point to avoid look-ahead bias.

- **Promoted/relegated teams and limited history**:
  - For promoted teams, backfill using previous league stats (scaled) or initial priors until enough Premier League data exists.[^12]
  - For players with limited history, use positional and team averages, shrinking towards priors.

- **Handling injuries/suspensions**:
  - Use flags as categorical features.
  - Combine with minutes models to reduce expected minutes when risk is high.

### 3.4 Feature selection and importance

To reduce overfitting and redundant features, apply:

- Regularisation-based models (e.g. L1/L2) or tree-based models’ built-in importance.
- Permutation importance on validation sets to identify influential features.
- Simple correlation and mutual information checks to remove highly collinear or non-informative features.

## 4. Modelling Approaches and Training Setup

### 4.1 Modelling options

For expected points and minutes prediction:

- **Tree-based gradient boosting (XGBoost, LightGBM, CatBoost)**:
  - Strong baseline for tabular sports data, handle non-linearities and interactions, and generally perform well with modest feature engineering.[^18][^19]

- **Random forests**:
  - Robust, easy to train; good benchmark but often slightly less performant than tuned gradient boosting.

- **Simple neural networks**:
  - Can be used but often require more tuning and data; may not outperform gradient boosting on tabular FPL data.

- **Position-specific ensembles**:
  - Train separate models per position and ensemble or stack them; this reflects differing scoring drivers and may improve performance.[^18]

A pragmatic approach is to start with gradient boosting models and compare against simpler baselines.

### 4.2 Training and validation respecting time

To avoid leakage and reflect real-world forecasting:

- Use **time-based splits**:
  - Train on early seasons and validate on later seasons.
  - For intra-season tuning, use expanding windows (e.g. train up to GW N, validate on GW N+1) or rolling windows.

- Avoid using future fixtures or stats as features; ensure all features are computed only from past data.

- Evaluate model stability across seasons to detect shifts in dynamics.

### 4.3 Loss functions and metrics

- For **expected points regression**:
  - Use losses like mean squared error (MSE) or mean absolute error (MAE), monitored on validation sets.
  - Complement with practical ranking metrics (e.g. correlation between predicted and actual points for top-N players, or hit-rate of top picks).

- For **start/60+ minutes classification**:
  - Use logistic loss, evaluate with Brier score, log-loss, ROC-AUC, and calibration curves.

- For **practical decision-making metrics**:
  - Evaluate how well the model ranks players relative to actual top performers in each GW or across horizons.

### 4.4 Uncertainty and variance

Uncertainty matters for risk-aware planning and chip decisions:

- **Quantifying uncertainty**:
  - Use ensemble methods (bagging/boosting) and inspect prediction variance.
  - Fit models for both mean and variance (e.g. quantile regression) to derive prediction intervals.
  - Scenario sampling: generate multiple realisations of player points based on predicted mean/variance.

- **Using uncertainty**:
  - Incorporate risk-aversion terms into optimisation (e.g. penalise high-variance players).
  - Evaluate strategies under multiple simulated seasons to understand robustness.

## 5. Squad, Transfer, and Chip Optimisation

### 5.1 Single-gameweek optimisation

For a single GW, given expected points per player:

- **Decision variables**:
  - Ownership binary variables for each player.
  - Starting XI binary variables subject to formation constraints.
  - Captain and vice-captain binary variables.

- **Objective**:
  - Maximise expected GW points: sum of expected points for starters plus captain multiplier, plus bench points if Bench Boost is active.[^5][^3]

- **Constraints**:
  - Budget and squad composition.
  - Formation rules.
  - At most one captain and one vice-captain.

This can be formulated as a mixed-integer linear programme with linear objective and constraints.

### 5.2 Multi-gameweek planning

Extending over a horizon of H GWs introduces dynamic elements:[^4][^3]

- **Additional decision variables**:
  - Transfer in/out variables per player per GW.
  - Chip-use variables per GW.

- **Budget dynamics**:
  - Track budget each GW based on player prices and transfers.

- **Transfer limits and hits**:
  - Free transfers per GW; extra transfers incur −4 points each.

- **Objective**:
  - Maximise sum of expected points across H GWs minus transfer hit penalties.
  - Optionally discount future GWs or incorporate risk penalties.

### 5.3 Encoding as MILP

A MILP formulation may include:[^3][^4][^5]

- Binary decision variables for ownership, starting XI, transfers, captain, vice-captain, chip usage.
- Equality and inequality constraints enforcing:
  - Squad size and composition at each GW.
  - Transfer balance between GWs (players in/out).
  - Budget equilibrium and limits.
  - Formation restrictions.
  - At most one of each chip per allowed period.

Solvers (e.g. PuLP + CBC, OR-Tools, or commercial solvers) can handle these problems for realistic squad sizes and planning horizons, though horizon length and integer complexity must be managed.

### 5.4 Chip timing strategies

Chip timing can be handled in several ways:

- **Explicit decision variables**:
  - Include binary variables for “Wildcard in GW t”, “Bench Boost in GW t”, etc., with constraints that each can be 1 at most once, and that conflicting chips cannot be used in the same GW.[^8][^2]

- **Scenario search**:
  - Pre-select a set of candidate chip weeks (e.g. likely double GWs) and solve the optimisation for each candidate choice, then compare outcomes.

- **Heuristic pre-rules**:
  - For early versions, apply simple heuristic rules (e.g. bench boost in biggest double GW; wildcard before a double/fixture swing) and focus the optimiser on squad/transfer decisions only.

### 5.5 Practical simplifications

For an initial implementation:

- Assume fixed player prices over the horizon or simple projections.
- Ignore detailed price change dynamics.
- Do not optimise chip timing initially; use heuristics.
- Limit planning horizon to 3–4 GWs to keep MILP size manageable.

Subsequent versions can incrementally add price dynamics, longer horizons, and chip decisions.

## 6. Backtesting on Historical Seasons

### 6.1 Backtest framework design

The backtest simulates entire seasons using historical data:

- **Workflow per season**:
  - For each GW in chronological order:
    - Construct the information set that would be available at that time (past match stats, fixtures, injury flags, etc.).
    - Use trained models to generate expected points and starting probabilities for future GWs within the planning horizon.
    - Run the optimiser to choose squad, transfers, and captain/chips decisions.
    - Apply the realised historical outcomes to compute the actual FPL points earned.

- Repeat across several seasons to evaluate robustness.

### 6.2 Reconstructing “world as known”

To avoid lookahead bias:

- Use only statistics from matches that have actually occurred up to the current GW.
- Use fixture lists as known at the time (double GWs may be announced with lead time; this can be approximated using historical fixture announcement dates or simplified by assuming final fixtures were known from season start).
- Injury/rotation information can be approximated via flags in the historical FPL data or by simple rules (e.g. if a player did not play the previous GW due to injury, reduce expected minutes in the next one).[^13][^12]

### 6.3 Handling injuries and rotation

- Option 1: Use the minutes model to estimate probability of starting and adjust expected points accordingly.
- Option 2: Apply simple heuristics (e.g. drop expected minutes after recent injury, or for players with heavy fixtures in a short span).

### 6.4 Double and blank gameweeks

- Represent double GWs as multiple fixtures for a player within the same GW and sum expected points accordingly.[^1]
- Blanks are simply GWs where a team has no fixture; expected points become zero.

### 6.5 Evaluation criteria

Backtest outputs and metrics include:

- **Total season points** and a proxy for overall rank (e.g. comparison to historical official ranks of typical point thresholds, where available).
- **Comparison to baselines**:
  - No-transfer team built at start.
  - Simple heuristic strategies (e.g. always captain top predicted player, no optimiser; or use static “template team”).

- **Robustness across seasons**:
  - Distribution of total points across seasons.
  - Performance relative to top-percentile benchmark teams or simple bots.

### 6.6 Storing and visualising results

- Store full backtest trajectories: squad composition per GW, transfers, captaincy, chip usage, and GW/season points.
- Visualise:
  - Season points progression.
  - Chip usage vs subsequent score spikes.
  - Comparison to baselines and/or historical rank proxies.

## 7. Compute Requirements and Tech Stack

### 7.1 Recommended stack

A pragmatic stack for a single developer:

- **Language and core libraries**:
  - Python with `pandas`, `numpy`, `scikit-learn`, and gradient boosting libraries (XGBoost/LightGBM/CatBoost).

- **Optimisation**:
  - PuLP or OR-Tools for linear and mixed-integer programming, using open-source solvers (CBC, SCIP where available).[^5][^3]

- **Data collection**:
  - Requests/HTTP clients for FPL API calls, or wrappers such as `fpl-api` and community APIs.[^15][^13]

- **Experiment tracking**:
  - Simple setup using MLflow, Weights & Biases, or custom logging.

- **Storage**:
  - Local CSV/Parquet for datasets; optional PostgreSQL for structured storage.

### 7.2 Compute estimates

Approximate requirements will depend on data volume and model size, but for a few thousand player–GW samples per season over 8+ seasons:

- **Model training**:
  - Gradient boosting models can typically train in seconds to a few minutes on a modern laptop CPU for tabular datasets of this size.

- **Weekly optimisation**:
  - Single-GW MILP solves with 600–700 candidate players and 15 selection slots are usually solvable in seconds with good solvers.[^3][^5]
  - Multi-GW horizons introduce more variables but with careful pruning and horizon length (e.g. 3–5 GWs) they should still run in minutes or less in most cases.

- **Full-season backtests**:
  - Simulating dozens of seasons with per-GW optimisations may take hours; this is manageable with batch runs and caching.

### 7.3 Code structure and scaling

- **Structure**:
  - Separate modules for data ingestion, feature engineering, modelling, optimisation, and evaluation.
  - Configuration-driven design (YAML/JSON) for model/optimiser parameters and paths.

- **Caching**:
  - Precompute and store expected points predictions for all player–GW combinations to avoid repeated model inference in backtests.

- **Scaling**:
  - If computation becomes heavy, offload training or backtesting to cloud instances while keeping weekly usage light.

## 8. Existing FPL ML and Optimisation Projects

### 8.1 Representative open projects

A non-exhaustive sample of relevant projects includes:

- **FPL-AI (player points prediction)**:
  - Uses FPL API data and ML models to predict player points for each GW.[^16]
  - Strength: end-to-end prediction with accessible code; limitation: may not integrate advanced optimisation or multi-season backtesting.

- **FantasyPremierLeagueTeamRecommender**:
  - Builds ML and deep learning models to predict player points and recommends team line-ups using the official FPL API.[^17]
  - Strength: combination of predictive modelling and recommendation; limitation: unclear robustness and backtesting coverage.

- **OpenFPL forecasting framework (academic)**:
  - An open-source FPL forecasting method with models and inference code that can rival commercial tools.[^18]
  - Strength: rigorous methodology and reproducible code; potential gold-standard reference for forecasting accuracy.

- **Linear programming optimisers**:
  - Repos such as FPL optimisation projects that use PuLP/LP to build optimal teams and transfer plans from expected points inputs.[^5][^3]
  - Strength: well-structured MILP formulations; limitation: often treat expected points as exogenous rather than co-designed with the FPL context.

- **Julia-based optimisation (JFPL-Optimization)**:
  - Julia implementation of an FPL optimisation tool that uses expected points projections (e.g. from FPL Review) and solves the team selection problem.[^21]
  - Strength: high-performance optimisation; limitation: narrower focus on optimisation rather than full ML pipeline.

### 8.2 Lessons and design patterns

From these projects:

- Using the **official FPL API and archives** as primary data sources is standard and effective.[^17][^11][^16][^13]
- Separating **prediction and optimisation** (modular design) makes it easy to swap in improved projection models without changing the optimiser.[^3][^5]
- Forecasting frameworks like OpenFPL demonstrate the value of careful **time-aware validation and model calibration**, and can be used as reference standards for forecasting quality.[^18]
- Existing optimisers show feasible MILP formulations and highlight trade-offs between planning horizon size and solver complexity.[^21][^5][^3]

### 8.3 Gold-standard references

- **OpenFPL** is a strong candidate for a gold standard in forecasting methodology, due to its open-source models, robust validation, and head-to-head performance vs established benchmarks.[^18]
- Well-engineered optimisation repos like FPL-Optimization or similar projects provide reference formulations for LP/MILP-based team selection.[^5][^3]

## 9. Additional Dimensions to Research

Further aspects worth exploring include:

- **Bookmaker odds and market-implied expectations**:
  - Odds can provide strong priors for goal and clean sheet probabilities and can improve prediction accuracy beyond historical stats alone.[^12]

- **Upside, variance and risk preferences**:
  - Especially relevant for mini-league chasing vs OR protection—optimisers can incorporate variance proxies or downside risk into objectives.

- **Scenario-based planning for double/blank GWs**:
  - Generate scenarios around fixture rescheduling and player availability to stress-test strategies.

- **News, injuries and rotation**:
  - Integrate structured injury feeds or scraped news sentiment to adjust minutes expectations.

- **Explainability and diagnostics**:
  - Use feature importance, SHAP values and sanity checks to verify that models are learning sensible relationships and to debug anomalies.

Prioritisation:

- "Should be in v1": basic bookmaker odds integration, simple risk-aware evaluation, and explainability diagnostics.
- "Nice-to-have later": detailed scenario-based planning, news/sentiment integration, and advanced risk-optimised strategies.

## 10. Phased Implementation Roadmap

### 10.1 High-level system overview

The target system consists of:

- A data pipeline ingesting FPL (and optional external) data into clean tables.
- ML models predicting expected points and start probabilities per player per GW.
- An optimisation engine (MILP) that takes forecasts and game rules to select squads, lineups, transfers, and chip usage.
- A backtesting framework simulating seasons to evaluate strategies.

### 10.2 Detailed research plan (tasks)

**Phase A – Understand rules and formalise optimisation**

- Extract and document all relevant FPL rules from official help and update pages.[^2][^7][^8][^1]
- Formalise single- and multi-GW optimisation models (variables, constraints, objectives), including chip logic.

**Phase B – Data audit and ingestion**

- Survey available FPL archives and Kaggle datasets, select a primary historical dataset.[^11][^10]
- Build an ingestion script for the official FPL API for current season data.[^13]
- Design a unified schema for player–GW, team–GW, and fixtures.

**Phase C – Feature engineering and target specification**

- Decide on primary targets (expected points and start probability).
- Implement rolling and lagged features, test for leakage.
- Handle promoted teams, new players, and missing data.

**Phase D – Modelling and validation**

- Implement baseline models (e.g. gradient boosting) and simple baselines (moving average form models).[^19][^18]
- Design time-based cross-validation across seasons.
- Evaluate models on both traditional metrics and decision-relevant ranking metrics.

**Phase E – Single-GW optimiser**

- Implement a single-GW optimiser that, given expected points, returns an optimal squad and XI with captaincy.[^3][^5]
- Validate against known simple scenarios and manual checks.

**Phase F – Multi-GW planning and transfers**

- Extend optimiser to include transfers across a horizon (e.g. 3–5 GWs), including hits.
- Implement basic chip heuristics and later integrate explicit chip decision variables.

**Phase G – Backtesting**

- Build backtest engine to simulate multiple seasons.
- Evaluate the system versus baselines across seasons.

**Phase H – Refinement and extras**

- Integrate bookmaker odds and richer advanced stats.
- Introduce uncertainty-aware optimisation and scenario analysis.
- Improve explainability and diagnostics.

### 10.3 Roadmap with versions

- **v0.1 – Minimal predictor and one-week optimiser (2–4 weeks)**
  - Ingest 3–5 seasons of FPL data (from archives or Kaggle).[^10][^11]
  - Implement simple features (recent points and minutes).
  - Train a basic gradient boosting model to predict expected points next GW.
  - Implement a single-GW MILP optimiser for team selection with fixed prices.

- **v0.2 – Robust forecasting and early backtesting (4–6 weeks)**
  - Add richer features (team strength, opponent, home/away).
  - Improve validation (multi-season time-based cross-validation).[^18]
  - Build backtest engine for 2–3 seasons using single-GW optimisation.
  - Implement a simple transfer heuristic (e.g. top expected-gain transfers) outside the optimiser.

- **v0.3 – Integrated multi-GW optimisation (4–6 weeks)**
  - Add transfer decision variables and hits into MILP for 3–5 GW horizons.[^4][^3]
  - Integrate basic chip heuristics (bench boost in biggest double, wildcard before fixture swing).
  - Backtest across multiple seasons with multi-GW planning.

- **v1.0 – Full-stack system with chip optimisation (6–10 weeks)**
  - Add explicit chip timing decision variables into optimisation.
  - Integrate bookmaker odds and richer advanced stats.
  - Implement risk-aware variants (variance penalties or downside constraints).
  - Run comprehensive multi-season backtests and benchmark vs public projection + optimisation tools.[^20][^18]

## 11. Additional Important Considerations

Beyond the requested brief, several aspects merit attention:

- **Data governance and reproducibility**:
  - Version datasets and modelling code to ensure that backtests and experiments are reproducible.

- **Ethics and fair use**:
  - Respect terms of service for data providers and refrain from scraping where prohibited.

- **User interface and usability**:
  - Even a simple CLI or notebook-based interface for scenario exploration will significantly improve practical usability for human managers.

These considerations improve the robustness and real-world value of the FPL ML + optimisation project.

---

## References

1. [Fantasy Football Game Rules & Help](https://fantasy.premierleague.com/en/help/rules) - Scoring. How are points scored? During the season, your fantasy football players will be allocated p...

2. [Fantasy Premier League 2026/27: All rule changes and ...](https://www.flashscore.com/news/soccer-premier-league-fantasy-premier-league-2026-27-all-rule-changes-and-new-features/KGmq7ts1/) - Projected bonus points will be added after 20 minutes of each match and can change as the game progr...

3. [dbozbay/FPL-Optimization: A Fantasy Premier League ...](https://github.com/dbozbay/FPL-Optimization) - A Fantasy Premier League optimiser that leverages linear programming techniques to maximise team poi...

4. [Data-Driven Team Selection in Fantasy Premier League ...](https://arxiv.org/html/2505.02170v1) - This paper proposes novel deterministic and robust integer programming models that select the optima...

5. [Fantasy Premier League: Team Selector and Optimizer](https://github.com/spinalwiz/fpl-optimiser) - FPL Optimizer selects the best possible fantasy football team by using an expected points model to g...

6. [2026/27 Game Updates | Fantasy Premier League](https://fantasy.premierleague.com/en/help/new) - A manager's overall points will also include bonus points as they come into play. Initially, bonus p...

7. [FPL basics explained: Scoring points](https://www.premierleague.com/en/news/2174909/fpl-basics-explained-scoring-points) - Any defender who reaches a combined total of 10 clearances, blocks, interceptions and tackles (CBIT)...

8. [All you need to know about changes to FPL for 2026/27](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627) - This season, projected bonus points will be added to your players' scores after 20 minutes of each m...

9. [FPL 2026/27 Changes: Live Ranks, Bonus Points & Chips](https://www.fantasyfootballfix.com/blog-index/fpl-2026-27-new-rules/) - Projected bonus is added to a player's score after 20 minutes of each match. Projected bonus points ...

10. [Fantasy Premier League Player Data (2016-2024)](https://www.kaggle.com/datasets/reevebarreto/fantasy-premier-league-player-data-2016-2024) - This dataset provides an archive of Fantasy Premier League (FPL) player performance data for eight s...

11. [vaastav/Fantasy-Premier-League: Creates a .csv file of all ...](https://github.com/vaastav/Fantasy-Premier-League) - Machine Learning Model by pratz · xA vs xG for Attacking Midfielders/Forwards ... How to Win at Fant...

12. [FPL Analysis: What five seasons' of data modelling have ...](https://mathematicallysafe.wordpress.com/2019/07/01/fpl-analysis-what-five-seasons-of-data-modelling-have-revealed-about-predictive-analysis-fixture-impact-and-optimal-team-structure-in-fantasy-premier-league/) - There are 1,554 unique player records in the data set covering five seasons, sourced from OPTA via t...

13. [FPL APIs Explained - Oliver Looney](https://www.oliverlooney.com/blogs/FPL-APIs-Explained) - Documentation for how to use FPL (premier league fantasy football) APIs and what endpoints are avail...

14. [FPL API Documentation](https://fpl-api-tau.vercel.app/) - Welcome to the FPL API Documentation (v1). This API provides data from the Fantasy Premier League ga...

15. [jeppe-smith/fpl-api - GitHub](https://github.com/jeppe-smith/fpl-api) - FPL API Use the public endpoints from the official api from https://fantasy.premierleague.com. by us...

16. [GitHub - saheedniyi02/fpl-ai: A machine learning system ...](https://github.com/saheedniyi02/fpl-ai) - A machine learning system that predicts fpl points of players. FPL has an API that gives access to f...

17. [Team Recommendation System for Fantasy Premier ...](https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender) - ... Fantasy API. Use different Machine Learning and Deep Learning algorithms to build a model that c...

18. [OpenFPL: An open-source forecasting method rivaling ...](https://arxiv.org/html/2508.09992v1) - OpenFPL, an open-source Fantasy Premier League forecasting method. Models and inference code are fre...

19. [Optimising FPL with Julia and JuMP](https://dm13450.github.io/2022/08/05/FPL-Optimising.html) - This is an expected points model and will take into account the player's position, form, and overall...

20. [FPL Review | Projections, Planner & Solver](https://fplreview.com/) - Build your Fantasy Premier League team with a leading predicted points model, transfer planner and s...

21. [Fantasy Premier League (FPL) Optimization](https://discourse.julialang.org/t/fantasy-premier-league-fpl-optimization/119080) - JFPL-Optimization is designed to help FPL managers identify the optimal team based on expected point...

