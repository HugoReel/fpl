# Odds ingest and the odds-only baseline

**Validation passes.** Every fixture in every season is priced, the de-vigged market beats a base rate on match outcomes in all 5 seasons, and clean sheet probabilities track realised rates to within 0.045 across every bin holding at least 50 team fixtures.

The sparse bins are noisier and should not be read as miscalibration. The worst is 0.0-0.1, holding 32 team fixtures, where 0.078 predicted against 0.000 realised amounts to about 2 expected clean sheets that did not happen.

Source is football-data.co.uk closing average prices, de-vigged with Shin's method. Shin models the bookmaker margin as protection against insider trading and removes proportionally more from longshots, where basic normalisation spreads it evenly and is known to overprice favourites. Longshot distortion matters here because it is worst at the heavy-win end, which is exactly where clean sheet probability lives.

## Join coverage

Matched on both club names plus kickoff date, because a postponed fixture keeps its teams but moves its date. Club name differences are mapped in `mapping/club_names.csv`, which needs three entries: Man United, Sheffield United and Tottenham.

| Season | Fixtures | Priced | Unmatched |
|---|---:|---:|---|
| 2021-22 | 380 | 100.0% | none |
| 2022-23 | 380 | 100.0% | none |
| 2023-24 | 380 | 100.0% | none |
| 2024-25 | 380 | 100.0% | none |
| 2025-26 | 380 | 100.0% | none |

## Floor check: do the odds predict match outcomes

Three way log loss of the de-vigged prices against realised results, next to a constant base rate. A market that failed to beat a base rate would mean a broken join or an inverted probability, not a bad market.

| Season | Matches | Market log loss | Base rate log loss | Improvement | Mean margin |
|---|---:|---:|---:|---:|---:|
| 2021-22 | 380 | 0.9360 | 1.0686 | +12.4% | 0.039 |
| 2022-23 | 380 | 0.9619 | 1.0469 | +8.1% | 0.039 |
| 2023-24 | 380 | 0.8991 | 1.0531 | +14.6% | 0.040 |
| 2024-25 | 380 | 0.9668 | 1.0776 | +10.3% | 0.042 |
| 2025-26 | 380 | 1.0122 | 1.0793 | +6.2% | 0.057 |

## Clean sheet calibration, 2025-26

One row per team per fixture, so each match contributes two clean sheet opportunities. Predicted comes from a Poisson zero on the opponent's goal expectation.

| Bin | Team fixtures | Mean predicted | Realised |
|---|---:|---:|---:|
| 0.0-0.1 | 32 | 0.078 | 0.000 |
| 0.1-0.2 | 200 | 0.153 | 0.145 |
| 0.2-0.3 | 261 | 0.251 | 0.268 |
| 0.3-0.4 | 183 | 0.346 | 0.301 |
| 0.4-0.5 | 73 | 0.441 | 0.466 |
| 0.5-0.6 | 8 | 0.540 | 0.500 |
| 0.6-0.7 | 3 | 0.634 | 0.667 |
| 0.7-0.8 | 0 | | |
| 0.8-0.9 | 0 | | |
| 0.9-1.0 | 0 | | |

## Known weaknesses

- **No scorer odds.** football-data.co.uk carries match markets only, so a player's share of his team's goals is allocated by position and price rank rather than by anything about the player. Inside a price bracket, a poacher and a midfielder who never shoots are indistinguishable. This is the baseline's weakest point and it is where a trained attack model should beat it.
- **Independent Poisson.** Goal expectations are recovered by fitting an independent Poisson pair to the 1X2 and over/under markets. Goals are correlated, particularly at low scores, which is exactly what Dixon-Coles corrects. The refit reproduces the market's own 1X2 probabilities to about one percentage point, so it is faithful to the prices even though the generative model is wrong.
- **One leaked constant.** The defensive contribution base rate is measured on 2025-26 because that is the only season in which the statistic was recorded. It is a single per position number, not anything player specific.

