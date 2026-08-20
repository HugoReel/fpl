# Team strength: Dixon-Coles against the market

**The market wins, so compose consumes market team expectations and keeps Dixon-Coles as the fallback.** Over 2021-22, 2022-23, 2023-24, 2024-25, de-vigged closing odds average 0.9417 outcome log loss against Dixon-Coles at 0.9824, a gap of 4.3%. Dixon-Coles is not discarded: it covers fixtures with no odds, it is the sanity cross-check, and it supplies the score matrix machinery the attack model needs.

Beating the bookmaker was never the target. A Dixon-Coles fitted on public results cannot see team news, transfers or money, and the closing price has absorbed all of it. Getting within a few percent is the realistic bar, and the gap here is 4.3%. What matters is that the question is now settled on a number rather than on preference.

Both are compared against `v0_trailing`, the team model implicit in the shrunk trailing rates compose uses today. v0 never states a team goal expectation, but it implies one, and reconstructing it is what makes this a three way comparison rather than a two way one.

## Match outcome log loss

Lower is better. The decision uses the pre-freeze seasons only; 2025-26 is shown for confirmation and took no part in choosing.

| Season | market | dixon_coles | v0_trailing | best |
|---|---:|---:|---:|---|
| 2021-22 | 0.9365 | 1.0009 | 1.0074 | market |
| 2022-23 | 0.9614 | 1.0219 | 1.0378 | market |
| 2023-24 | 0.9004 | 0.9239 | 0.9987 | market |
| 2024-25 | 0.9684 | 0.9828 | 1.0212 | market |
| 2025-26 (frozen) | 1.0144 | 1.0358 | 1.0562 | market |

## Clean sheet Brier score

Clean sheets are what this actually feeds into compose, so they get their own column. One observation per team per match.

| Season | market | dixon_coles | v0_trailing |
|---|---:|---:|---:|
| 2021-22 | 0.1803 | 0.1898 | 0.1921 |
| 2022-23 | 0.1863 | 0.1959 | 0.1975 |
| 2023-24 | 0.1537 | 0.1610 | 0.1579 |
| 2024-25 | 0.1702 | 0.1726 | 0.1742 |
| 2025-26 | 0.1774 | 0.1820 | 0.1876 |

## Dixon-Coles clean sheet calibration, 2025-26

Tracks to within 0.073 across bins holding at least 30 team fixtures. Probabilities come from the corrected score grid rather than a Poisson zero, which matters because the Dixon-Coles correction is largest at exactly the low scores a clean sheet depends on.

| Bin | Team fixtures | Mean predicted | Realised |
|---|---:|---:|---:|
| 0.0-0.1 | 24 | 0.075 | 0.208 |
| 0.1-0.2 | 197 | 0.155 | 0.137 |
| 0.2-0.3 | 273 | 0.246 | 0.256 |
| 0.3-0.4 | 198 | 0.341 | 0.293 |
| 0.4-0.5 | 55 | 0.437 | 0.509 |
| 0.5-0.6 | 12 | 0.529 | 0.500 |
| 0.6-0.7 | 0 | | |
| 0.7-0.8 | 0 | | |
| 0.8-0.9 | 1 | 0.827 | 0.000 |
| 0.9-1.0 | 0 | | |

## What this changes

compose gains a `team_source` setting. It reads `market` where that source covers the fixture and falls back to `dixon_coles` where it does not, which in practice means fixtures with no priced market. The trailing rates stay available as a third fallback and as the pre-swap comparison for the harness run.

The wider point is architectural. The market holds team strength and cannot see who starts or who takes the shots. This project holds a calibrated minutes model and is not going to out-predict closing prices on match outcomes. Feeding market team expectations into this project's minutes and allocation uses each where it is strong, and it is roughly what the commercial services are doing.

## The fallback is load bearing right now, not in theory

football-data.co.uk publishes closing prices for matches that have already been played. That is perfect for backtesting and useless for a Saturday deadline. Running compose against the live 2026-27 gameweek 1 resolves to `{market: 138707, dixon_coles: 599}`: every historical row takes the market, and every upcoming fixture falls through to Dixon-Coles, because no source in this repository prices a match that has not happened yet.

Two consequences follow, and both are operational rather than modelling:

1. **Dixon-Coles is currently carrying the entire live season.** The accepted configuration is market-first, but for prediction rather than replay the market half is not connected. What the harness accepted is the architecture; what runs on a Saturday is still the fallback.
2. **A live odds feed moves from optional to load bearing.** The Odds API free tier is the obvious candidate and its request budget and fixture coverage want checking before anything depends on it for a deadline. The chain degrades in the right direction when a pull fails, which is the whole reason it is a chain, and the no-coverage path is covered by tests rather than assumed.

