# Expected points v0, sanity check

Season 2025-26, gameweek 20, 790 players. Expected points come from the v0 rate estimators and the minutes model, composed through `scoring/rules_2026_27.py`. Realised points are recomputed from components rather than read from the stored total.

**All sanity checks pass.** The list below is not mad.

## Checks

| Check | Result | Evidence |
|---|---|---|
| Ranking carries signal | pass | top 20 by EP averaged 4.20 realised points against 1.14 for the whole pool |
| Rank correlation with realised points | pass | Spearman 0.733 across 790 players |
| No bench fodder in the top 10 | pass | 10 of 10 are expected to play at least 60 minutes |
| Top 20 skews expensive, as it should | pass | mean price 6.92 against 4.90 for the whole pool |
| Top expected points reaches a plausible level | pass | highest EP in the gameweek is 5.79, and a genuine premium in a good fixture should clear 5 |

## Top 20 by expected points, 2025-26 gw20

`realised` is what they actually scored, shown so the ranking can be judged rather than admired. One gameweek of realised points is mostly noise, so treat the column as a smell test, not an evaluation. The real evaluation is the harness in task 3.

| # | Player | Team | Pos | Price | xMins | xGoals | p(CS) | xBonus | EP | Realised |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Trossard | ARS | MID | 6.6 | 72 | 0.52 | 0.37 | 0.36 | 5.79 | 1 |
| 2 | Van Hecke | BHA | DEF | 4.7 | 80 | 0.18 | 0.50 | 0.04 | 4.71 | 6 |
| 3 | Saka | ARS | MID | 10.0 | 68 | 0.16 | 0.37 | 0.30 | 4.64 | 4 |
| 4 | Pickford | EVE | GKP | 5.6 | 86 | 0.00 | 0.54 | 0.24 | 4.62 | 1 |
| 5 | Verbruggen | BHA | GKP | 4.6 | 86 | 0.00 | 0.50 | 0.14 | 4.62 | 7 |
| 6 | Rogers | AVL | MID | 7.3 | 76 | 0.34 | 0.37 | 0.42 | 4.51 | 5 |
| 7 | Gabriel | ARS | DEF | 7.3 | 67 | 0.16 | 0.37 | 0.59 | 4.42 | 9 |
| 8 | Tarkowski | EVE | DEF | 5.8 | 82 | 0.01 | 0.54 | 0.23 | 4.40 | 2 |
| 9 | Bowen | WHU | FWD | 7.8 | 77 | 0.43 | 0.08 | 0.30 | 4.37 | 4 |
| 10 | Wilson | FUL | MID | 5.8 | 72 | 0.23 | 0.39 | 0.36 | 4.20 | 6 |
| 11 | Roefs | SUN | GKP | 4.8 | 84 | 0.00 | 0.43 | 0.14 | 4.19 | 3 |
| 12 | Haaland | MCI | FWD | 14.7 | 74 | 0.39 | 0.45 | 0.45 | 4.18 | 2 |
| 13 | Casemiro | MUN | MID | 5.8 | 65 | 0.26 | 0.04 | 0.29 | 4.07 | 7 |
| 14 | Foden | MCI | MID | 8.0 | 71 | 0.25 | 0.45 | 0.32 | 3.99 | 2 |
| 15 | Matheus N. | MCI | DEF | 5.3 | 78 | 0.01 | 0.45 | 0.39 | 3.94 | 1 |
| 16 | Bruno G. | NEW | MID | 6.9 | 77 | 0.19 | 0.14 | 0.39 | 3.94 | 10 |
| 17 | Wissa | NEW | FWD | 7.3 | 70 | 0.35 | 0.14 | 0.35 | 3.85 | 2 |
| 18 | Cunha | MUN | MID | 8.1 | 76 | 0.28 | 0.04 | 0.17 | 3.84 | 9 |
| 19 | Donnarumma | MCI | GKP | 5.6 | 84 | 0.00 | 0.45 | 0.00 | 3.83 | 2 |
| 20 | Welbeck | BHA | FWD | 6.3 | 59 | 0.52 | 0.50 | 0.16 | 3.82 | 1 |

## Where the expensive players rank

Price is the market's own expected points estimate, so a v0 model that ranks the ten most expensive players far down the list is disagreeing with several million people and should be able to say why.

| Player | Team | Pos | Price | EP | EP rank | Realised |
|---|---|---|---:|---:|---:|---:|
| Haaland | MCI | FWD | 14.7 | 4.18 | 12 | 2 |
| M.Salah | LIV | MID | 14.0 | 0.67 | 329 | 0 |
| B.Fernandes | MUN | MID | 10.4 | 1.09 | 282 | 0 |
| Palmer | CHE | MID | 10.3 | 2.47 | 133 | 2 |
| Isak | LIV | FWD | 10.3 | 1.15 | 278 | 0 |
| Saka | ARS | MID | 10.0 | 4.64 | 3 | 4 |
| Gyökeres | ARS | FWD | 9.1 | 2.27 | 157 | 2 |
| Ekitiké | LIV | FWD | 9.0 | 3.56 | 35 | 0 |
| Watkins | AVL | FWD | 8.7 | 3.32 | 53 | 7 |
| Son | TOT | MID | 8.5 | 0.13 | 442 | 0 |

## Assessment

The ranking works. Spearman against realised points is 0.733, the top 20 averaged 4.20 points against a pool average of 1.14, and every one of the top 10 is a player expected to start. There is no bench fodder near the top, which was the specific failure this check exists to catch.

The clear weakness is compression at the top. The highest expected points in the gameweek is 5.79, and the spread between an elite forward and a nailed cheap defender is far narrower than it should be. Two causes, both known and both by design at v0:

1. The empirical Bayes prior is a flat 600 minutes toward the position mean, which is far too aggressive for a striker with two seasons of elite scoring behind them. Shrinkage should weaken as evidence accumulates, and right now it does not.
2. Expected minutes sits near 70 to 80 for most starters rather than 90, which scales every per-90 rate down. That is correct on average and still costs the genuinely nailed players, because their conditional distribution is much tighter than the positional average the estimate is drawn from.

The practical consequence is that this v0 will under-captain premiums and over-rank cheap defenders on good clean sheet odds. That matters for the optimiser in task 2, which doubles the captain's score and will happily captain a 4.5 million defender. It is the thing the Poisson attack model in a later phase exists to fix, and until then the number to watch is not RMSE but whether the optimised team beats the baselines in the task 3 harness.

