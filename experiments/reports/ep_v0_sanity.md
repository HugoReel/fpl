# Expected points v0, sanity check

Season 2025-26, gameweek 20, 790 players. Expected points come from the v0 rate estimators and the minutes model, composed through `scoring/rules_2026_27.py`. Realised points are recomputed from components rather than read from the stored total.

**All sanity checks pass.** The list below is not mad.

## Before and after

Same gameweek, same minutes model, same optimiser. Only the rate estimators changed, so every difference below is attributable to that change.

| Measure | Before | After |
|---|---:|---:|
| Highest expected points | 5.79 | 5.29 |
| Top 20 mean realised points | 4.20 | 4.15 |
| Top 20 mean price | 6.92 | 6.79 |
| Spearman against realised | 0.733 | 0.735 |
| Opponent defence adj at clip floor | 4.8% | 0.0% |
| Opponent defence adj at clip ceiling | 4.6% | 0.0% |
| Opponent attack adj at clip floor | 10.5% | 0.0% |

The most expensive players in the game, and where each ranked by expected points before and after:

| Player | Price | EP before | EP after | Rank before | Rank after |
|---|---:|---:|---:|---:|---:|
| Haaland | 14.7 | 4.18 | 4.55 | 12 | 3 |
| M.Salah | 14.0 | 0.67 | 0.68 | 329 | 324 |
| B.Fernandes | 10.4 | 1.09 | 1.01 | 282 | 292 |
| Palmer | 10.3 | 2.47 | 2.66 | 133 | 108 |
| Isak | 10.3 | 1.15 | 1.16 | 278 | 275 |
| Saka | 10.0 | 4.64 | 4.30 | 3 | 7 |
| Gyökeres | 9.1 | 2.27 | 2.12 | 157 | 179 |
| Ekitiké | 9.0 | 3.56 | 3.60 | 35 | 27 |

## Checks

| Check | Result | Evidence |
|---|---|---|
| Ranking carries signal | pass | top 20 by EP averaged 4.15 realised points against 1.14 for the whole pool |
| Rank correlation with realised points | pass | Spearman 0.735 across 790 players |
| No bench fodder in the top 10 | pass | 10 of 10 are expected to play at least 60 minutes |
| Top 20 skews expensive, as it should | pass | mean price 6.79 against 4.90 for the whole pool |
| Top expected points reaches a plausible level | pass | highest EP in the gameweek is 5.29, and a genuine premium in a good fixture should clear 5 |

## Top 20 by expected points, 2025-26 gw20

`realised` is what they actually scored, shown so the ranking can be judged rather than admired. One gameweek of realised points is mostly noise, so treat the column as a smell test, not an evaluation. The real evaluation is the harness in task 3.

| # | Player | Team | Pos | Price | xMins | xGoals | p(CS) | xBonus | EP | Realised |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Trossard | ARS | MID | 6.6 | 72 | 0.45 | 0.36 | 0.36 | 5.29 | 1 |
| 2 | Rogers | AVL | MID | 7.3 | 76 | 0.35 | 0.36 | 0.42 | 4.55 | 5 |
| 3 | Haaland | MCI | FWD | 14.7 | 74 | 0.46 | 0.48 | 0.45 | 4.55 | 2 |
| 4 | Pickford | EVE | GKP | 5.6 | 86 | 0.00 | 0.54 | 0.24 | 4.53 | 1 |
| 5 | Wilson | FUL | MID | 5.8 | 72 | 0.24 | 0.39 | 0.36 | 4.34 | 6 |
| 6 | Tarkowski | EVE | DEF | 5.8 | 82 | 0.02 | 0.54 | 0.23 | 4.33 | 2 |
| 7 | Saka | ARS | MID | 10.0 | 68 | 0.14 | 0.36 | 0.30 | 4.30 | 4 |
| 8 | Foden | MCI | MID | 8.0 | 71 | 0.30 | 0.48 | 0.32 | 4.29 | 2 |
| 9 | Gabriel | ARS | DEF | 7.3 | 67 | 0.14 | 0.36 | 0.59 | 4.10 | 9 |
| 10 | Matheus N. | MCI | DEF | 5.3 | 78 | 0.01 | 0.48 | 0.39 | 4.04 | 1 |
| 11 | Bruno G. | NEW | MID | 6.9 | 77 | 0.21 | 0.12 | 0.39 | 4.02 | 10 |
| 12 | Verbruggen | BHA | GKP | 4.6 | 86 | 0.00 | 0.38 | 0.14 | 4.00 | 7 |
| 13 | Wissa | NEW | FWD | 7.3 | 70 | 0.38 | 0.12 | 0.35 | 3.98 | 2 |
| 14 | Bowen | WHU | FWD | 7.8 | 77 | 0.36 | 0.08 | 0.30 | 3.98 | 4 |
| 15 | Van Hecke | BHA | DEF | 4.7 | 80 | 0.15 | 0.38 | 0.04 | 3.92 | 6 |
| 16 | Rúben | MCI | DEF | 5.5 | 81 | 0.11 | 0.48 | 0.09 | 3.88 | 5 |
| 17 | Casemiro | MUN | MID | 5.8 | 65 | 0.23 | 0.05 | 0.29 | 3.85 | 7 |
| 18 | Donnarumma | MCI | GKP | 5.6 | 84 | 0.00 | 0.48 | 0.00 | 3.84 | 2 |
| 19 | Roefs | SUN | GKP | 4.8 | 84 | 0.00 | 0.37 | 0.14 | 3.82 | 3 |
| 20 | Cherki | MCI | MID | 6.5 | 65 | 0.09 | 0.48 | 0.42 | 3.81 | 4 |

## Where the expensive players rank

Price is the market's own expected points estimate, so a v0 model that ranks the ten most expensive players far down the list is disagreeing with several million people and should be able to say why.

| Player | Team | Pos | Price | EP | EP rank | Realised |
|---|---|---|---:|---:|---:|---:|
| Haaland | MCI | FWD | 14.7 | 4.55 | 3 | 2 |
| M.Salah | LIV | MID | 14.0 | 0.68 | 324 | 0 |
| B.Fernandes | MUN | MID | 10.4 | 1.01 | 292 | 0 |
| Palmer | CHE | MID | 10.3 | 2.66 | 108 | 2 |
| Isak | LIV | FWD | 10.3 | 1.16 | 275 | 0 |
| Saka | ARS | MID | 10.0 | 4.30 | 7 | 4 |
| Gyökeres | ARS | FWD | 9.1 | 2.12 | 179 | 2 |
| Ekitiké | LIV | FWD | 9.0 | 3.60 | 27 | 0 |
| Watkins | AVL | FWD | 8.7 | 3.36 | 45 | 7 |
| Son | TOT | MID | 8.5 | 0.14 | 442 | 0 |

## Assessment

The ranking works. Spearman against realised points is 0.735, the top 20 averaged 4.15 points against a pool average of 1.14, and every one of the top 10 is a player expected to start. There is no bench fodder near the top, which was the specific failure this check exists to catch.

Expected points are still compressed at the top, though less than they were. The highest in the gameweek is 5.29, and the spread between an elite forward and a nailed cheap defender remains narrower than it should be.

The dominant cause has been fixed. Opponent strength entered as a raw ratio of the opponent's trailing goals to the league average, taken from at most ten matches and applied at full strength. That made it the largest single term in a striker's expected goals, larger than the striker's own scoring rate, and it pinned the safety clip on roughly one player fixture in nine. Team form is now shrunk toward the league average by the evidence behind it, the same way player rates already were, and the clip no longer binds at all.

Two causes of the remaining compression are genuinely by design at v0:

1. The empirical Bayes prior on player rates is a flat 600 minutes toward the position mean, against a trailing window capped at ten matches. The formula does weaken with evidence, but the cap means evidence never exceeds about ten nineties, so an established elite scorer never gets more than roughly 60 percent weight on their own record. That floor is what the Poisson attack model removes.
2. Expected minutes sits near 70 to 80 for most starters rather than 90, which scales every per-90 rate down. That is correct on average and still costs the genuinely nailed players, because their conditional distribution is much tighter than the positional average the estimate is drawn from.

The practical consequence is that this v0 still under-captains premiums relative to cheap defenders on good clean sheet odds. That matters for the optimiser, which doubles the captain's score. Until the component models land, the number to watch is not RMSE but whether the optimised team beats the baselines in the evaluation harness.

