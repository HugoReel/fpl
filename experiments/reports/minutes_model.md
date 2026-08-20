# Minutes model, v0

Model version `v0`, generated 2026-08-20 19:57 UTC.

**The model beats the started-last-match baseline on p_start log loss in every held out season.**

The margin is not uniform, and the weakest season is the informative one: 2023-24 improves by only 0.2%, against 25.4% at best. 2023-24 is the season with the least training history behind it, and the history it does have is the least trustworthy, since 2021-22 carries no exact start flags at all. The edge grows as seasons accumulate, which is the expected shape but is worth restating: on two seasons of training data this model is barely worth its complexity over a lookup table.

The baseline is a genuine one. It is P(start | started last match, position) estimated on the training seasons, which is strictly stronger than the hard 0 or 1 rule the spec names, so beating it is a stronger claim than beating the literal version.

Walk forward only: each season is scored by a model trained purely on earlier seasons, with an isotonic calibrator fitted on the tail of those seasons that the booster never saw.

## Head metrics by held out season

Baseline for p_start is P(start | started last match, position) estimated on the training seasons. For the other heads it is the training base rate. Improvement is the reduction in log loss.

| Season | Head | Rows | Base rate | Model log loss | Baseline log loss | Improvement | Model Brier | Baseline Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2023-24 | p_start | 29,725 | 0.281 | 0.3373 | 0.3379 | +0.2% | 0.0966 | 0.0975 |
| 2023-24 | p_60_given_start | 8,360 | 0.935 | 0.2350 | 0.2413 | +2.6% | 0.0603 | 0.0610 |
| 2023-24 | p_sub | 21,365 | 0.142 | 0.2855 | 0.4229 | +32.5% | 0.0882 | 0.1261 |
| 2024-25 | p_start | 27,283 | 0.306 | 0.2953 | 0.3713 | +20.5% | 0.0922 | 0.1096 |
| 2024-25 | p_60_given_start | 8,360 | 0.931 | 0.2460 | 0.2518 | +2.3% | 0.0638 | 0.0645 |
| 2024-25 | p_sub | 18,923 | 0.169 | 0.2943 | 0.4558 | +35.4% | 0.0914 | 0.1410 |
| 2025-26 | p_start | 29,747 | 0.281 | 0.2524 | 0.3382 | +25.4% | 0.0781 | 0.0978 |
| 2025-26 | p_60_given_start | 8,360 | 0.929 | 0.2464 | 0.2553 | +3.5% | 0.0643 | 0.0656 |
| 2025-26 | p_sub | 21,387 | 0.146 | 0.2582 | 0.4208 | +38.6% | 0.0804 | 0.1262 |

## Expected minutes

Error against realised minutes. The naive rule is 90 minutes if the player started their previous match, otherwise 0.

| Season | Model MAE | Naive MAE | Model RMSE | Naive RMSE | RMSE improvement |
|---|---:|---:|---:|---:|---:|
| 2023-24 | 14.70 | 11.89 | 23.36 | 29.02 | +19.5% |
| 2024-25 | 15.00 | 13.38 | 23.50 | 30.59 | +23.2% |
| 2025-26 | 13.44 | 11.77 | 21.88 | 28.66 | +23.7% |

**The naive rule wins on MAE and loses on RMSE, and that is expected rather than a defect.** MAE is minimised by the conditional median, and 61% of player-fixture rows are non appearances whose median is exactly 0. A hard 0 or 90 rule scores well on MAE precisely because it refuses to hedge, and it pays for that with a much worse RMSE and a far worse log loss when it is wrong. Splitting the last held out season shows where each one earns its error:

| Rows | Model MAE | Naive MAE |
|---|---:|---:|
| Appearances | 21.48 | 23.32 |
| Non appearances | 8.37 | 4.50 |

The model is better on the rows where somebody actually played. It is worse only on rows where the answer was zero and the naive rule guessed zero exactly. Note also that expected minutes is a diagnostic here, not the production output: the scoring module consumes p_appear and p_60, where the model wins outright.

Conditional means estimated from the training seasons, which is why expected minutes is not just p_start times 90:

| Position | E[min given start] | E[min given sub appearance] |
|---|---:|---:|
| GKP | 89.5 | 82.8 |
| DEF | 85.8 | 39.4 |
| MID | 80.5 | 28.4 |
| FWD | 79.9 | 24.6 |

## Calibration of p_start, held out 2025-26

A calibrated model puts the realised column next to the predicted one. Bins are fixed width, so sparse bins at the extremes are expected.

| Bin | Rows | Mean predicted | Realised |
|---|---:|---:|---:|
| 0.0-0.1 | 16,464 | 0.016 | 0.017 |
| 0.1-0.2 | 1,178 | 0.141 | 0.142 |
| 0.2-0.3 | 2,214 | 0.248 | 0.237 |
| 0.3-0.4 | 2 | 0.348 | 0.500 |
| 0.4-0.5 | 1,627 | 0.430 | 0.418 |
| 0.5-0.6 | 832 | 0.534 | 0.542 |
| 0.6-0.7 | 1,216 | 0.662 | 0.674 |
| 0.7-0.8 | 1,218 | 0.750 | 0.783 |
| 0.8-0.9 | 2,876 | 0.867 | 0.871 |
| 0.9-1.0 | 2,120 | 0.938 | 0.935 |

## Notes and known limits

- The start labels pass a hard integrity check: every held out season has exactly 8,360 starter rows, which is 380 fixtures times 22 starters. Eleven players per side per match is a law of the game, so an exact match across three seasons with different squad sizes says the labels are not drifting.
- 81.7% of start labels come from the archive's exact starts flag. The remainder, 2021-22 only, fall back to a minutes >= 45 proxy, which misclassifies roughly one non starter in seven.
- status and chance_of_playing_next_round are NOT used. The archive preserves only the end of season snapshot of both, so training on them would leak. They become usable once the weekly live snapshots have accumulated a season of history, and they are the single most valuable addition available to this model.
- Current season set piece order is excluded for the same reason. The previous season's order is used instead, which is leak free.
- Hyperparameters are untuned on purpose. Tuning before the evaluation harness exists would be chasing noise.

