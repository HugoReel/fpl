# Experiment 1: minutes versus rates error attribution

Season 2025-26, gameweek 6 onward, 26,159 player-fixture rows. All expected points and all realised points go through `scoring/rules_2026_27.py`. Realised points are recomputed from components rather than read from the stored total.

## Result

- Perfect minutes knowledge is worth **0.280 MAE** per player-fixture.
- Perfect knowledge of a player's true season rates is worth **0.079 MAE**.
- The trained minutes model captured **0.057 MAE**, which is **20%** of what perfect minutes would have bought.

**Minutes dominate rates.** Effort spent on the minutes model pays back more than the same effort spent on per-90 modelling, which is the ordering this project committed to before measuring. The commitment survives contact with the data.

For context only, knowing this fixture's actual outcomes is worth 0.666 MAE. That number is not a modelling target. It reveals the answer rather than describing the player, and no model can approach it even in principle, because goals stay random however well the rate is known. It is reported to show how much of the residual error is irreducible match noise rather than something a better model could remove.

## Variants

| Variant | Minutes source | Rate source |
|---|---|---|
| A oracle_minutes | true minutes | trailing per-90 |
| B naive_minutes | started last match implies 90 | trailing per-90 |
| M model_minutes | trained minutes model | trailing per-90 |
| D oracle_rates | started last match implies 90 | player's true season per-90 |
| C oracle_outcomes | started last match implies 90 | this fixture's true outcomes |

A and D are the like-for-like pair: each fixes one input at its true value and leaves everything else uncertain. B is the fully naive system and M is the one that could actually be deployed on a Friday, since it is the only variant here that uses no information from after the deadline.

## Error by variant

| Variant | MAE | RMSE | GKP | DEF | MID | FWD |
|---|---:|---:|---:|---:|---:|---:|
| A oracle_minutes | 0.753 | 1.856 | 0.484 | 0.877 | 0.697 | 0.889 |
| B naive_minutes | 1.033 | 2.182 | 0.556 | 1.133 | 1.036 | 1.220 |
| M model_minutes | 0.975 | 2.021 | 0.567 | 1.114 | 0.949 | 1.097 |
| D oracle_rates | 0.954 | 2.008 | 0.507 | 1.037 | 0.960 | 1.148 |
| C oracle_outcomes | 0.367 | 0.831 | 0.159 | 0.460 | 0.354 | 0.357 |

## Conclusion

Swapping naive minutes for true minutes removes 0.280 MAE, and swapping trailing rates for a player's true season rates removes 0.079. The minutes model recovers 20% of the minutes gap while using nothing from after the deadline.

The unrecovered majority of that gap is mostly team news. The model cannot see a manager's Friday press conference, and it cannot see the availability flags either, because the archive preserves only an end of season snapshot of them. That is the single largest identified improvement available, it requires no new modelling technique, and it needs only the weekly live snapshots that `ingest/snapshot.py` is already collecting. Roughly a season of them turns status and chance_of_playing_next_round into usable, leak free features.

These rate estimates are deliberately crude trailing averages and are identical across variants A, B and M, so the comparison isolates minutes cleanly. They are not a claim about how good rate modelling can get.

