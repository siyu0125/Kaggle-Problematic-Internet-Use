# Track B — class-3 imbalance + pseudo-labeling

- labeled 2736, unlabeled 1224; tabular-only features; 5-fold x 3 seeds; validation = real labels only.


| config | LightGBM | XGBoost | CatBoost | Ensemble | Δens | rec(2) | rec(3) |
|---|---|---|---|---|---|---|---|
| baseline | 0.4470 | 0.4475 | 0.4591 | 0.4582 | +0.0000 | 0.45 | 0.00 |
| class_weight | 0.4583 | 0.4511 | 0.4514 | 0.4528 | -0.0055 | 0.46 | 0.00 |
| pseudo | 0.4457 | 0.4438 | 0.4612 | 0.4574 | -0.0009 | 0.40 | 0.00 |
| class_weight+pseudo | 0.4551 | 0.4487 | 0.4543 | 0.4536 | -0.0047 | 0.46 | 0.00 |

> Baseline class-3 recall is 0.00 (34 samples). Levers are judged on whether they lift ensemble QWK *and* class-2/3 recall without hurting overall QWK.
