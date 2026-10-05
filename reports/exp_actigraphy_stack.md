# Track A — Two-model actigraphy stack

- Actigraphy-covered labeled rows: 996/2736; stage-1 = actigraphy-only (lgb+cat, OOF); stage-2 = tabular ensemble + `acti_pred`.

- Stage-1 actigraphy-only QWK on covered subset: **0.1885** (how much sii signal is in actigraphy alone).


| config | LightGBM | XGBoost | CatBoost | Ensemble |
|---|---|---|---|---|
| baseline (tabular-only) | 0.4470 | 0.4568 | 0.4622 | 0.4591 |
| stack (tab + acti_pred) | 0.4461 | 0.4447 | 0.4503 | 0.4534 |

**Delta (ensemble): -0.0057**
