# Weighted/stacked blend + pseudo-labeling

- 5-fold x 3 seeds; each model uses its best pseudo variant; blend weights optimized on OOF.


## Single (model, mode) OOF QWK

| model | plain | pseudo |
|---|---|---|
| lgb | 0.4470 | 0.4457 |
| xgb | 0.4475 | 0.4438 |
| cat | 0.4591 | 0.4612 |

## Blends (over best variant per model)

- equal-weight: **0.4597**
- optimized-weight: **0.4623**  weights={'lgb': 0.208, 'xgb': 0.198, 'cat': 0.594}

**Winner: `opt_blend` = 0.4623** vs current final 0.4612.


**submission.csv regenerated from `opt_blend`.**
