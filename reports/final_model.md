# Final model

- **pseudo-labeled CatBoost (tabular-only): OOF QWK = 0.4612**
- thresholds: [0.527, 1.011, 2.757]
- submission class counts: {0: 10, 1: 10}

Selected over: tabular baseline CatBoost 0.4591, ensembles ~0.458, actigraphy stack 0.453, class-weighting 0.453.

## OOF confusion matrix
```
[[1078  424   92    0]
 [ 261  318  151    0]
 [  69  155  154    0]
 [   0    4   30    0]]
```
```
              precision    recall  f1-score   support

           0       0.77      0.68      0.72      1594
           1       0.35      0.44      0.39       730
           2       0.36      0.41      0.38       378
           3       0.00      0.00      0.00        34

    accuracy                           0.57      2736
   macro avg       0.37      0.38      0.37      2736
weighted avg       0.59      0.57      0.58      2736

```
