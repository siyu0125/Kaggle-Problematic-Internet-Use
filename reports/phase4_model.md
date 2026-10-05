# Phase 4 — Modeling results

- 5-fold x 3 seeds; regression + QWK-optimized thresholds.


## OOF QWK by configuration

| configuration | LightGBM | XGBoost | CatBoost | Ensemble |
|---|---|---|---|---|
| tabular-only | 0.4470 | 0.4475 | 0.4591 | 0.4582 |
| tabular+actigraphy | 0.4494 | 0.4502 | 0.4543 | 0.4518 |

**Best (by OOF QWK): `tabular-only:cat` = 0.4591** → used for submission.


## OOF confusion matrix (tabular-only ensemble)
```
[[1078  395  121    0]
 [ 257  292  181    0]
 [  63  145  170    0]
 [   0    5   29    0]]
```

```
              precision    recall  f1-score   support

           0       0.77      0.68      0.72      1594
           1       0.35      0.40      0.37       730
           2       0.34      0.45      0.39       378
           3       0.00      0.00      0.00        34

    accuracy                           0.56      2736
   macro avg       0.36      0.38      0.37      2736
weighted avg       0.59      0.56      0.57      2736

```


## Top 20 features (LightGBM, best config)
```
SDS-SDS_Total_Raw          540.9
PAQ_total                  396.5
CGAS-CGAS_Score            395.9
Physical-Systolic_BP       365.2
Physical-HeartRate         364.5
Physical-Weight_per_age    361.5
internet_hours_x_age       336.4
Physical-BMI               316.9
Physical-Weight            311.3
Physical-Diastolic_BP      311.1
Physical-Height_per_age    299.2
Physical-Height            288.1
PAQ_C-PAQ_C_Total          285.1
BIA-BIA_BMC                263.7
SDS-SDS_Total_T            239.8
FGC-FGC_CU                 239.3
BIA-BIA_DEE                213.9
FGC-FGC_PU                 210.2
FGC-FGC_SRR                209.4
BIA-BIA_LDM                208.4
```


## Submission
- model: `tabular-only:cat`
- class counts: {0: 7, 1: 12, 2: 1}
