# CMI — Problematic Internet Use: Study Report

Predicting the **Severity Impairment Index (`sii` ∈ {0,1,2,3})** for children/adolescents from physical,
fitness, demographic, and questionnaire data (Healthy Brain Network study).
Competition metric: **Quadratic Weighted Kappa (QWK)**.

---

## Executive summary
- **Best model: OOF-weight-optimized blend — QWK = 0.4623.**
  CatBoost 0.59 / LightGBM 0.21 / XGBoost 0.20, with CatBoost using per-fold pseudo-labeling.
- **Score progression (all on matched 5-fold × 3-seed OOF):**
  `0.459` tabular CatBoost → `0.4612` + pseudo-labeling → **`0.4623`** weight-optimized blend.
- **Three findings that shaped everything:** (1) the `PCIAT-*` columns leak the target and were excluded;
  (2) 31% of training rows are unlabeled and missingness is not random; (3) the visible test is 20 rows,
  so model selection relies on OOF QWK, never the public LB.
- **Two negative results, established rigorously:** wrist actigraphy carries almost no `sii` signal
  (actigraphy-only QWK ≈ 0.19) and did not help; class 3 (34 rows) is effectively unlearnable and class
  weighting did not help.

Pipeline: `src/eda.py` → `src/pipeline.py` (clean+features) → `src/model.py` / experiment scripts.
Companion outputs: [phase1_eda.md](phase1_eda.md), [phase4_model.md](phase4_model.md),
[exp_actigraphy_stack.md](exp_actigraphy_stack.md), [exp_class3_pseudo.md](exp_class3_pseudo.md),
[exp_blend.md](exp_blend.md), [metrics.json](metrics.json). A formal write-up is in [paper.md](paper.md).

---

## Phase 1 — Data Analysis
- **Shape:** train 3960×82, test 20×59 visible (hidden test much larger). 81 documented fields, 11 instruments.
- **Leakage audit:** 23 train-only columns = 20 `PCIAT-*` items + `PCIAT_Total` + `PCIAT-Season` + `sii`.
  `sii` reconstructs from `PCIAT_Total` with **100%** agreement (0–30→0, 31–49→1, 50–79→2, 80+→3)
  → **all `PCIAT-*` excluded from features.**
- **Target:** only **69.1%** of rows labeled (1224 missing `sii`). Severe imbalance
  58.3 / 26.7 / 13.8 / **1.2%** (class 3 = 34 rows).
- **Missing-not-at-random:** unlabeled rows are older, higher BMI, more internet hours (Mann–Whitney p≪0.001).
- **Heavy, instrument-structured missingness:** PAQ_A 88%, Fitness_Endurance ~81%, BIA ~50%
  → per-instrument "was completed" flags + per-row missing count.
- **Data-quality catches:** `CGAS-CGAS_Score` 999 sentinels (valid 1–100); `BIA-BIA_Fat` impossible
  negatives (min −8745); extreme/negative BIA masses; 7 implausible BMI rows.
- **Associations with `sii`:** age & body-size dominate the continuous signal (Kruskal ε²≈0.15, age-driven);
  **`computerinternet_hoursday` is the strongest categorical predictor (Cramér's V = 0.21);** sleep (SDS)
  and fitness also associate. Figures in `reports/figures/`.

## Phase 2 — Cleaning & Imputation (`pipeline.clean`)
- Sentinel/impossible → NaN (documented rules): CGAS ∉ [1,100]; BMI ∉ [8,60]; body-fat ∉ [0,75];
  BIA masses/rates ≤ 0; vitals & height/weight out of human range.
- **No value imputation** — gradient-boosted trees consume NaN natively; missingness is preserved as
  signal via indicator flags. This beat mean/median filling in early tests.

## Phase 3 — Feature Engineering (`pipeline.add_features`)
- 77 tabular features (19 engineered), 10 categoricals:
  instrument-completed flags, `n_missing`, `PAQ_total` (merges child/adolescent forms),
  `FGC_zone_sum`, BIA ratios, **age-normalized** `*_per_age`, `internet_hours_x_age`.
- Seasons/categoricals mapped to a shared category space (integer codes, −1 = missing).

## Phase 4 — Modeling (`model.py`)
- **Design:** regression on `sii` + **QWK-optimized rounding thresholds** (Nelder–Mead on OOF),
  StratifiedKFold(5) × 3 seeds, LightGBM + XGBoost + CatBoost + equal-weight ensemble. Both feature
  configurations evaluated; globally-best predictor by OOF QWK is submitted.

  | configuration | LightGBM | XGBoost | CatBoost | Ensemble |
  |---|---|---|---|---|
  | tabular-only | 0.447 | 0.448 | **0.459** | 0.458 |
  | tabular + actigraphy | 0.449 | 0.450 | 0.454 | 0.452 |

- **Top features:** `SDS_Total_Raw`, `PAQ_total`, `CGAS_Score`, blood pressure/heart rate,
  `Physical-Weight_per_age`, `internet_hours_x_age`, BMI — engineered features rank highly.
- **Confusion matrix (OOF):** classes 0–2 separated reasonably; **class 3 never predicted.**

## Phase 5 — Actigraphy features (`actigraphy.py`)
- Downloaded all 996 train + 2 test wrist-accelerometer series (6.3 GB); aggregated each participant's
  time series into **31 features**: enmo activity stats + percentiles, wear time, non-wear fraction,
  days recorded, anglez posture, light, 6-hour circadian bins, night/day & weekend/weekday ratios.
- **Result: no improvement.** Ensemble OOF QWK 0.458 → 0.452; no actigraphy feature in the LightGBM top-15.
  Cause: only **36%** of labeled rows have actigraphy, so 32 mostly-NaN columns add noise for the other 64%.

## Phase 6 — Improvement experiments (two parallel tracks)
Each experiment computes its own matched baseline (same seeds/folds); validation always uses real labels.

**Track A — two-model actigraphy stack (`exp_actigraphy_stack.py`).**
Stage-1 actigraphy-only model → OOF `acti_pred` meta-feature → stage-2 tabular ensemble.
- **Stage-1 actigraphy-only OOF QWK = 0.19** — decisive: actigraphy alone barely predicts `sii`.
- Stack 0.453 vs baseline 0.459 (−0.006). **Actigraphy is a dead end** via aggregates or a meta-feature.

**Track B — class-3 imbalance + pseudo-labeling (`exp_class3_pseudo.py`).**

| config | CatBoost | Ensemble | Δens | rec(2) | rec(3) |
|---|---|---|---|---|---|
| baseline | 0.4591 | 0.4582 | — | 0.45 | 0.00 |
| class_weight | 0.4514 | 0.4528 | −0.005 | 0.46 | 0.00 |
| **pseudo** | **0.4612** | 0.4574 | −0.001 | 0.40 | 0.00 |
| class_weight+pseudo | 0.4543 | 0.4536 | −0.005 | 0.46 | 0.00 |

- **Pseudo-labeling helps CatBoost (0.4591 → 0.4612):** per-fold self-training, an LGB generator predicts
  soft targets for the 1224 unlabeled rows, added to the training fold at weight 0.3 (regenerated inside
  each fold ⇒ no leakage). It hurts LGB/XGB, so the equal-weight ensemble doesn't gain.
- **Class weighting fails; class-3 recall stays 0.00** everywhere — 34 rows is too few, and QWK barely
  rewards it (distance-weighted, 1.2% of rows).

## Phase 7 — Weight-optimized blend (`exp_blend.py`)
Each base model uses its best pseudo setting (CatBoost→pseudo, LGB/XGB→plain); blend weights are then
optimized on OOF (linear stacking) instead of fixed at 1/3.

| candidate | OOF QWK |
|---|---|
| CatBoost + pseudo (single) | 0.4612 |
| equal-weight blend (best variants) | 0.4597 |
| **OOF-optimized blend** | **0.4623** |

- Optimized weights **CatBoost 0.59 / LGB 0.21 / XGB 0.20** — confirms the CatBoost-heavy hypothesis while
  weak learners still add small decorrelation. New best; submission regenerated.
- Caveat: 2 weights tuned on OOF QWK carry mild overfit risk; +0.0011 over cat-pseudo is small but
  directionally consistent.

## Limitations & next steps
1. **Class 3 is effectively unlearnable** (34 rows) — QWK gains live on the 0/1/2 boundary.
2. **Actigraphy exhausted** with aggregate/stack methods (signal ≈ QWK 0.19); only a learned sequence
   representation might help, expected payoff low.
3. **Diminishing returns** — remaining ideas (nested threshold optimization, CatBoost hyperparameter
   search, richer interactions) each look worth < 0.005 QWK.
4. Public LB is ~20 visible rows; trust OOF QWK.

## Reproducibility
```bash
# environment
python3 -m venv .venv && ./.venv/bin/pip install -U pandas numpy scipy scikit-learn statsmodels \
  matplotlib seaborn lightgbm xgboost catboost pyarrow kagglehub
# data: 4 CSVs in data/ ; actigraphy series via kagglehub into ~/.cache/kagglehub/...
./.venv/bin/python src/eda.py            # Phase 1 -> reports/phase1_eda.md + figures
./.venv/bin/python src/actigraphy.py     # Phase 5 features -> data/actigraphy_*.parquet
./.venv/bin/python src/model.py          # Phase 4 baseline + actigraphy comparison
./.venv/bin/python src/exp_actigraphy_stack.py   # Track A
./.venv/bin/python src/exp_class3_pseudo.py      # Track B
./.venv/bin/python src/exp_blend.py      # Phase 7 -> final submission.csv
```
Seeds fixed at {42, 2024, 7}; all reported QWK values are 5-fold × 3-seed out-of-fold.
