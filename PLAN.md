# Plan — CMI: Problematic Internet Use

Competition: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

**Deliverables (from project framework):**
1. End-to-end pipeline: data analysis → model development.
2. A comprehensive study report / paper.

Structured in 4 phases. **Phase 1 (Data Analysis) is deliberately front-loaded — we do not touch LightGBM until the data is understood.**

---

## 0. Problem framing & setup (do once)
- **Goal:** predict the **Severity Impairment Index (`sii`)**, ordinal 0–3 (None/Mild/Moderate/Severe), from physical/fitness/demographic data + wrist actigraphy.
- **Target derivation:** `sii` comes from `PCIAT-PCIAT_Total`: 0–30→0, 31–49→1, 50–79→2, 80+→3.
- **Metric:** **Quadratic Weighted Kappa (QWK)** — ordinal; distance-penalized.
- **Data files:** `train.csv` (~3960), `test.csv`, `series_train.parquet`/`series_test.parquet` (per-`id` actigraphy folders), `data_dictionary.csv`, `sample_submission.csv`.
- **Setup:**
  ```bash
  pip install kaggle pandas pyarrow numpy scipy scikit-learn statsmodels matplotlib seaborn lightgbm xgboost catboost
  kaggle competitions download -c child-mind-institute-problematic-internet-use
  unzip child-mind-institute-problematic-internet-use.zip -d data/
  ```

**Three gotchas that shape every phase below:**
1. **PCIAT leakage** — `PCIAT-*` columns are train-only and *define* the target. Exclude from features (reserve only for optional pseudo-labeling).
2. **Missing labels** — ~1200 train rows have no `sii`. Decide per-experiment: drop / keep for unsupervised use / pseudo-label.
3. **Hidden test is huge; visible test is tiny** — trust local CV, not the public LB.

---

## Phase 1: Data Analysis & Exploration (the focus)

### 1a. Data overview
- Load train/test; print shape, dtypes, memory.
- Group the ~80 columns by **instrument** (Demographics, Internet Use, CGAS, Physical Measures, FitnessGram Vitals/Treadmill/Child, BIA, PAQ, Sleep Disturbance Scale, PCIAT, Season). Use `data_dictionary.csv` to build a `column → instrument → type → description` table.
- Identify column types: continuous, ordinal questionnaire, categorical, `Season` (categorical per instrument), id.
- **Train vs test column diff** — confirm which columns (PCIAT, sii) are train-only. This is the leakage audit.

### 1b. Missingness analysis (central to this dataset)
- Missing-rate per column and per instrument; sort and visualize (bar + heatmap/matrix).
- **Missingness by instrument-completeness:** many participants skipped whole instruments → build "instrument present" flags; test whether missingness is informative (does it correlate with `sii`?).
- Actigraphy coverage: fraction of participants with any series; overlap of series availability with `sii`.
- Label missingness: count rows with missing `sii`; check if their feature profile differs from labeled rows (MCAR vs MNAR question).

### 1c. Target analysis
- `sii` class distribution + pie/bar — expect strong imbalance (class 3 rare).
- Reconstruct `sii` from `PCIAT-PCIAT_Total` to verify the mapping and quantify borderline cases near thresholds (30/49/79) — these drive QWK errors.
- Distribution of `PCIAT_Total` itself (continuous) — informs regression-on-total as an alternative target.

### 1d. Distribution analysis (univariate)
- Histograms / boxplots for continuous features (age, BMI, BIA metrics, fitness, sleep scores).
- Flag skew (log/quantile candidates) and outliers (physiologically impossible values — e.g., zero/negative BMI, implausible BIA readings → cleaning list for Phase 2).
- Value-counts for categoricals / ordinals; spot rare levels and anomalies.

### 1e. Cross-variable relationships (bivariate/multivariate)
- **Correlation matrix** (Spearman for ordinal/skewed) among continuous features; cluster to find redundant groups (e.g., BIA metrics collinear).
- **Feature ↔ target association:**
  - Continuous vs `sii`: ANOVA / Kruskal–Wallis + effect size (η²); boxplots by class.
  - Categorical/ordinal vs `sii`: chi-square / Cramér's V.
- Rank features by association strength → candidate predictor shortlist.
- Age/sex stratification: many measures are age-dependent → check interactions; motivates age-normalized features.
- Targeted pairplots for top-associated features.

### 1f. Actigraphy exploration
- Inspect one participant's series: fields `enmo`, `anglez`, `non-wear_flag`, `light`, `battery_voltage`, `time_of_day`, `weekday`, `relative_date_PCIAT`.
- Days recorded, wear-time fraction, sampling cadence per participant.
- Preliminary aggregates (mean/std enmo, non-wear %) vs `sii` — is there signal worth the engineering cost?

### 1g. Hypothesis formation (feeds the report)
- Write explicit, testable hypotheses (e.g., "higher sedentary actigraphy → higher sii", "poorer sleep score → higher sii") with p-values and effect sizes.
- Output of Phase 1: a **feature shortlist**, a **cleaning task list**, and a **hypothesis log** — all captured in the report.

**Phase 1 artifacts:** `notebooks/01_eda.ipynb`, saved figures in `reports/figures/`, a data-dictionary summary table, and an EDA write-up section.

---

## Phase 2: Data Cleaning & Imputation
- Fix categorical anomalies / inconsistent codings found in 1d.
- Outlier treatment: cap or drop physiologically impossible values (documented, per-column).
- **Imputation strategy chosen empirically, validated in CV:**
  - Tree models: keep NaN native (baseline — often best).
  - For linear/report models: median (skewed continuous), mode (categorical), KNN or group-based (by age/sex) where justified.
  - Always add **missing-indicator flags** for informative missingness (from 1b).
- Keep a clean, reproducible cleaning function reused for train and test.

## Phase 3: Feature Engineering
- **Tabular:** composite features (BMI-for-age, BIA fat/muscle ratios, activity scores), age×feature interactions, per-instrument missing-count, instrument-present flags; encode `Season` and ordinals.
- **Actigraphy → one row per `id`:** summary stats of enmo/anglez/light (mean/std/percentiles/skew/kurtosis), non-wear %, total wear time, days recorded, circadian features (time-of-day bins, day/night ratio, weekday vs weekend). Optional autoencoder/statistical embedding of the series.
- **Selection/reduction:** drop redundant collinear features (from 1e), importance-based pruning; optional PCA on collinear BIA block.
- **Transforms:** log/quantile for skewed continuous (mainly for non-tree models).

## Phase 4: Modeling
- **Validation:** StratifiedKFold (5-fold) on `sii`, multi-seed; report mean OOF QWK. Optimize rounding thresholds **on OOF predictions** (scipy Nelder-Mead), apply to test.
- **Approach:** regression on `sii` (or `PCIAT_Total`) + tuned thresholds — beats plain classification on QWK.
- **Models:** baseline logistic/linear → LightGBM → XGBoost → CatBoost; then a blended ensemble with seed averaging.
- **Advanced (only if OOF QWK improves):** pseudo-label missing-`sii` rows; autoencoder actigraphy features.
- **Evaluation for report:** QWK plus confusion matrix, per-class recall, calibration; feature importance / SHAP tied back to Phase 1 hypotheses.
- **Final:** retrain on full data; notebook runs offline end-to-end and emits valid `submission.csv`.

---

## Suggested repo layout
```
data/            raw + processed
notebooks/       01_eda, 02_cleaning, 03_features, 04_modeling
src/             reusable cleaning/feature/cv functions
reports/figures/ saved plots
reports/report.md study write-up (grows each phase)
PLAN.md
```

## Definition of done
- Phase 1 EDA notebook + figures + hypothesis log.
- Reproducible cleaning & feature pipeline (train/test consistent).
- Multi-seed CV with logged OOF QWK per experiment; ensemble beating baseline.
- Offline notebook producing valid submission.
- Study report covering all four phases.
