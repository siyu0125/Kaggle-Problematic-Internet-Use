"""Phase 4 — Modeling. Regression + tuned QWK thresholds, StratifiedKFold CV,
LightGBM/XGBoost/CatBoost ensemble, multi-seed. Writes submission.csv and a report.

Run: ./.venv/bin/python src/model.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import pipeline as P

warnings.filterwarnings("ignore")
SEEDS = [42, 2024, 7]
N_SPLITS = 5
rng = np.random.RandomState(0)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def to_classes(x, thr):
    t1, t2, t3 = thr
    return np.where(x < t1, 0, np.where(x < t2, 1, np.where(x < t3, 2, 3)))


def optimize_thresholds(y_true, y_cont, init=(0.5, 1.5, 2.5)):
    def neg_qwk(thr):
        thr = np.sort(thr)
        return -qwk(y_true, to_classes(y_cont, thr))
    res = minimize(neg_qwk, init, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
    return np.sort(res.x), -res.fun


# ---- prepare data --------------------------------------------------------
def prepare(use_actigraphy=True):
    train, test, cat_cols = P.build(use_actigraphy=use_actigraphy)
    feats = P.feature_columns(train)
    # convert category dtype -> integer codes (-1 = missing); robust across all 3 libs
    code_cols = []
    for c in cat_cols:
        if c in feats:
            train[c] = train[c].cat.codes.astype("int32")
            test[c] = test[c].cat.codes.astype("int32")
            code_cols.append(c)
    labeled = train[train[P.TARGET].notna()].reset_index(drop=True)
    X = labeled[feats]
    y = labeled[P.TARGET].astype(int).values
    Xtest = test[feats]
    return X, y, Xtest, test, feats, code_cols


# ---- model factories -----------------------------------------------------
def fit_lgb(Xtr, ytr, Xva, seed, cat_idx):
    m = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.02, num_leaves=31, max_depth=6,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_child_samples=40,
        random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, categorical_feature=cat_idx)
    return m


def fit_xgb(Xtr, ytr, Xva, seed, cat_idx):
    m = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
        min_child_weight=5, random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(Xtr, ytr)
    return m


def fit_cat(Xtr, ytr, Xva, seed, cat_idx):
    m = CatBoostRegressor(
        iterations=800, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
        loss_function="RMSE", random_seed=seed, verbose=0,
        allow_writing_files=False)
    m.fit(Xtr, ytr)
    return m


MODELS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}


def run(use_actigraphy=True):
    """Train 3 models x N_SPLITS x seeds for one feature configuration.
    Returns a dict of OOF/test predictions, scores, importances."""
    X, y, Xtest, test, feats, code_cols = prepare(use_actigraphy=use_actigraphy)
    cat_idx = [X.columns.get_loc(c) for c in code_cols]
    n = len(X)
    cfg = "tabular+actigraphy" if use_actigraphy else "tabular-only"
    print(f"\n[{cfg}] labeled rows: {n} | features: {len(feats)} | seeds: {SEEDS}")

    oof = {k: np.zeros(n) for k in MODELS}
    test_pred = {k: np.zeros(len(Xtest)) for k in MODELS}
    importances = np.zeros(len(feats))

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X, y):
            Xtr, Xva, ytr = X.iloc[tr_idx], X.iloc[va_idx], y[tr_idx]
            for name, fit in MODELS.items():
                m = fit(Xtr, ytr, Xva, seed, cat_idx)
                oof[name][va_idx] += m.predict(Xva) / len(SEEDS)
                test_pred[name] += m.predict(Xtest) / (len(SEEDS) * N_SPLITS)
                if name == "lgb":
                    importances += m.feature_importances_ / (len(SEEDS) * N_SPLITS)

    results = {name: optimize_thresholds(y, oof[name]) for name in MODELS}  # name -> (thr, score)
    ens_oof = np.mean([oof[k] for k in MODELS], axis=0)
    ens_test = np.mean([test_pred[k] for k in MODELS], axis=0)
    ens_thr, ens_score = optimize_thresholds(y, ens_oof)
    for name in MODELS:
        print(f"  {name:>4}: QWK={results[name][1]:.4f}")
    print(f"  ENSEMBLE: QWK={ens_score:.4f}")

    return dict(cfg=cfg, use_actigraphy=use_actigraphy, y=y, n=n, feats=feats,
                results=results, ens_oof=ens_oof, ens_thr=ens_thr, ens_score=ens_score,
                ens_test=ens_test, test_pred=test_pred, test=test,
                importances=importances)


def candidates(r):
    """Yield (tag, oof_score, test_cont, thr) for every selectable predictor in a run."""
    for name, (thr, score) in r["results"].items():
        yield (f"{r['cfg']}:{name}", score, r["test_pred"][name], thr)
    yield (f"{r['cfg']}:ensemble", r["ens_score"], r["ens_test"], r["ens_thr"])


def main():
    from sklearn.metrics import confusion_matrix, classification_report
    runs = [run(use_actigraphy=False), run(use_actigraphy=True)]

    # global selection across both configs by OOF QWK
    all_cands = [c for r in runs for c in candidates(r)]
    best = max(all_cands, key=lambda c: c[1])
    best_tag, best_score, best_cont, best_thr = best

    # comparison table
    md = ["# Phase 4 — Modeling results\n",
          f"- {N_SPLITS}-fold x {len(SEEDS)} seeds; regression + QWK-optimized thresholds.\n",
          "\n## OOF QWK by configuration\n",
          "| configuration | LightGBM | XGBoost | CatBoost | Ensemble |",
          "|---|---|---|---|---|"]
    print("\n=== COMPARISON (OOF QWK) ===")
    for r in runs:
        row = (f"| {r['cfg']} | {r['results']['lgb'][1]:.4f} | {r['results']['xgb'][1]:.4f} "
               f"| {r['results']['cat'][1]:.4f} | {r['ens_score']:.4f} |")
        md.append(row); print(row)
    md.append(f"\n**Best (by OOF QWK): `{best_tag}` = {best_score:.4f}** → used for submission.\n")
    print(f"\nBest: {best_tag} = {best_score:.4f}")

    # diagnostics from the run that owns the best config
    best_cfg = best_tag.split(":")[0]
    r = next(r for r in runs if r["cfg"] == best_cfg)
    oof_cls = to_classes(r["ens_oof"], r["ens_thr"])
    cm = confusion_matrix(r["y"], oof_cls)
    md.append(f"\n## OOF confusion matrix ({best_cfg} ensemble)\n```\n{cm}\n```\n")
    md.append("```\n" + classification_report(r["y"], oof_cls, zero_division=0) + "\n```\n")
    imp = pd.Series(r["importances"], index=r["feats"]).sort_values(ascending=False)
    md.append("\n## Top 20 features (LightGBM, best config)\n```\n" + imp.head(20).round(1).to_string() + "\n```\n")
    print("\nConfusion matrix (best cfg ensemble):\n", cm)
    print("\nTop 15 features:\n", imp.head(15).round(1).to_string())

    # submission from globally-best predictor
    sub = pd.DataFrame({"id": r["test"][P.ID], "sii": to_classes(best_cont, best_thr).astype(int)})
    sub.to_csv("submission.csv", index=False)
    print("\nWrote submission.csv:", sub["sii"].value_counts().sort_index().to_dict())
    md.append(f"\n## Submission\n- model: `{best_tag}`\n- class counts: "
              f"{sub['sii'].value_counts().sort_index().to_dict()}\n")

    Path("reports/phase4_model.md").write_text("\n".join(md))
    json.dump({r["cfg"]: {"oof_qwk": {k: r["results"][k][1] for k in MODELS},
                          "ensemble_qwk": r["ens_score"]} for r in runs}
              | {"best": best_tag, "best_qwk": best_score},
              open("reports/metrics.json", "w"), indent=2)
    print("\nWrote reports/phase4_model.md and reports/metrics.json")


if __name__ == "__main__":
    main()
