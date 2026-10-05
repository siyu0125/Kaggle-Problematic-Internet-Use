"""Track B — class-3 imbalance + pseudo-labeling.

Two levers vs a matched baseline, measured by OOF QWK (validation is always REAL labels):
  * class_weight: sqrt-inverse-frequency sample weights (upweights rare classes 2/3).
  * pseudo: per-fold self-training — a generator trained on the fold's real labels predicts
    soft continuous targets for the 1224 unlabeled rows, which are added to that fold's
    training set (downweighted). Regenerated inside every fold => no target leakage.

Run: ./.venv/bin/python src/exp_class3_pseudo.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import model as M
import pipeline as P

SEEDS = M.SEEDS
NS = M.N_SPLITS
PSEUDO_W = 0.3


def prepare_full():
    """Return labeled (X,y), unlabeled X, test, feats, cat code indices."""
    train, test, cat_cols = P.build(use_actigraphy=False)   # tabular-only (baseline winner)
    feats = P.feature_columns(train)
    code_cols = []
    for c in cat_cols:
        if c in feats:
            train[c] = train[c].cat.codes.astype("int32")
            test[c] = test[c].cat.codes.astype("int32")
            code_cols.append(c)
    lab = train[train[P.TARGET].notna()].reset_index(drop=True)
    unl = train[train[P.TARGET].isna()].reset_index(drop=True)
    X = lab[feats]; y = lab[P.TARGET].astype(int).values
    Xun = unl[feats]
    cat_idx = [X.columns.get_loc(c) for c in code_cols]
    return X, y, Xun, test[feats], feats, cat_idx


def class_weights(yv):
    K = 4; N = len(yv)
    counts = np.bincount(yv, minlength=K)
    w_per = np.sqrt(N / (K * np.maximum(counts, 1)))
    return w_per[yv]


def fit_all(Xtr, ytr, w, cat_idx, seed):
    out = {}
    l = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=31, max_depth=6,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                          reg_alpha=1.0, reg_lambda=5.0, min_child_samples=40,
                          random_state=seed, n_jobs=-1, verbose=-1)
    l.fit(Xtr, ytr, sample_weight=w, categorical_feature=cat_idx)
    out["lgb"] = l
    x = xgb.XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=5, subsample=0.8,
                         colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
                         random_state=seed, n_jobs=-1, tree_method="hist")
    x.fit(Xtr, ytr, sample_weight=w)
    out["xgb"] = x
    c = CatBoostRegressor(iterations=800, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                          loss_function="RMSE", random_seed=seed, verbose=0, allow_writing_files=False)
    c.fit(Xtr, ytr, sample_weight=w)
    out["cat"] = c
    return out


def run_config(X, y, Xun, cat_idx, use_class_weight, use_pseudo):
    oof = {k: np.zeros(len(X)) for k in ("lgb", "xgb", "cat")}
    for seed in SEEDS:
        skf = StratifiedKFold(NS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            Xtr, ytr = X.iloc[tr], y[tr]
            wtr = class_weights(ytr) if use_class_weight else np.ones(len(ytr))

            if use_pseudo and len(Xun):
                gen = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                        subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                        min_child_samples=40, random_state=seed, verbose=-1)
                gen.fit(Xtr, ytr, categorical_feature=cat_idx)
                pseudo_y = np.clip(gen.predict(Xun), 0, 3)
                Xtr = pd.concat([Xtr, Xun], ignore_index=True)
                ytr = np.concatenate([ytr, pseudo_y])
                wtr = np.concatenate([wtr, np.full(len(Xun), PSEUDO_W)])

            models = fit_all(Xtr, ytr, wtr, cat_idx, seed)
            for name, m in models.items():
                oof[name][va] += m.predict(X.iloc[va]) / len(SEEDS)

    scores, thr = {}, {}
    for k in oof:
        t, s = M.optimize_thresholds(y, oof[k]); scores[k] = s; thr[k] = t
    ens_oof = np.mean([oof[k] for k in oof], axis=0)
    et, es = M.optimize_thresholds(y, ens_oof)
    # class-3 recall from ensemble
    cm = confusion_matrix(y, M.to_classes(ens_oof, et))
    rec3 = cm[3, 3] / cm[3].sum() if cm.shape[0] > 3 and cm[3].sum() else 0.0
    rec2 = cm[2, 2] / cm[2].sum() if cm[2].sum() else 0.0
    return scores, es, rec2, rec3


def main():
    X, y, Xun, Xtest, feats, cat_idx = prepare_full()
    print(f"labeled: {len(X)} | unlabeled: {len(Xun)} | class counts: {np.bincount(y).tolist()}")

    configs = [("baseline", False, False),
               ("class_weight", True, False),
               ("pseudo", False, True),
               ("class_weight+pseudo", True, True)]
    rows, results = [], {}
    for tag, cw, ps in configs:
        sc, es, rec2, rec3 = run_config(X, y, Xun, cat_idx, cw, ps)
        results[tag] = {"lgb": sc["lgb"], "xgb": sc["xgb"], "cat": sc["cat"],
                        "ensemble": es, "recall_class2": rec2, "recall_class3": rec3}
        line = (f"{tag:>20}: ens={es:.4f}  cat={sc['cat']:.4f}  "
                f"rec2={rec2:.2f} rec3={rec3:.2f}")
        print(line); rows.append((tag, sc, es, rec2, rec3))

    base = results["baseline"]["ensemble"]
    md = ["# Track B — class-3 imbalance + pseudo-labeling\n",
          f"- labeled {len(X)}, unlabeled {len(Xun)}; tabular-only features; "
          f"{NS}-fold x {len(SEEDS)} seeds; validation = real labels only.\n",
          "\n| config | LightGBM | XGBoost | CatBoost | Ensemble | Δens | rec(2) | rec(3) |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, sc, es, rec2, rec3 in rows:
        md.append(f"| {tag} | {sc['lgb']:.4f} | {sc['xgb']:.4f} | {sc['cat']:.4f} "
                  f"| {es:.4f} | {es-base:+.4f} | {rec2:.2f} | {rec3:.2f} |")
    md.append("\n> Baseline class-3 recall is 0.00 (34 samples). Levers are judged on whether they lift "
              "ensemble QWK *and* class-2/3 recall without hurting overall QWK.\n")
    Path("reports/exp_class3_pseudo.md").write_text("\n".join(md))
    json.dump(results, open("reports/exp_class3_pseudo.json", "w"), indent=2)
    print("\nWrote reports/exp_class3_pseudo.md")


if __name__ == "__main__":
    main()
