"""Track A — two-model actigraphy stack.

Stage 1: an actigraphy-only model trained on the ~36% of rows that HAVE actigraphy,
producing an out-of-fold meta-prediction `acti_pred` (NaN where no actigraphy).
Stage 2: the main tabular ensemble, with raw actigraphy columns replaced by the single
`acti_pred` meta-feature. Compares OOF QWK to a matched tabular-only baseline.

Run: ./.venv/bin/python src/exp_actigraphy_stack.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostRegressor

import model as M
import pipeline as P

SEEDS = M.SEEDS
NS = M.N_SPLITS


def stage1_oof(Xc, yc, Xtest_c):
    """OOF actigraphy-only predictions on covered rows + test predictions (avg lgb+cat, multi-seed)."""
    oof = np.zeros(len(Xc)); test = np.zeros(len(Xtest_c)) if len(Xtest_c) else np.zeros(0)
    for seed in SEEDS:
        skf = StratifiedKFold(NS, shuffle=True, random_state=seed)
        for tr, va in skf.split(Xc, yc):
            l = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                                  min_child_samples=20, random_state=seed, verbose=-1)
            c = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
                                  random_seed=seed, verbose=0, allow_writing_files=False)
            l.fit(Xc.iloc[tr], yc[tr]); c.fit(Xc.iloc[tr], yc[tr])
            oof[va] += (l.predict(Xc.iloc[va]) + c.predict(Xc.iloc[va])) / 2 / len(SEEDS)
    # test preds from models fit on all covered
    if len(Xtest_c):
        for seed in SEEDS:
            l = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                                  min_child_samples=20, random_state=seed, verbose=-1)
            c = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
                                  random_seed=seed, verbose=0, allow_writing_files=False)
            l.fit(Xc, yc); c.fit(Xc, yc)
            test += (l.predict(Xtest_c) + c.predict(Xtest_c)) / 2 / len(SEEDS)
    return oof, test


def ensemble_cv(X, y, Xtest, code_cols):
    """Main 3-model ensemble CV; returns per-model + ensemble OOF QWK and test preds."""
    cat_idx = [X.columns.get_loc(c) for c in code_cols if c in X.columns]
    oof = {k: np.zeros(len(X)) for k in M.MODELS}
    tp = {k: np.zeros(len(Xtest)) for k in M.MODELS}
    for seed in SEEDS:
        skf = StratifiedKFold(NS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            for name, fit in M.MODELS.items():
                m = fit(X.iloc[tr], y[tr], X.iloc[va], seed, cat_idx)
                oof[name][va] += m.predict(X.iloc[va]) / len(SEEDS)
                tp[name] += m.predict(Xtest) / (len(SEEDS) * NS)
    scores = {k: M.optimize_thresholds(y, oof[k])[1] for k in M.MODELS}
    ens_oof = np.mean([oof[k] for k in M.MODELS], axis=0)
    ens_test = np.mean([tp[k] for k in M.MODELS], axis=0)
    ens_thr, ens_score = M.optimize_thresholds(y, ens_oof)
    return scores, ens_score, ens_test, ens_thr


def main():
    X, y, Xtest, test, feats, code_cols = M.prepare(use_actigraphy=True)
    acti_cols = [c for c in feats if c.startswith("acti_")]      # 31 raw aggregates
    tab_feats = [c for c in feats if c not in acti_cols]         # keeps has_actigraphy flag
    has = X["has_actigraphy"].values == 1
    has_test = Xtest["has_actigraphy"].values == 1
    print(f"actigraphy-covered labeled rows: {has.sum()}/{len(X)} | test covered: {has_test.sum()}/{len(Xtest)}")

    # ---- matched baseline (tabular-only) ----
    base_scores, base_ens, _, _ = ensemble_cv(X[tab_feats], y, Xtest[tab_feats], code_cols)
    print(f"\nBASELINE (tabular-only): {base_scores} ens={base_ens:.4f}")

    # ---- stage 1: actigraphy meta-feature ----
    Xc = X.loc[has, acti_cols].reset_index(drop=True)
    yc = y[has]
    Xtest_c = Xtest.loc[has_test, acti_cols].reset_index(drop=True)
    oof_c, test_c = stage1_oof(Xc, yc, Xtest_c)
    stage1_qwk = M.optimize_thresholds(yc, oof_c)[1]
    print(f"\nSTAGE-1 actigraphy-only QWK (covered subset, n={has.sum()}): {stage1_qwk:.4f}")

    acti_pred = np.full(len(X), np.nan); acti_pred[has] = oof_c
    acti_pred_test = np.full(len(Xtest), np.nan)
    if has_test.sum():
        acti_pred_test[has_test] = test_c

    # ---- stage 2: tabular + acti_pred meta-feature ----
    X2 = X[tab_feats].copy(); X2["acti_pred"] = acti_pred
    Xtest2 = Xtest[tab_feats].copy(); Xtest2["acti_pred"] = acti_pred_test
    stk_scores, stk_ens, stk_test, stk_thr = ensemble_cv(X2, y, Xtest2, code_cols)
    print(f"\nSTACK (tabular + acti_pred): {stk_scores} ens={stk_ens:.4f}")

    md = ["# Track A — Two-model actigraphy stack\n",
          f"- Actigraphy-covered labeled rows: {int(has.sum())}/{len(X)}; "
          f"stage-1 = actigraphy-only (lgb+cat, OOF); stage-2 = tabular ensemble + `acti_pred`.\n",
          f"- Stage-1 actigraphy-only QWK on covered subset: **{stage1_qwk:.4f}** "
          "(how much sii signal is in actigraphy alone).\n",
          "\n| config | LightGBM | XGBoost | CatBoost | Ensemble |",
          "|---|---|---|---|---|",
          f"| baseline (tabular-only) | {base_scores['lgb']:.4f} | {base_scores['xgb']:.4f} "
          f"| {base_scores['cat']:.4f} | {base_ens:.4f} |",
          f"| stack (tab + acti_pred) | {stk_scores['lgb']:.4f} | {stk_scores['xgb']:.4f} "
          f"| {stk_scores['cat']:.4f} | {stk_ens:.4f} |",
          f"\n**Delta (ensemble): {stk_ens - base_ens:+.4f}**\n"]
    Path("reports/exp_actigraphy_stack.md").write_text("\n".join(md))
    json.dump({"baseline": base_scores | {"ensemble": base_ens},
               "stack": stk_scores | {"ensemble": stk_ens},
               "stage1_qwk": stage1_qwk, "delta_ens": stk_ens - base_ens},
              open("reports/exp_actigraphy_stack.json", "w"), indent=2)
    print(f"\nDELTA ensemble: {stk_ens - base_ens:+.4f}  -> reports/exp_actigraphy_stack.md")


if __name__ == "__main__":
    main()
