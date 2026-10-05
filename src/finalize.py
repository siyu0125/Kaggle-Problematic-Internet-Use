"""Finalize the best model found across all experiments and write submission.csv.

Best (by matched OOF QWK): pseudo-labeled CatBoost, tabular-only = 0.4612
(> tabular-only baseline CatBoost 0.4591; actigraphy stack and class-weighting did not help).

Produces OOF thresholds + bagged test predictions from per-fold pseudo-trained CatBoost.
Run: ./.venv/bin/python src/finalize.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb
from catboost import CatBoostRegressor

import model as M
from exp_class3_pseudo import prepare_full, PSEUDO_W

SEEDS = M.SEEDS
NS = M.N_SPLITS


def make_cat(seed):
    return CatBoostRegressor(iterations=800, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                             loss_function="RMSE", random_seed=seed, verbose=0,
                             allow_writing_files=False)


def make_gen(seed, cat_idx):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                             min_child_samples=40, random_state=seed, verbose=-1)


def main():
    X, y, Xun, Xtest, feats, cat_idx = prepare_full()
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(Xtest))

    for seed in SEEDS:
        skf = StratifiedKFold(NS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            Xtr, ytr = X.iloc[tr], y[tr]
            # per-fold pseudo-labels (soft, downweighted) — no leakage into validation
            gen = make_gen(seed, cat_idx); gen.fit(Xtr, ytr, categorical_feature=cat_idx)
            pseudo_y = np.clip(gen.predict(Xun), 0, 3)
            Xaug = pd.concat([Xtr, Xun], ignore_index=True)
            yaug = np.concatenate([ytr, pseudo_y])
            waug = np.concatenate([np.ones(len(ytr)), np.full(len(Xun), PSEUDO_W)])

            m = make_cat(seed); m.fit(Xaug, yaug, sample_weight=waug)
            oof[va] += m.predict(X.iloc[va]) / len(SEEDS)
            test_pred += m.predict(Xtest) / (len(SEEDS) * NS)

    thr, score = M.optimize_thresholds(y, oof)
    print(f"FINAL pseudo-labeled CatBoost OOF QWK = {score:.4f}  thresholds={np.round(thr,3).tolist()}")
    cm = confusion_matrix(y, M.to_classes(oof, thr))
    print("OOF confusion matrix:\n", cm)

    sub = pd.DataFrame({"id": pd.read_csv("data/test.csv")["id"],
                        "sii": M.to_classes(test_pred, thr).astype(int)})
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub["sii"].value_counts().sort_index().to_dict())

    # record final metrics
    metrics = json.load(open("reports/metrics.json"))
    metrics["final"] = {"model": "pseudo-labeled CatBoost (tabular-only)",
                        "oof_qwk": score, "thresholds": np.round(thr, 4).tolist()}
    json.dump(metrics, open("reports/metrics.json", "w"), indent=2)

    Path("reports/final_model.md").write_text(
        "# Final model\n\n"
        f"- **pseudo-labeled CatBoost (tabular-only): OOF QWK = {score:.4f}**\n"
        f"- thresholds: {np.round(thr,3).tolist()}\n"
        f"- submission class counts: {sub['sii'].value_counts().sort_index().to_dict()}\n\n"
        "Selected over: tabular baseline CatBoost 0.4591, ensembles ~0.458, "
        "actigraphy stack 0.453, class-weighting 0.453.\n\n"
        "## OOF confusion matrix\n```\n" + str(cm) + "\n```\n"
        "```\n" + classification_report(y, M.to_classes(oof, thr), zero_division=0) + "\n```\n")
    print("Wrote reports/final_model.md and updated reports/metrics.json")


if __name__ == "__main__":
    main()
