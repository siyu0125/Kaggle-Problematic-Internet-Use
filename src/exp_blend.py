"""Weighted / stacked blend + pseudo-labeling.

Motivation: the equal-weight ensemble (0.458) is dragged down by LGB/XGB (~0.447) while
CatBoost sits at ~0.46. Here each base model uses its *best* pseudo setting, then blend
weights are optimized on OOF (linear stacking) instead of fixed at 1/3 each.

Generates OOF + test preds for {lgb,xgb,cat} x {plain, pseudo} in one CV pass, then compares:
  - best single, equal-weight blend, OOF-optimized-weight blend.
If the winner beats the current final (0.4612), regenerates submission.csv.

Run: ./.venv/bin/python src/exp_blend.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import model as M
from exp_class3_pseudo import prepare_full, PSEUDO_W

SEEDS = M.SEEDS
NS = M.N_SPLITS
CURRENT_FINAL = 0.4612


def mk(name, seed):
    if name == "lgb":
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=31, max_depth=6,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                 reg_alpha=1.0, reg_lambda=5.0, min_child_samples=40,
                                 random_state=seed, n_jobs=-1, verbose=-1)
    if name == "xgb":
        return xgb.XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=5, subsample=0.8,
                                colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
                                random_state=seed, n_jobs=-1, tree_method="hist")
    return CatBoostRegressor(iterations=800, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                             loss_function="RMSE", random_seed=seed, verbose=0, allow_writing_files=False)


def generate():
    X, y, Xun, Xtest, feats, cat_idx = prepare_full()
    keys = [(m, mode) for m in ("lgb", "xgb", "cat") for mode in ("plain", "pseudo")]
    oof = {k: np.zeros(len(X)) for k in keys}
    tp = {k: np.zeros(len(Xtest)) for k in keys}

    for seed in SEEDS:
        skf = StratifiedKFold(NS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            Xtr, ytr = X.iloc[tr], y[tr]
            # per-fold pseudo augmentation (LGB generator, soft targets)
            gen = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                    subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                    min_child_samples=40, random_state=seed, verbose=-1)
            gen.fit(Xtr, ytr, categorical_feature=cat_idx)
            py = np.clip(gen.predict(Xun), 0, 3)
            Xaug = pd.concat([Xtr, Xun], ignore_index=True)
            yaug = np.concatenate([ytr, py])
            waug = np.concatenate([np.ones(len(ytr)), np.full(len(Xun), PSEUDO_W)])

            for name in ("lgb", "xgb", "cat"):
                # plain
                mp = mk(name, seed)
                if name == "lgb":
                    mp.fit(Xtr, ytr, categorical_feature=cat_idx)
                else:
                    mp.fit(Xtr, ytr)
                oof[(name, "plain")][va] += mp.predict(X.iloc[va]) / len(SEEDS)
                tp[(name, "plain")] += mp.predict(Xtest) / (len(SEEDS) * NS)
                # pseudo
                ms = mk(name, seed)
                if name == "lgb":
                    ms.fit(Xaug, yaug, sample_weight=waug, categorical_feature=cat_idx)
                else:
                    ms.fit(Xaug, yaug, sample_weight=waug)
                oof[(name, "pseudo")][va] += ms.predict(X.iloc[va]) / len(SEEDS)
                tp[(name, "pseudo")] += ms.predict(Xtest) / (len(SEEDS) * NS)

    return X, y, Xtest, oof, tp, keys


def optimize_weights(preds, y):
    """Maximize QWK over a simplex of blend weights; thresholds re-optimized per eval."""
    P = np.vstack(preds)  # (n_models, n)

    def neg(z):
        w = np.abs(z); s = w.sum()
        if s == 0:
            return 0.0
        w = w / s
        blend = w @ P
        return -M.optimize_thresholds(y, blend)[1]

    best = None
    for init in (np.ones(len(preds)), np.array([1.0] * (len(preds) - 1) + [3.0])):
        r = minimize(neg, init, method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    w = np.abs(best.x); w = w / w.sum()
    return w, -best.fun


def main():
    X, y, Xtest, oof, tp, keys = generate()

    singles = {k: M.optimize_thresholds(y, oof[k])[1] for k in keys}
    print("=== single (model, mode) OOF QWK ===")
    for k in keys:
        print(f"  {k[0]}:{k[1]:>6}  {singles[k]:.4f}")

    # best pseudo-variant per model
    best_var = {}
    for m in ("lgb", "xgb", "cat"):
        best_var[m] = max([(m, "plain"), (m, "pseudo")], key=lambda k: singles[k])
    print("\nbest variant per model:", {m: best_var[m][1] for m in best_var})
    chosen = [best_var[m] for m in ("lgb", "xgb", "cat")]
    preds_oof = [oof[k] for k in chosen]
    preds_test = [tp[k] for k in chosen]

    # equal-weight blend of best variants
    eq_blend = np.mean(preds_oof, axis=0)
    eq_thr, eq_qwk = M.optimize_thresholds(y, eq_blend)
    print(f"\nequal-weight blend (best variants): {eq_qwk:.4f}")

    # OOF-optimized weights
    w, opt_qwk = optimize_weights(preds_oof, y)
    print(f"optimized-weight blend: {opt_qwk:.4f}  weights="
          f"{dict(zip(['lgb','xgb','cat'], np.round(w,3)))}")

    cat_pseudo = singles[("cat", "pseudo")]
    candidates = {"cat_pseudo_single": (cat_pseudo, tp[("cat", "pseudo")],
                                        M.optimize_thresholds(y, oof[("cat", "pseudo")])[0]),
                  "equal_blend": (eq_qwk, np.mean(preds_test, axis=0), eq_thr),
                  "opt_blend": (opt_qwk, (w @ np.vstack(preds_test)),
                                M.optimize_thresholds(y, (w @ np.vstack(preds_oof)))[0])}
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_qwk, best_test, best_thr = candidates[best_name]
    print(f"\nBEST: {best_name} = {best_qwk:.4f}  (current final {CURRENT_FINAL:.4f})")

    md = ["# Weighted/stacked blend + pseudo-labeling\n",
          f"- {NS}-fold x {len(SEEDS)} seeds; each model uses its best pseudo variant; "
          "blend weights optimized on OOF.\n",
          "\n## Single (model, mode) OOF QWK\n",
          "| model | plain | pseudo |", "|---|---|---|",
          f"| lgb | {singles[('lgb','plain')]:.4f} | {singles[('lgb','pseudo')]:.4f} |",
          f"| xgb | {singles[('xgb','plain')]:.4f} | {singles[('xgb','pseudo')]:.4f} |",
          f"| cat | {singles[('cat','plain')]:.4f} | {singles[('cat','pseudo')]:.4f} |",
          "\n## Blends (over best variant per model)\n",
          f"- equal-weight: **{eq_qwk:.4f}**",
          f"- optimized-weight: **{opt_qwk:.4f}**  weights="
          f"{dict(zip(['lgb','xgb','cat'], np.round(w,3).tolist()))}",
          f"\n**Winner: `{best_name}` = {best_qwk:.4f}** vs current final {CURRENT_FINAL:.4f}.\n"]
    Path("reports/exp_blend.md").write_text("\n".join(md))
    json.dump({"singles": {f"{k[0]}_{k[1]}": singles[k] for k in keys},
               "equal_blend": eq_qwk, "opt_blend": opt_qwk,
               "opt_weights": dict(zip(["lgb", "xgb", "cat"], w.tolist())),
               "winner": best_name, "winner_qwk": best_qwk},
              open("reports/exp_blend.json", "w"), indent=2)

    if best_qwk > CURRENT_FINAL + 1e-4:
        sub = pd.DataFrame({"id": pd.read_csv("data/test.csv")["id"],
                            "sii": M.to_classes(best_test, best_thr).astype(int)})
        sub.to_csv("submission.csv", index=False)
        print(f"IMPROVED -> wrote submission.csv from {best_name}:",
              sub["sii"].value_counts().sort_index().to_dict())
        md.append(f"\n**submission.csv regenerated from `{best_name}`.**\n")
        Path("reports/exp_blend.md").write_text("\n".join(md))
    else:
        print("No improvement over current final; submission.csv unchanged.")
    print("Wrote reports/exp_blend.md and reports/exp_blend.json")


if __name__ == "__main__":
    main()
