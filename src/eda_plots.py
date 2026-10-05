"""Extra data-understanding plots to embed inline in the paper.
Writes to reports/figures/. Run: ./.venv/bin/python src/eda_plots.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
PAL = "viridis"

train = pd.read_csv("data/train.csv")
lab = train[train["sii"].notna()].copy(); lab["sii"] = lab["sii"].astype(int)


def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=110, bbox_inches="tight"); plt.close()
    print("wrote", name)


# 1) Age distribution by SII (age is the dominant confounder)
plt.figure(figsize=(7, 4))
sns.violinplot(data=lab, x="sii", y="Basic_Demos-Age", palette=PAL, cut=0)
plt.xlabel("SII (0=None … 3=Severe)"); plt.ylabel("Age (years)")
plt.title("Age distribution by SII class")
save("age_by_sii.png")

# 2) Internet hours/day vs SII — proportion within each SII class
ih = "PreInt_EduHx-computerinternet_hoursday"
ct = pd.crosstab(lab["sii"], lab[ih], normalize="index")
plt.figure(figsize=(7, 4))
bottom = np.zeros(len(ct))
colors = sns.color_palette(PAL, ct.shape[1])
labels_map = {0: "<1h", 1: "~1h", 2: "~2h", 3: "3h+"}
for j, col in enumerate(ct.columns):
    plt.bar(ct.index, ct[col].values, bottom=bottom, color=colors[j],
            label=labels_map.get(col, str(col)))
    bottom += ct[col].values
plt.xlabel("SII class"); plt.ylabel("proportion"); plt.title("Reported internet hours/day within each SII class")
plt.legend(title="hours/day", bbox_to_anchor=(1.02, 1), loc="upper left")
save("internet_hours_by_sii.png")

# 3) Feature→SII association ranking (Kruskal eps^2 for continuous, Cramer's V for categorical)
y = lab["sii"].values
num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])
            and not c.startswith("PCIAT") and c not in ("sii",) and train[c].dtype != bool
            and train[c].notna().mean() > 0.3]
cont = []
for c in num_cols:
    groups = [lab.loc[y == g, c].dropna().values for g in sorted(np.unique(y))]
    groups = [g for g in groups if len(g) > 5]
    if len(groups) >= 2:
        H, _ = stats.kruskal(*groups); N = sum(len(g) for g in groups); k = len(groups)
        cont.append((c, max((H - k + 1) / (N - k), 0)))
cont = pd.DataFrame(cont, columns=["feature", "assoc"]).sort_values("assoc", ascending=False).head(12)
plt.figure(figsize=(7, 5))
sns.barplot(data=cont, y="feature", x="assoc", palette="mako")
plt.xlabel("Kruskal–Wallis ε² (effect size)"); plt.ylabel("")
plt.title("Top continuous features associated with SII")
save("assoc_continuous.png")

# 4) Labeled vs unlabeled population (not-missing-at-random probe)
train["labeled"] = np.where(train["sii"].notna(), "labeled", "unlabeled")
probe = {"Basic_Demos-Age": "Age", "Physical-BMI": "BMI",
         "PreInt_EduHx-computerinternet_hoursday": "Internet hrs/day"}
plt.figure(figsize=(9, 3.4))
for i, (c, name) in enumerate(probe.items(), 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=train, x="labeled", y=c, palette=["#4C72B0", "#C44E52"])
    plt.title(name, fontsize=10); plt.xlabel(""); plt.ylabel("")
plt.suptitle("Labeled vs unlabeled participants differ (labels not missing at random)", y=1.04)
save("labeled_vs_unlabeled.png")

# 5) Actigraphy null result: enmo activity vs SII
acti_path = Path("data/actigraphy_train.parquet")
if acti_path.exists():
    acti = pd.read_parquet(acti_path)
    m = lab[["id", "sii"]].merge(acti[["id", "acti_enmo_mean", "acti_night_day_ratio"]], on="id")
    plt.figure(figsize=(9, 3.6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=m, x="sii", y="acti_enmo_mean", palette=PAL)
    plt.title("Mean daytime activity (ENMO) by SII", fontsize=10); plt.xlabel("SII")
    plt.subplot(1, 2, 2)
    sns.boxplot(data=m, x="sii", y="acti_night_day_ratio", palette=PAL)
    plt.ylim(0, 1.2)
    plt.title("Night/day activity ratio by SII", fontsize=10); plt.xlabel("SII")
    plt.suptitle("Actigraphy shows little separation across SII classes", y=1.04)
    save("actigraphy_by_sii.png")
else:
    print("skip actigraphy plot (no cache)")

print("done")
