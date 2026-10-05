"""Aggregate per-participant actigraphy series into one feature row per id.

Scans series_{train,test}.parquet/id=*/part-0.parquet, computes activity /
wear-time / circadian / distribution features, caches to
data/actigraphy_{train,test}.parquet.

Run: ./.venv/bin/python src/actigraphy.py
"""
from __future__ import annotations
import glob, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
CACHE = Path.home() / ".cache/kagglehub/competitions/child-mind-institute-problematic-internet-use"
SRC = Path(os.environ.get("CMI_SERIES", CACHE))
OUT = Path("data")

USE_COLS = ["enmo", "anglez", "non-wear_flag", "light",
            "time_of_day", "weekday", "relative_date_PCIAT"]


def _pcts(x, qs=(10, 25, 50, 75, 90)):
    return {f"p{q}": float(np.nanpercentile(x, q)) for q in qs} if len(x) else {}


def features_for_id(path: str) -> dict:
    df = pd.read_parquet(path, columns=USE_COLS)
    n = len(df)
    f = {"acti_n_obs": n,
         "acti_n_days": int(df["relative_date_PCIAT"].nunique()),
         "acti_nonwear_frac": float(df["non-wear_flag"].mean())}

    wear = df[df["non-wear_flag"] == 0]
    f["acti_wear_hours"] = len(wear) * 5.0 / 3600.0  # 5-second epochs
    e = wear["enmo"].to_numpy()
    a = wear["anglez"].to_numpy()
    lt = wear["light"].to_numpy()

    if len(e) > 10:
        f.update({
            "acti_enmo_mean": float(np.nanmean(e)),
            "acti_enmo_std": float(np.nanstd(e)),
            "acti_enmo_max": float(np.nanmax(e)),
            "acti_enmo_skew": float(stats.skew(e, nan_policy="omit")),
            "acti_enmo_kurt": float(stats.kurtosis(e, nan_policy="omit")),
            "acti_frac_active": float(np.nanmean(e > 0.05)),      # moving
            "acti_frac_sedentary": float(np.nanmean(e < 0.01)),   # still
            "acti_anglez_mean": float(np.nanmean(a)),
            "acti_anglez_std": float(np.nanstd(a)),
            "acti_light_mean": float(np.nanmean(lt)),
            "acti_light_std": float(np.nanstd(lt)),
        })
        f.update({f"acti_enmo_{k}": v for k, v in _pcts(e).items()})

        # circadian: hour of day from nanoseconds-since-midnight
        hour = wear["time_of_day"].to_numpy() / 3.6e12
        night = e[(hour >= 0) & (hour < 6)]      # overnight
        day = e[(hour >= 8) & (hour < 20)]       # daytime
        evening = e[(hour >= 20) | (hour < 2)]   # late-night screen window
        f["acti_enmo_night_mean"] = float(np.nanmean(night)) if len(night) else np.nan
        f["acti_enmo_day_mean"] = float(np.nanmean(day)) if len(day) else np.nan
        f["acti_enmo_evening_mean"] = float(np.nanmean(evening)) if len(evening) else np.nan
        f["acti_night_day_ratio"] = (f["acti_enmo_night_mean"] /
                                     (f["acti_enmo_day_mean"] + 1e-6)) if len(night) and len(day) else np.nan
        # activity per 6-hour bin
        for b in range(4):
            seg = e[(hour >= b * 6) & (hour < (b + 1) * 6)]
            f[f"acti_enmo_bin{b}"] = float(np.nanmean(seg)) if len(seg) else np.nan

        # weekday vs weekend
        wd = wear["weekday"].to_numpy()
        wk = e[wd <= 5]; we = e[wd >= 6]
        f["acti_enmo_weekday_mean"] = float(np.nanmean(wk)) if len(wk) else np.nan
        f["acti_enmo_weekend_mean"] = float(np.nanmean(we)) if len(we) else np.nan
        f["acti_weekend_weekday_ratio"] = (f["acti_enmo_weekend_mean"] /
                                           (f["acti_enmo_weekday_mean"] + 1e-6)) if len(wk) and len(we) else np.nan
    return f


def build_split(split: str) -> pd.DataFrame:
    base = SRC / f"series_{split}.parquet"
    folders = sorted(glob.glob(str(base / "id=*")))
    rows = []
    for i, folder in enumerate(folders):
        sid = os.path.basename(folder).split("=")[1]
        part = os.path.join(folder, "part-0.parquet")
        if not os.path.exists(part):
            hits = glob.glob(os.path.join(folder, "*.parquet"))
            if not hits:
                continue
            part = hits[0]
        try:
            feats = features_for_id(part)
        except Exception as ex:
            print(f"  ! {sid}: {ex}")
            continue
        feats["id"] = sid
        rows.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {split}: {i+1}/{len(folders)}")
    df = pd.DataFrame(rows)
    print(f"{split}: {len(df)} ids, {df.shape[1]-1} features")
    return df


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for split in ["train", "test"]:
        base = SRC / f"series_{split}.parquet"
        if not base.exists():
            print(f"skip {split}: {base} not found")
            continue
        df = build_split(split)
        dst = OUT / f"actigraphy_{split}.parquet"
        df.to_parquet(dst, index=False)
        print(f"wrote {dst}")
