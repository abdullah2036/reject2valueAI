"""Train the soft sensor (Mode A: sensors only, Mode B: sensors + last feed lab analysis).

LightGBM per target on log-concentration (as a residual over a physics prior, see log_offset), Saudi-likeness sample weights, quantile models
(alpha 0.1 / 0.9) for Mg and Ca. Grouped holdout by US state + held-out seawater body.

Run: python -m ml.soft_sensor.train
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ml.common.evaluation import FIG_DIR, group_split, metrics, parity_plot
from ml.soft_sensor.predict import MODEL_PATH, QUANTILE_TARGETS, TARGETS, build_features, log_offset
from simulation import ro_model as ro
from simulation.load_usgs import saudi_weight

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "reject_dataset.parquet"
TEST_PRED = ROOT / "data" / "processed" / "test_predictions.parquet"
METRICS = ROOT / "ml" / "models" / "metrics.json"

PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              n_jobs=4, verbose=-1, random_state=0)
Q_PARAMS = {**PARAMS, "n_estimators": 250}


def target_mask(df, t):
    m = df[t].notna() & (df[t] > 0)
    if t == "sio2":  # imputed silica is not a real measurement
        m &= ~df["sio2_imputed"]
    return m.to_numpy()


def baseline(df, tr, mode):
    """Naive physics. Mode A: average (Saudi-like) feed x concentration factor.
    Mode B: last lab feed analysis x concentration factor."""
    r = df["m_recovery"].fillna(df["m_recovery"].median()).to_numpy()
    rej = ro.mean_rejection()
    sw = (df["plant_type"] == "seawater").to_numpy()
    out = pd.DataFrame(index=df.index)
    for t in TARGETS:
        cf = (1 - r * (1 - rej[t])) / (1 - r)
        if mode == "A":
            ref_br = tr & ~sw & df["saudi_top20"].to_numpy()
            feed = np.where(sw, df.loc[tr & sw, f"feed_{t}"].mean(), df.loc[ref_br, f"feed_{t}"].mean())
        else:
            feed = df[f"lab_{t}"].to_numpy()
        out[t] = feed * cf
    return out


def evaluate(df, te, pred, base, mode):
    subsets = {"overall": te,
               "saudi_like_top20": te & df["saudi_top20"].to_numpy(),
               "brackish": te & (df["plant_type"] != "seawater").to_numpy(),
               "seawater": te & (df["plant_type"] == "seawater").to_numpy()}
    res = {}
    for name, s in subsets.items():
        res[name] = {}
        for t in TARGETS:
            m = s & target_mask(df, t)
            if m.sum() < 30:
                continue
            res[name][t] = metrics(df.loc[m, t], pred.loc[m, t])
            res[name][t]["baseline_within_15pct"] = metrics(df.loc[m, t], base.loc[m, t])["within_15pct"]
            res[name][t]["baseline_mape_pct"] = metrics(df.loc[m, t], base.loc[m, t])["mape_pct"]
        for t in QUANTILE_TARGETS:
            m = s & target_mask(df, t)
            inside = (df.loc[m, t] >= pred.loc[m, f"{t}_p10"]) & (df.loc[m, t] <= pred.loc[m, f"{t}_p90"])
            res[name][t]["p10_p90_coverage_pct"] = float(inside.mean() * 100)
    return res


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    tr, te, test_groups = group_split(df)
    w = saudi_weight(df["saudi_score"])
    # conformal calibration groups for the P10-P90 band (taken from the training groups only)
    tr_groups = df.loc[tr & (df["plant_type"] != "seawater").to_numpy(), "group"].unique()
    cal_groups = np.random.default_rng(1).choice(tr_groups, size=max(3, len(tr_groups) // 7), replace=False)
    cal = tr & df["group"].isin(cal_groups).to_numpy()
    print(f"train {tr.sum():,} rows / test {te.sum():,} rows; held-out groups: {test_groups}")
    bundle, report = {}, {"test_groups": test_groups, "conformal_calibration_groups": sorted(cal_groups.tolist()),
                          "sample_weight": "0.05 + saudi_score^2", "n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "modes": {}}
    test_out = df.loc[te, ["group", "plant_type", "saudi_top20", "m_reject_ph", "m_temp_c", "m_recovery",
                           "m_reject_flow_m3d", "reject_ph", "temp_c", "recovery", "reject_flow_m3d",
                           "feed_flow_m3d"] + TARGETS].copy()
    for mode in ["A", "B"]:
        x = build_features(df, mode)
        feats = list(x.columns)
        models = {"features": feats, "median": {}, "q10": {}, "q90": {}, "q_margin": {}}
        pred = pd.DataFrame(index=df.index)
        for t in TARGETS:
            m = tr & target_mask(df, t)
            off = log_offset(x, mode, t)
            y = np.log(np.maximum(df[t].to_numpy(), 1e-6)) - off
            m &= np.isfinite(y)
            models["median"][t] = LGBMRegressor(**PARAMS).fit(x[m], y[m], sample_weight=w[m])
            pred[t] = np.exp(models["median"][t].predict(x) + off)
            print(f"  mode {mode} {t:5s} trained ({time.time() - t0:.0f}s)")
        for t in QUANTILE_TARGETS:
            m = tr & ~cal & target_mask(df, t)
            off = log_offset(x, mode, t)
            y = np.log(np.maximum(df[t].to_numpy(), 1e-6)) - off
            m &= np.isfinite(y)
            raw = {}
            for q, a in [("q10", 0.1), ("q90", 0.9)]:
                models[q][t] = LGBMRegressor(objective="quantile", alpha=a, **Q_PARAMS).fit(
                    x[m], y[m], sample_weight=w[m])
                raw[q] = models[q][t].predict(x)
            # conformalised quantile regression: widen the band so it covers 80% on unseen groups
            c = cal & target_mask(df, t) & np.isfinite(y)
            score = np.maximum(raw["q10"][c] - y[c], y[c] - raw["q90"][c])
            margin = float(np.quantile(score, min(1.0, 0.8 * (1 + 1 / c.sum()))))
            models["q_margin"][t] = margin
            pred[f"{t}_p10"] = np.exp(raw["q10"] - margin + off)
            pred[f"{t}_p90"] = np.exp(raw["q90"] + margin + off)
            pred[f"{t}_p10"] = np.minimum(pred[f"{t}_p10"], pred[t])
            pred[f"{t}_p90"] = np.maximum(pred[f"{t}_p90"], pred[t])
        bundle[mode] = models
        base = baseline(df, tr, mode)
        report["modes"][mode] = evaluate(df, te, pred, base, mode)
        imp = pd.Series(models["median"]["mg"].booster_.feature_importance("gain"), index=feats)
        report["modes"][mode]["top_features_mg"] = (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict()
        for t in ["mg", "ca"]:
            m = te & target_mask(df, t)
            parity_plot(df.loc[m, t], pred.loc[m, t],
                        f"Reject {t.capitalize()} - Mode {mode} ({'sensors only' if mode == 'A' else 'sensors + last lab'})\n"
                        f"held-out states + Gulf seawater, n={m.sum():,}",
                        FIG_DIR / f"parity_{t}_mode{mode}.png",
                        pred.loc[m, f"{t}_p10"], pred.loc[m, f"{t}_p90"])
        for c in pred.columns:
            test_out[f"pred{mode}_{c}"] = pred.loc[te, c]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    test_out.to_parquet(TEST_PRED)
    report["train_seconds"] = round(time.time() - t0, 1)
    METRICS.write_text(json.dumps(report, indent=2))
    print_table(report)
    return report


def print_table(report):
    for mode, res in report["modes"].items():
        print(f"\nMode {mode}")
        for sub in ["overall", "saudi_like_top20", "seawater"]:
            if sub not in res:
                continue
            row = "  ".join(f"{t}:{v['within_15pct']:.0f}%" for t, v in res[sub].items())
            print(f"  {sub:18s} within+/-15%  {row}")


if __name__ == "__main__":
    main()
