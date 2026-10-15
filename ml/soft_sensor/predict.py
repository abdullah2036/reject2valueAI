"""Soft-sensor inference: estimate reject chemistry from cheap online sensors (+ optional last lab).

Mode A - sensors only: reject EC, reject pH, temperature, feed EC, recovery, plant type.
Mode B - Mode A + the last lab analysis of the feed water (may be weeks old).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulation import chemistry as chem
from simulation.ro_model import mean_rejection

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "soft_sensor.joblib"
TARGETS = ["ca", "mg", "na", "k", "cl", "so4", "hco3", "sio2"]
QUANTILE_TARGETS = ["mg", "ca"]
SENSOR_COLS = ["m_reject_ec", "m_reject_ph", "m_temp_c", "m_feed_ec", "m_recovery"]
LAB_COLS = [f"lab_{i}" for i in chem.IONS]
MEAN_REJECTION = mean_rejection()


def build_features(df, mode):
    """Physics-informed features. `df` holds SENSOR_COLS, `plant_type` and (Mode B) LAB_COLS."""
    x = pd.DataFrame(index=df.index)
    for c in SENSOR_COLS:
        x[c] = df[c].astype(float)
    x["is_seawater"] = (df["plant_type"] == "seawater").astype(float)
    x["cf"] = 1.0 / (1.0 - x["m_recovery"])
    x["log_reject_ec"] = np.log10(x["m_reject_ec"])
    x["log_feed_ec"] = np.log10(x["m_feed_ec"])
    x["ec_ratio"] = x["m_reject_ec"] / x["m_feed_ec"]
    x["ec_ratio_over_cf"] = x["ec_ratio"] / x["cf"]  # ~ effective salt rejection / sensor consistency
    x["tds_fao_reject"] = chem.tds_from_ec_fao(x["m_reject_ec"])
    if mode == "B":
        lab = pd.DataFrame({i: df[f"lab_{i}"].astype(float).to_numpy() for i in chem.IONS}, index=df.index)
        lab_ec = chem.predict_ec(lab)
        # how far the feed has moved since the lab sample (measured feed EC vs lab-implied EC)
        x["feed_drift"] = x["m_feed_ec"] / lab_ec
        scale_ec = x["m_reject_ec"] / lab_ec  # concentration factor implied by conductivity
        for i in chem.IONS:
            x[f"lab_{i}"] = lab[i]
            x[f"lab_{i}_x_cf"] = lab[i] * x["cf"]
            x[f"lab_{i}_x_ec"] = lab[i] * scale_ec
        tot = sum(chem.meq(lab, i) for i in chem.CATIONS + chem.ANIONS)
        for i in ["ca", "mg", "so4", "hco3", "na"]:
            x[f"lab_{i}_frac"] = chem.meq(lab, i) / tot
    return x


def log_offset(x, mode, t):
    """Physics prior the model corrects (residual learning, so predictions extrapolate with salinity).
    Mode A: log(reject EC) - the model learns ion/EC ratios.
    Mode B: log(last-lab feed ion x concentration factor) - the model learns the mass-balance correction."""
    if mode == "A":
        ec = x["m_reject_ec"].fillna(x["m_feed_ec"] * x["cf"])
        return np.log(ec.to_numpy(dtype=float))
    rej = MEAN_REJECTION[t]
    r = x["m_recovery"].to_numpy(dtype=float)
    cf = (1 - r * (1 - rej)) / (1 - r)
    cf = np.where(np.isfinite(cf), cf, x["ec_ratio"].to_numpy(dtype=float))
    return np.log(np.maximum(x[f"lab_{t}"].to_numpy(dtype=float) * cf, 1e-6))


def has_lab(df):
    return all(c in df.columns for c in LAB_COLS) and df[LAB_COLS[:7]].notna().all(axis=1)


class SoftSensor:
    def __init__(self, path=MODEL_PATH):
        # joblib (pickle) is only safe for files we produced ourselves: this loads the models
        # written by ml/soft_sensor/train.py on this machine, never a downloaded/untrusted file.
        self.bundle = joblib.load(path)

    def predict(self, df, mode=None, contributions=False):
        """Return dict with per-target median predictions (mg/L), P10/P90 for QUANTILE_TARGETS,
        the mode used, and optionally LightGBM feature contributions (log-space) per target."""
        mode = mode or ("B" if bool(np.all(has_lab(df))) else "A")
        m = self.bundle[mode]
        xf = build_features(df, mode)
        x = xf[m["features"]]
        out = {"mode": mode, "pred": pd.DataFrame(index=df.index)}
        for t in TARGETS:
            out["pred"][t] = np.exp(m["median"][t].predict(x) + log_offset(xf, mode, t))
        for t in QUANTILE_TARGETS:
            off = log_offset(xf, mode, t)
            mg_ = m["q_margin"][t]
            lo = np.exp(m["q10"][t].predict(x) - mg_ + off)
            hi = np.exp(m["q90"][t].predict(x) + mg_ + off)
            med = out["pred"][t].to_numpy()
            out["pred"][f"{t}_p10"] = np.minimum(lo, med)
            out["pred"][f"{t}_p90"] = np.maximum(hi, med)
        if contributions:
            out["contrib"] = {t: pd.DataFrame(m["median"][t].predict(x, pred_contrib=True)[:, :-1],
                                              columns=m["features"], index=df.index)
                              for t in ["mg", "ca", "so4"]}
        return out
