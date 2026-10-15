"""Water chemistry helpers: TDS/EC, ionic strength, LSI, gypsum saturation, SAR, Mg(OH)2 potential.

All concentrations are mg/L unless stated. Functions accept pandas DataFrames / Series or
dicts of numpy arrays and are vectorised so the simulator and engine can run on 100k+ rows.

Activity model: Davies equation (vectorised). Davies ignores ion pairing (CaSO4°), so the
gypsum saturation ratio is corrected with a calibration fitted against PHREEQC
(phreeqc.dat) when the `phreeqc` package is available -- see `fit_gypsum_calibration`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "ml" / "models"
EC_MODEL_PATH = MODELS_DIR / "ec_model.json"
GYPSUM_CAL_PATH = MODELS_DIR / "gypsum_calibration.json"

IONS = ["ca", "mg", "na", "k", "cl", "so4", "hco3", "sio2"]
MW = {"ca": 40.078, "mg": 24.305, "na": 22.990, "k": 39.098, "cl": 35.453,
      "so4": 96.06, "hco3": 61.017, "sio2": 60.08, "br": 79.904}
CHARGE = {"ca": 2, "mg": 2, "na": 1, "k": 1, "cl": -1, "so4": -2, "hco3": -1, "sio2": 0, "br": -1}
CATIONS = ["ca", "mg", "na", "k"]
ANIONS = ["cl", "so4", "hco3"]

KSP_GYPSUM_25C = 10 ** -4.58          # source: assumptions.yaml scaling.gypsum_ksp
MG_TO_MGOH2 = 58.32 / 24.305          # mass ratio Mg(OH)2 / Mg


def _get(df, ion, default=0.0):
    if isinstance(df, pd.DataFrame):
        if ion in df.columns:
            return df[ion].fillna(default).to_numpy(dtype=float)
        return np.full(len(df), default, dtype=float)
    v = df.get(ion, default)
    return np.asarray(default if v is None else v, dtype=float)


def meq(df, ion):
    """mg/L -> meq/L (absolute charge)."""
    return _get(df, ion) / MW[ion] * abs(CHARGE[ion])


def charge_balance_error(df):
    """Percent charge-balance error: (cations - anions) / (cations + anions) * 100."""
    cat = sum(meq(df, i) for i in CATIONS)
    an = sum(meq(df, i) for i in ANIONS)
    return (cat - an) / np.maximum(cat + an, 1e-9) * 100.0


def tds_from_ions(df):
    """Sum-of-constituents TDS (USGS convention: HCO3 counted as CO3, factor 0.4917)."""
    tds = sum(_get(df, i) for i in ["ca", "mg", "na", "k", "cl", "so4", "sio2"])
    return tds + 0.4917 * _get(df, "hco3")


def ionic_strength(df):
    """Ionic strength (mol/kg, approximating mg/L ~ mg/kg)."""
    i = 0.0
    for ion in ["ca", "mg", "na", "k", "cl", "so4", "hco3", "br"]:
        i = i + _get(df, ion) / MW[ion] / 1000.0 * CHARGE[ion] ** 2
    return 0.5 * i


def davies_log_gamma(z, ionic_str, temp_c=25.0):
    """Davies activity coefficient (log10). A(T) linearised from 0-60 C tables."""
    a = 0.4883 + 8.074e-4 * np.asarray(temp_c, dtype=float)
    s = np.sqrt(ionic_str)
    return -a * z ** 2 * (s / (1.0 + s) - 0.3 * ionic_str)


def gypsum_sr_davies(df, temp_c=25.0):
    """Gypsum saturation ratio IAP/Ksp with Davies activities (no ion pairing)."""
    ist = ionic_strength(df)
    lg = davies_log_gamma(2, ist, temp_c)
    m_ca = _get(df, "ca") / MW["ca"] / 1000.0
    m_so4 = _get(df, "so4") / MW["so4"] / 1000.0
    log_iap = np.log10(np.maximum(m_ca, 1e-12)) + np.log10(np.maximum(m_so4, 1e-12)) + 2 * lg
    return 10 ** (log_iap - np.log10(KSP_GYPSUM_25C))


_GYP_CAL = None


def _gypsum_calibration():
    global _GYP_CAL
    if _GYP_CAL is None:
        _GYP_CAL = json.loads(GYPSUM_CAL_PATH.read_text()) if GYPSUM_CAL_PATH.exists() else {}
    return _GYP_CAL


def gypsum_sr(df, temp_c=25.0):
    """Gypsum saturation ratio. Davies + PHREEQC-fitted correction when calibration exists:
    log10 SR = log10 SR_davies + c0 + c1*sqrt(I) + c2*I (+ c3*(T-25))."""
    sr = gypsum_sr_davies(df, temp_c)
    cal = _gypsum_calibration()
    if not cal:
        return sr
    ist = ionic_strength(df)
    c = cal["coef"]
    corr = c[0] + c[1] * np.sqrt(ist) + c[2] * ist + c[3] * (np.asarray(temp_c, dtype=float) - 25.0)
    return sr * 10 ** corr


def lsi(df, ph, temp_c, tds=None):
    """Langelier Saturation Index.
    pHs = (9.3 + A + B) - (C + D); A=(log10 TDS - 1)/10; B=-13.12 log10(T_K)+34.55;
    C=log10(Ca as CaCO3)-0.4; D=log10(alk as CaCO3). Ca as CaCO3 = Ca*2.497, alk = HCO3*0.820.
    Returns (lsi, valid_flag) - the formula is only valid for TDS < 10,000 mg/L."""
    tds = tds_from_ions(df) if tds is None else np.asarray(tds, dtype=float)
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    a = (np.log10(np.maximum(tds, 1.0)) - 1.0) / 10.0
    b = -13.12 * np.log10(t_k) + 34.55
    c = np.log10(np.maximum(_get(df, "ca") * 2.497, 1e-6)) - 0.4
    d = np.log10(np.maximum(_get(df, "hco3") * 0.820, 1e-6))
    phs = (9.3 + a + b) - (c + d)
    return np.asarray(ph, dtype=float) - phs, tds < 10000


def sar(df):
    """Sodium adsorption ratio, meq/L: Na / sqrt((Ca + Mg) / 2)."""
    return meq(df, "na") / np.sqrt(np.maximum((meq(df, "ca") + meq(df, "mg")) / 2.0, 1e-9))


def mg_oh2_potential(df):
    """Stoichiometric Mg(OH)2 mass potential, mg/L (= g/m3)."""
    return _get(df, "mg") * MG_TO_MGOH2


# ---------------------------------------------------------------- conductivity model
def _ec_features(df):
    tds = np.maximum(tds_from_ions(df), 1.0)
    tot = sum(meq(df, i) for i in CATIONS + ANIONS) + 1e-9
    lt = np.log10(tds)
    return np.column_stack([np.ones_like(lt), lt, lt ** 2,
                            meq(df, "so4") / tot, meq(df, "hco3") / tot,
                            (meq(df, "ca") + meq(df, "mg")) / tot])


def fit_ec_model(df, ec_col="ec_us_cm"):
    """Fit log10(EC) = f(log TDS, ionic composition) on real (ions, specific conductance) pairs."""
    m = df[ec_col].notna() & (df[ec_col] > 0)
    x, y = _ec_features(df[m]), np.log10(df.loc[m, ec_col].to_numpy(dtype=float))
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    tds = tds_from_ions(df[m])
    model = {"coef": coef.tolist(), "tds_min": float(np.percentile(tds, 0.5)),
             "tds_max": float(np.percentile(tds, 99.5)), "n": int(m.sum()),
             "rmse_log10": float(np.sqrt(np.mean(resid ** 2))),
             "median_abs_pct_err": float(np.median(np.abs(10 ** resid - 1)) * 100),
             "fallback": "beyond tds_max: EC grows with FAO slope (TDS mg/L = EC dS/m x 800 above 5 dS/m), "
                         "below tds_min: TDS = EC dS/m x 640 (FAO Irrigation & Drainage Paper 29)"}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EC_MODEL_PATH.write_text(json.dumps(model, indent=2))
    global _EC_MODEL
    _EC_MODEL = model
    return model


_EC_MODEL = None


def predict_ec(df):
    """Specific conductance (uS/cm at 25 C) from ion composition."""
    global _EC_MODEL
    if _EC_MODEL is None:
        _EC_MODEL = json.loads(EC_MODEL_PATH.read_text()) if EC_MODEL_PATH.exists() else None
    tds = tds_from_ions(df)
    if _EC_MODEL is None:  # pure FAO factors
        ec = np.where(tds / 640.0 < 5.0, tds / 640.0, tds / 800.0) * 1000.0
        return ec
    coef = np.asarray(_EC_MODEL["coef"])
    ec = 10 ** (_ec_features(df) @ coef)
    tmax, tmin = _EC_MODEL["tds_max"], _EC_MODEL["tds_min"]
    hi = tds > tmax
    if hi.any():
        # EC at the edge of the fitted range for this composition, then FAO slope (1/0.8 uS/cm per mg/L)
        scale = np.where(hi, tmax / np.maximum(tds, 1e-9), 1.0)
        edge = pd.DataFrame({i: _get(df, i) * scale for i in IONS})
        ec_edge = 10 ** (_ec_features(edge) @ coef)
        ec = np.where(hi, ec_edge + (tds - tmax) / 0.8, ec)
    lo = tds < tmin
    ec = np.where(lo, tds / 0.640, ec)
    return ec


def tds_from_ec_fao(ec_us_cm):
    """FAO-29 conversion: TDS = EC(dS/m) x 640 below 5 dS/m, x 800 above."""
    ds = np.asarray(ec_us_cm, dtype=float) / 1000.0
    return np.where(ds < 5.0, ds * 640.0, ds * 800.0)


# ---------------------------------------------------------------- PHREEQC
def phreeqc_available():
    try:
        import phreeqc  # noqa: F401
        return True
    except Exception:
        return False


def phreeqc_indices(df, ph, temp_c, database="phreeqc.dat"):
    """Run PHREEQC on each row; returns DataFrame with ionic strength, SI gypsum, SI calcite."""
    import phreeqc
    p = phreeqc.Phreeqc()
    p.LoadBuiltInDatabase(database)
    ph = np.broadcast_to(np.asarray(ph, dtype=float), (len(df),))
    temp_c = np.broadcast_to(np.asarray(temp_c, dtype=float), (len(df),))
    blocks = []
    for n, (i, row) in enumerate(df.reset_index(drop=True).iterrows(), start=1):
        blocks.append(
            f"SOLUTION {n}\n temp {temp_c[i]:.2f}\n pH {ph[i]:.3f}\n units mg/l\n"
            f" Ca {row['ca']:.4f}\n Mg {row['mg']:.4f}\n Na {row['na']:.4f}\n K {row.get('k', 0) or 0:.4f}\n"
            f" Cl {row['cl']:.4f}\n S(6) {row['so4']:.4f} as SO4\n"
            f" Alkalinity {row['hco3']:.4f} as HCO3\n Si {row.get('sio2', 0) or 0:.4f} as SiO2\n")
    p.RunString("".join(blocks) + "SELECTED_OUTPUT\n -reset false\n -ionic_strength true\n"
                " -si Gypsum Calcite\nEND\n")
    out = pd.DataFrame(p.GetSelectedOutput())
    return out.rename(columns={"mu": "ionic_strength", "si_Gypsum": "si_gypsum", "si_Calcite": "si_calcite"})


def fit_gypsum_calibration(df, ph, temp_c, n=3000, seed=0):
    """Fit the Davies->PHREEQC gypsum correction on a sample of compositions."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    ph_s, t_s = np.asarray(ph)[idx], np.asarray(temp_c)[idx]
    res = phreeqc_indices(sub, ph_s, t_s)
    ok = res["si_gypsum"].notna().to_numpy() & (res["si_gypsum"] > -50).to_numpy()
    ist = ionic_strength(sub)
    y = res["si_gypsum"].to_numpy() - np.log10(gypsum_sr_davies(sub, t_s))
    x = np.column_stack([np.ones(len(sub)), np.sqrt(ist), ist, t_s - 25.0])
    coef, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
    err = y[ok] - x[ok] @ coef
    cal = {"coef": coef.tolist(), "n": int(ok.sum()), "database": "phreeqc.dat",
           "rmse_log10_si": float(np.sqrt(np.mean(err ** 2))),
           "raw_davies_bias_log10": float(np.mean(y[ok])),
           "ionic_strength_range": [float(ist.min()), float(ist.max())]}
    GYPSUM_CAL_PATH.write_text(json.dumps(cal, indent=2))
    global _GYP_CAL
    _GYP_CAL = cal
    return cal
