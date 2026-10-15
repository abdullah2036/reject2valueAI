"""Load + clean the USGS Brackish Groundwater major-ions dataset and score "Saudi-likeness".

Source: Qi, S.L., and Harris, A.C., 2017, Geochemical Database for the Brackish Groundwater
Assessment of the United States, USGS data release, https://doi.org/10.5066/F72F7KK1
(ScienceBase item 58937228e4b0fa1e59b73361, "Major-ions dataset - comma delimited text file").

Run: python -m simulation.load_usgs   ->  data/processed/usgs_clean.parquet
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulation import chemistry as chem

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "usgs_clean.parquet"
PROFILES = ROOT / "data" / "reference" / "saudi_profiles.yaml"

COLMAP = {"Ca_mgL": "ca", "Mg_mgL": "mg", "Na_mgL": "na", "K_mgL": "k", "Cl_mgL": "cl",
          "SO4_mgL": "so4", "HCO3_mgl": "hco3_raw", "Alk_mgL": "alk_caco3", "CO3_mgL": "co3",
          "Si_mgL": "sio2", "Br_mgL": "br", "ph": "ph", "Temp_C": "temp_c", "SC_uScm": "ec_us_cm",
          "TDS_mgL": "tds_reported", "state_alpha": "state", "principal_aq": "aquifer",
          "sourceID": "source", "unique_site_ID": "site_id", "samp_start_dt": "sample_date",
          "model_si_Gypsum": "usgs_si_gypsum"}
FLAGS = ["Ca_mgL_flag", "Mg_mgL_flag", "Na_mgL_flag", "Cl_mgL_flag", "SO4_mgL_flag",
         "HCO3_mgl_flag", "Alk_mgL_flag"]


def find_raw_csv():
    csv = RAW / "usgs" / "Major_Ions.csv"
    if csv.exists():
        return csv
    zips = sorted(RAW.glob("Major_Ions*.zip"))
    if not zips:
        raise FileNotFoundError("Put 'Major_Ions_csv.zip' (USGS major-ions dataset) in data/raw/")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(RAW / "usgs")
    return csv


def load_raw():
    df = pd.read_csv(find_raw_csv(), usecols=list(COLMAP) + FLAGS, low_memory=False)
    print(f"raw rows: {len(df):,}; columns used: {list(COLMAP)}")
    return df


def clean(raw):
    n0 = len(raw)
    log = {"raw_rows": n0}
    # censored (< detection limit) or flagged-invalid major ions are dropped
    bad_flag = np.zeros(len(raw), bool)
    for f in FLAGS:
        s = raw[f].astype(str)
        bad_flag |= s.str.contains("<", regex=False) | s.str.contains("-4444", regex=False)
    df = raw.loc[~bad_flag].rename(columns=COLMAP).drop(columns=FLAGS)
    log["after_censored_flags"] = len(df)

    # bicarbonate: measured HCO3, else alkalinity as CaCO3 x 1.2193; CO3 folded in as HCO3-equivalent
    hco3 = df["hco3_raw"].where(df["hco3_raw"] > 0, df["alk_caco3"] * (61.017 / 50.04))
    df["hco3"] = hco3 + df["co3"].fillna(0).clip(lower=0) * (2 * 61.017 / 60.009)
    df = df.drop(columns=["hco3_raw", "alk_caco3", "co3"])

    req = ["ca", "mg", "na", "cl", "so4", "hco3", "ph"]
    df = df.dropna(subset=req)
    df = df[(df[req[:-1]] > 0).all(axis=1)]
    log["after_required_present"] = len(df)

    # physically plausible ranges
    df = df[df["ph"].between(4.0, 10.5)]
    df = df[df["temp_c"].isna() | df["temp_c"].between(0, 60)]
    df = df[df["ec_us_cm"].isna() | (df["ec_us_cm"] > 0)]
    df = df[(df["k"].isna()) | (df["k"] >= 0)]
    df = df[(df["sio2"].isna()) | df["sio2"].between(0, 300)]
    log["after_plausible_ranges"] = len(df)

    # K: impute from the median K/Na ratio when missing (small effect on charge balance)
    kna = (df["k"] / df["na"]).median()
    df["k_imputed"] = df["k"].isna()
    df["k"] = df["k"].fillna(df["na"] * kna)

    df["cbe_pct"] = chem.charge_balance_error(df)
    df = df[df["cbe_pct"].abs() <= 10]
    log["after_charge_balance_10pct"] = len(df)

    df["tds"] = chem.tds_from_ions(df)
    df = df[df["tds"].between(300, 15000)]
    log["after_tds_300_15000"] = len(df)

    # specific conductance must agree roughly with TDS (TDS/EC 0.4-1.0 is the physical range)
    ratio = df["tds"] / df["ec_us_cm"]
    df.loc[~ratio.between(0.4, 1.0), "ec_us_cm"] = np.nan

    df = df.drop_duplicates(subset=["site_id", "sample_date"])
    df = df.drop_duplicates(subset=["ca", "mg", "na", "cl", "so4", "hco3", "ph"])
    log["after_dedupe"] = len(df)

    # silica: impute missing with the state median (flagged; not used as an ML target)
    df["sio2_imputed"] = df["sio2"].isna()
    df["sio2"] = df["sio2"].fillna(df.groupby("state")["sio2"].transform("median")).fillna(df["sio2"].median())
    df["state"] = df["state"].fillna("NA")
    df["aquifer"] = df["aquifer"].fillna("unknown")
    log["final_rows"] = len(df)
    for k, v in log.items():
        print(f"  {k:32s} {v:>9,}")
    return df.reset_index(drop=True), log


def saudi_likeness(df):
    """Similarity (0-1) of each sample to the Saudi well-water signature in saudi_profiles.yaml:
    TDS range, ion dominance order, and Ca/Mg, SO4/Cl, Na/Cl mass ratios."""
    sig = yaml.safe_load(PROFILES.read_text(encoding="utf-8"))["saudi_signature"]
    lo, hi = sig["tds_range"]
    lt = np.log10(df["tds"].to_numpy())
    dist = np.maximum(np.log10(lo) - lt, 0) + np.maximum(lt - np.log10(hi), 0)
    tds_f = np.exp(-(dist / 0.15) ** 2)

    def order_score(order):
        pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
        return np.mean([(df[a] > df[b]).to_numpy() for a, b in pairs], axis=0)

    order = 0.5 * order_score(sig["cation_order"]) + 0.5 * order_score(sig["anion_order"])
    z2 = 0.0
    for name, spec in sig["ratios"].items():
        a, b = name.split("_")
        z2 = z2 + ((np.log10(df[a] / df[b]) - spec["centre"]) / spec["sigma"]) ** 2
    ratio_f = np.exp(-0.5 * z2 / len(sig["ratios"]))
    return tds_f * (0.5 * order + 0.5 * ratio_f)


def main():
    raw = load_raw()
    df, log = clean(raw)
    df["saudi_score"] = saudi_likeness(df)
    df["sample_weight"] = 0.2 + df["saudi_score"]
    df["saudi_top20"] = df["saudi_score"] >= df["saudi_score"].quantile(0.80)
    ec = chem.fit_ec_model(df)
    print(f"EC model: n={ec['n']:,} rmse(log10)={ec['rmse_log10']:.4f} "
          f"median |err|={ec['median_abs_pct_err']:.1f}% TDS range {ec['tds_min']:.0f}-{ec['tds_max']:.0f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    top = df[df["saudi_top20"]]
    summary = {"cleaning_log": log, "n_states": int(df["state"].nunique()),
               "n_aquifers": int(df["aquifer"].nunique()),
               "saudi_top20_n": int(len(top)),
               "saudi_top20_median": top[["tds", "ca", "mg", "na", "cl", "so4", "hco3"]].median().round(1).to_dict(),
               "all_median": df[["tds", "ca", "mg", "na", "cl", "so4", "hco3"]].median().round(1).to_dict()}
    (ROOT / "ml" / "models" / "usgs_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return df


if __name__ == "__main__":
    main()
