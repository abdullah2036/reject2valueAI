"""Generate the simulated RO reject dataset from real feed waters.

Brackish/well feeds: cleaned USGS analyses (1-3 scenarios each).
Seawater feeds: Saudi seawater reference profiles with natural variability.
Each scenario: recovery, membrane rejection, temperature, plant size -> concentrate chemistry,
true + noisy sensor readings, a stale "last lab" feed analysis, scaling/reuse indicators.

Run: python -m simulation.generate_dataset  ->  data/processed/reject_dataset.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine import A
from simulation import chemistry as chem
from simulation import ro_model as ro

ROOT = Path(__file__).resolve().parents[1]
USGS = ROOT / "data" / "processed" / "usgs_clean.parquet"
OUT = ROOT / "data" / "processed" / "reject_dataset.parquet"
PROFILES = ROOT / "data" / "reference" / "saudi_profiles.yaml"
MAX_ROWS = 150_000
SW_PROFILES = {"SW_REDSEA": "red_sea_standard_ratio_seawater", "SW_GULF": "arabian_gulf_kuwait_seawater"}


def brackish_feeds(rng):
    usgs = pd.read_parquet(USGS)
    n_scen = rng.integers(1, 4, size=len(usgs))
    feeds = usgs.loc[usgs.index.repeat(n_scen)].reset_index(drop=True)
    feeds["feed_id"] = usgs.index.repeat(n_scen)
    feeds["plant_type"] = "brackish_well"
    feeds["group"] = feeds["state"]
    t = feeds["temp_c"]
    lo, hi = A("simulation", "temp_brackish_c")
    feeds["temp_c"] = np.where(t.between(lo, hi), t, rng.uniform(lo, hi, len(feeds)))
    return feeds


def seawater_feeds(n, rng):
    prof = yaml.safe_load(PROFILES.read_text(encoding="utf-8"))
    rows = []
    groups = rng.choice(list(SW_PROFILES), size=n)
    for g in SW_PROFILES:
        m = int((groups == g).sum())
        p = prof[SW_PROFILES[g]]
        scale = rng.uniform(0.93, 1.07, m)  # seasonal / location salinity variation
        d = {ion: p[ion] * scale * rng.lognormal(0, 0.03, m) for ion in ["ca", "mg", "na", "k", "cl", "so4"]}
        d["hco3"] = p["hco3"] * rng.lognormal(0, 0.08, m)
        d["br"] = p.get("br", 0) * scale
        d["sio2"] = rng.uniform(0.5, 3.0, m)
        d["ph"] = rng.uniform(7.9, 8.3, m)
        d["temp_c"] = rng.uniform(*A("simulation", "temp_seawater_c"), m)
        df = pd.DataFrame(d)
        df["group"], df["state"], df["aquifer"], df["source"] = g, "SA", "seawater", SW_PROFILES[g]
        rows.append(df)
    sw = pd.concat(rows, ignore_index=True)
    sw["plant_type"] = "seawater"
    sw["ec_us_cm"] = np.nan
    sw["feed_id"] = -1 - np.arange(len(sw))
    sw["saudi_score"], sw["sample_weight"], sw["saudi_top20"] = 1.0, 1.0, False
    sw["sio2_imputed"] = True
    sw["tds"] = chem.tds_from_ions(sw)
    return sw


def simulate(feeds, rng):
    n = len(feeds)
    sw = (feeds["plant_type"] == "seawater").to_numpy()
    # recovery: brackish capped so the concentrate stays under the RO salinity limit
    r_lo, r_hi = A("simulation", "recovery_brackish")
    cap = np.clip(1.0 - feeds["tds"].to_numpy() / A("scaling", "ro_tds_max_mg_l"), r_lo + 0.01, r_hi)
    rec = rng.uniform(r_lo, cap)
    rec[sw] = rng.uniform(*A("simulation", "recovery_seawater"), sw.sum())
    rej = ro.sample_rejection(n, rng)

    conc = ro.concentrate(feeds, rec, rej)
    perm = ro.permeate(feeds, rej)
    out = pd.DataFrame({"feed_id": feeds["feed_id"], "group": feeds["group"], "state": feeds["state"],
                        "aquifer": feeds["aquifer"], "plant_type": feeds["plant_type"],
                        "saudi_score": feeds["saudi_score"], "sample_weight": feeds["sample_weight"],
                        "saudi_top20": feeds["saudi_top20"].astype(bool),
                        "sio2_imputed": feeds["sio2_imputed"].astype(bool),
                        "recovery": rec, "temp_c": feeds["temp_c"].to_numpy(),
                        "feed_ph": feeds["ph"].to_numpy(), "feed_tds": feeds["tds"].to_numpy()})
    for ion in chem.IONS:
        out[f"feed_{ion}"] = chem._get(feeds, ion)
        out[ion] = conc[ion].to_numpy()
        out[f"perm_{ion}"] = perm[ion].to_numpy()
    out["rej_mono"], out["rej_di"] = rej["na"], rej["ca"]
    out["reject_ph"] = np.clip(ro.concentrate_ph(feeds["ph"], chem._get(feeds, "hco3"), conc["hco3"])
                               + rng.normal(0, 0.05, n), 6.0, 9.0)

    # conductivity: real measured feed EC where available; reject EC from the fitted EC model,
    # carrying over each water's own deviation from the model (composition-specific conductivity)
    ec_feed_model = chem.predict_ec(feeds)
    ec_feed = feeds["ec_us_cm"].to_numpy(dtype=float)
    ec_feed = np.where(np.isfinite(ec_feed), ec_feed, ec_feed_model)
    anomaly = np.clip(ec_feed / ec_feed_model, 0.8, 1.25)
    out["feed_ec"] = ec_feed
    out["reject_ec"] = chem.predict_ec(conc) * anomaly
    out["tds"] = chem.tds_from_ions(conc)

    # plant size and flows
    lo, hi = np.where(sw, A("simulation", "plant_feed_m3d_seawater")[0], A("simulation", "plant_feed_m3d_brackish")[0]), \
        np.where(sw, A("simulation", "plant_feed_m3d_seawater")[1], A("simulation", "plant_feed_m3d_brackish")[1])
    out["feed_flow_m3d"] = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    out["reject_flow_m3d"] = out["feed_flow_m3d"] * (1 - rec)
    return out


def add_sensors(df, rng):
    """Noisy, drifting, partly missing sensor readings (what the plant actually logs)."""
    sn = A("simulation", "sensor_noise")
    n = len(df)
    drift = np.where(rng.random(n) < sn["drift_frac_rows"],
                     rng.uniform(-sn["drift_max_rel"], sn["drift_max_rel"], n), 0.0)
    df["m_reject_ec"] = df["reject_ec"] * (1 + rng.normal(0, sn["ec_rel"], n)) * (1 + drift)
    df["m_feed_ec"] = df["feed_ec"] * (1 + rng.normal(0, sn["ec_rel"], n))
    df["m_reject_ph"] = df["reject_ph"] + rng.normal(0, sn["ph_abs"], n)
    df["m_temp_c"] = df["temp_c"] + rng.normal(0, sn["temp_abs"], n)
    m_feed = df["feed_flow_m3d"] * (1 + rng.normal(0, sn["flow_rel"], n))
    m_perm = df["feed_flow_m3d"] * df["recovery"] * (1 + rng.normal(0, sn["flow_rel"], n))
    df["m_recovery"] = np.clip(m_perm / m_feed, 0.2, 0.97)
    df["m_reject_flow_m3d"] = df["reject_flow_m3d"] * (1 + rng.normal(0, sn["flow_rel"], n))
    for c in ["m_reject_ec", "m_feed_ec", "m_reject_ph", "m_temp_c", "m_recovery"]:
        df.loc[rng.random(n) < sn["missing_frac"], c] = np.nan
    # stale "last lab" feed analysis: true feed perturbed by 0-15% (taken weeks ago)
    d = rng.uniform(0, 0.15, n)
    for ion in chem.IONS:
        df[f"lab_{ion}"] = df[f"feed_{ion}"] * (1 + rng.uniform(-1, 1, n) * d)
    return df


def add_indicators(df):
    conc = df[chem.IONS]
    df["lsi"], df["lsi_valid"] = chem.lsi(conc, df["reject_ph"], df["temp_c"], df["tds"])
    df["gypsum_sr"] = chem.gypsum_sr(conc, df["temp_c"])
    df["sar"] = chem.sar(conc)
    df["mg_oh2_mg_l"] = chem.mg_oh2_potential(conc)
    return df


def phreeqc_checks(df, usgs, rng):
    """Calibrate Davies gypsum SR against PHREEQC on simulated concentrates, and check our
    PHREEQC setup against the USGS-published PHREEQC gypsum SI on the raw feeds."""
    if not chem.phreeqc_available():
        return {"phreeqc": "not available - Davies-only gypsum saturation (uncalibrated)"}
    cal = chem.fit_gypsum_calibration(df[chem.IONS], df["reject_ph"].to_numpy(), df["temp_c"].to_numpy())
    sub = usgs.dropna(subset=["usgs_si_gypsum"]).sample(500, random_state=1)
    ours = chem.phreeqc_indices(sub[chem.IONS], sub["ph"].to_numpy(),
                                sub["temp_c"].fillna(25).to_numpy())["si_gypsum"].to_numpy()
    diff = ours - sub["usgs_si_gypsum"].to_numpy()
    return {"phreeqc": "phreeqc.dat via `phreeqc` Python package",
            "gypsum_calibration": cal,
            "check_vs_usgs_published_si_gypsum": {"n": 500, "mean_diff": float(np.mean(diff)),
                                                  "mae": float(np.mean(np.abs(diff)))}}


def main(seed=42):
    rng = np.random.default_rng(seed)
    br = brackish_feeds(rng)
    n_sw = int(len(br) * A("simulation", "seawater_share_of_dataset") / (1 - A("simulation", "seawater_share_of_dataset")))
    feeds = pd.concat([br, seawater_feeds(n_sw, rng)], ignore_index=True)
    if len(feeds) > MAX_ROWS:
        feeds = feeds.sample(MAX_ROWS, random_state=seed).reset_index(drop=True)
    df = simulate(feeds, rng)
    checks = phreeqc_checks(df, pd.read_parquet(USGS), rng)
    df = add_indicators(add_sensors(df, rng))
    df.to_parquet(OUT, index=False)
    summary = {"rows": len(df), "by_plant_type": df["plant_type"].value_counts().to_dict(),
               "unique_real_feeds": int((df["feed_id"] >= 0).sum() and df.loc[df["feed_id"] >= 0, "feed_id"].nunique()),
               "reject_tds_p10_p50_p90": df["tds"].quantile([.1, .5, .9]).round(0).tolist(),
               **checks}
    (ROOT / "ml" / "models" / "simulation_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return df


if __name__ == "__main__":
    main()
