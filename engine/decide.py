"""Run all paths, rank them, compute RVI and plain-language reasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import A
from engine.paths import dispose_path, recover_path, reduce_path, reuse_path
from engine.rvi import rvi
from simulation import chemistry as chem

PATHS = ["reduce", "reuse", "recover", "dispose"]
LABELS = {"reduce": "Reduce", "reuse": "Reuse", "recover": "Recover (Mg)", "dispose": "Treat/Dispose"}


def run_engine(s):
    """Vectorised engine over a DataFrame of reject compositions + operating data."""
    s = s.copy()
    s.index = pd.RangeIndex(len(s))
    out = pd.concat([reduce_path(s), reuse_path(s), recover_path(s), dispose_path(s)], axis=1)
    out["lsi"], out["lsi_valid"] = chem.lsi(s, s["reject_ph"], s["temp_c"])
    out["gypsum_sr"] = chem.gypsum_sr(s, s["temp_c"])
    out["sar"] = chem.sar(s)
    out["tds"] = chem.tds_from_ions(s)
    scores = np.column_stack([np.where(out[f"{p}_feasible"], out[f"{p}_score"], -1.0) for p in PATHS])
    out["recommended"] = np.array(PATHS, dtype=object)[scores.argmax(axis=1)]
    out["rvi"] = rvi(out["recover_score"], out["reuse_score"], out["reduce_score"], out["dispose_burden"])
    return out


def reasons(r):
    """Plain-language reasons for one engine row (a Series from run_engine)."""
    lim = A("scaling", "reduce_min_reduction")
    rs = {}
    if r["reduce_reduction"] > 0:
        rs["reduce"] = (f"Recovery can rise safely from {r['reduce_r_now']:.2f} to {r['reduce_r_max']:.2f}; "
                        f"reject volume drops {r['reduce_reduction'] * 100:.0f}% "
                        f"({r['reduce_water_saved_m3d']:.0f} m3/day saved). Limited by: {r['reduce_limiting']}.")
    else:
        rs["reduce"] = f"No safe headroom to raise recovery - {r['reduce_limiting']}."
    if r["reduce_reduction"] > 0 and r["reduce_reduction"] < lim:
        rs["reduce"] += f" Below the {lim * 100:.0f}% minimum worth acting on."
    share = r["reuse_max_reject_share"]
    rs["reuse"] = (f"Reject alone is FAO class '{r['reuse_class']}' (TDS {r['reuse_tds']:.0f} mg/L, SAR {r['reuse_sar']:.1f}). "
                   + (f"Blending {share * 100:.0f}% reject with {100 - share * 100:.0f}% permeate reaches "
                      f"'slight-moderate' irrigation class." if share > 0 else
                      "No blend with permeate reaches the 'slight-moderate' irrigation class."))
    rs["recover"] = (f"{r['recover_t_yr']:.1f} t/yr Mg(OH)2 at {'high' if r['recover_high_purity'] else 'low'} purity "
                     f"(price ~{r['recover_price_usd_t']:.0f} vs cost ~{r['recover_cost_usd_t']:.0f} USD/t at this scale): "
                     f"net {r['recover_net_sar_yr'] / 1e3:,.0f}k SAR/yr.")
    rs["dispose"] = (f"Salt load {r['dispose_salt_kg_d']:,.0f} kg/day; disposal ~{r['dispose_cost_sar_yr'] / 1e3:,.0f}k SAR/yr "
                     f"(placeholder rate, pending confirmation).")
    return rs


def rank(r):
    """Ranked list of paths for one engine row."""
    items = [{"path": LABELS[p], "key": p, "score": round(float(r[f"{p}_score"]), 1),
              "feasible": bool(r[f"{p}_feasible"])} for p in PATHS]
    rs = reasons(r)
    for it in items:
        it["reason"] = rs[it["key"]]
    return sorted(items, key=lambda d: (d["feasible"], d["score"]), reverse=True)
