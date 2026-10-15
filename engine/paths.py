"""The four reject-management paths. Each returns score 0-100, feasibility and the numbers behind it.

All functions are vectorised over a DataFrame `s` with columns:
ca, mg, na, k, cl, so4, hco3, sio2 (reject, mg/L), reject_ph, temp_c, recovery, reject_flow_m3d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import A
from simulation import chemistry as chem
from simulation import ro_model as ro

LIMITS = ["lsi", "silica", "gypsum", "tds"]


def _limits_ok(c):
    return {"lsi": c["lsi"].to_numpy() <= A("scaling", "lsi_max_with_antiscalant"),
            "silica": c["sio2"].to_numpy() <= A("scaling", "silica_max_mg_l"),
            "gypsum": c["gypsum_sr"].to_numpy() <= A("scaling", "gypsum_saturation_max"),
            "tds": c["tds"].to_numpy() <= A("scaling", "ro_tds_max_mg_l")}


def reduce_path(s, step=0.01):
    """Highest recovery (current -> 0.95) at which the concentrate stays within all scaling limits.
    The feed is reconstructed from the reject by inverting the mass balance."""
    r0 = s["recovery"].to_numpy(dtype=float)
    feed = ro.feed_from_concentrate(s, r0)
    ph_feed = s["reject_ph"].to_numpy() - np.log10(np.maximum(s["hco3"].to_numpy(), 1e-6) /
                                                    np.maximum(feed["hco3"].to_numpy(), 1e-6))
    temp = s["temp_c"].to_numpy(dtype=float)
    r_max_search = A("scaling", "reduce_recovery_max")
    now = ro.concentrate_state(feed, ph_feed, temp, r0)
    ok_now = _limits_ok(now)
    ok_all = np.logical_and.reduce(list(ok_now.values()))
    limiting = np.where(ok_all, "", "current operation: " + _first_fail(ok_now))
    r_best = np.where(ok_all, r0, np.nan)
    alive = ok_all.copy()
    for k in range(1, int(round((r_max_search - 0.60) / step)) + 40):
        r = r0 + k * step
        active = alive & (r <= r_max_search + 1e-9)
        if not active.any():
            break
        c = ro.concentrate_state(feed[active], ph_feed[active], temp[active], r[active])
        oks = _limits_ok(c)
        ok = np.logical_and.reduce(list(oks.values()))
        idx = np.flatnonzero(active)
        r_best[idx[ok]] = r[idx[ok]]
        failed = idx[~ok]
        limiting[failed] = _first_fail({kk: v[~ok] for kk, v in oks.items()})
        alive[failed] = False
    limiting = np.where(alive, "search ceiling (R=0.95)", limiting)
    r_best = np.minimum(r_best, r_max_search)
    reduction = np.where(np.isfinite(r_best), 1 - (1 - r_best) / (1 - r0), 0.0)
    reduction = np.clip(reduction, 0, 1)
    score = 100 * np.clip(reduction / (2 * A("scaling", "reduce_target_reduction")), 0, 1)
    return pd.DataFrame({"reduce_r_now": r0, "reduce_r_max": r_best, "reduce_reduction": reduction, "reduce_limiting": limiting,
                         "reduce_feasible": reduction >= A("scaling", "reduce_min_reduction"),
                         "reduce_water_saved_m3d": s["reject_flow_m3d"].to_numpy() * reduction,
                         "reduce_score": score, "now_lsi": now["lsi"].to_numpy(),
                         "now_gypsum_sr": now["gypsum_sr"].to_numpy()}, index=s.index)


def _first_fail(oks):
    names = {"lsi": "LSI > 1.8 (CaCO3 scaling)", "silica": "silica > 150 mg/L",
             "gypsum": "gypsum saturation > 2.3", "tds": "TDS > 70,000 mg/L (RO pressure limit)"}
    out = np.full(len(next(iter(oks.values()))), "", dtype=object)
    for k in reversed(LIMITS):
        out = np.where(~oks[k], names[k], out)
    return out


def _fao_class(tds, sar_):
    t_lo, t_hi = A("reuse_irrigation", "tds_no_restriction"), A("reuse_irrigation", "tds_slight_moderate")
    s_lo, s_hi = A("reuse_irrigation", "sar_surface_low"), A("reuse_irrigation", "sar_surface_high")
    sev = np.maximum(np.select([tds <= t_lo, tds <= t_hi], [0, 1], 2), np.select([sar_ < s_lo, sar_ <= s_hi], [0, 1], 2))
    return np.array(["none", "slight-moderate", "severe"], dtype=object)[sev]


def reuse_path(s):
    """FAO-29 irrigation class of the reject, and the largest reject share in a blend with permeate
    that reaches the 'slight-moderate' class (TDS <= 2000 and SAR <= 9)."""
    r0 = s["recovery"].to_numpy(dtype=float)
    feed = ro.feed_from_concentrate(s, r0)
    perm = ro.permeate(feed, ro.mean_rejection())
    rej = s[chem.IONS].reset_index(drop=True)
    tds_r, sar_r = chem.tds_from_ions(rej), chem.sar(rej)
    f_grid = np.linspace(0, 1, 101)
    f_max = np.zeros(len(s))
    for f in f_grid:
        b = rej * f + perm * (1 - f)
        ok = (chem.tds_from_ions(b) <= A("reuse_irrigation", "tds_slight_moderate")) & \
             (chem.sar(b) <= A("reuse_irrigation", "sar_surface_high"))
        f_max = np.where(ok, f, f_max)
    feasible = f_max >= A("reuse_irrigation", "min_reject_share_in_blend")
    return pd.DataFrame({"reuse_class": _fao_class(tds_r, sar_r), "reuse_tds": tds_r, "reuse_sar": sar_r,
                         "reuse_max_reject_share": f_max, "reuse_feasible": feasible,
                         "reuse_score": 100 * f_max}, index=s.index)


def recover_path(s):
    """Mg(OH)2 recovery economics (NaOH precipitation at pH 11)."""
    q = s["reject_flow_m3d"].to_numpy(dtype=float)
    mg = s["mg"].to_numpy(dtype=float)
    t_yr = mg * q * 365 * A("recover_magnesium", "mg_oh2_per_mg") * A("recover_magnesium", "mg_recovery_efficiency") / 1e6
    high_purity = (s["ca"].to_numpy() / np.maximum(mg, 1e-9)) < A("recover_magnesium", "high_purity_max_ca_mg_ratio")
    price = np.where(high_purity, np.mean(A("recover_magnesium", "mg_oh2_price_usd_t_high_purity")),
                     np.mean(A("recover_magnesium", "mg_oh2_price_usd_t_low_purity")))
    base_cost = np.mean(A("recover_magnesium", "production_cost_eur_kg_industrial")) * 1000 * A("economics", "eur_to_usd")
    big, small = A("recover_magnesium", "industrial_scale_t_yr"), A("recover_magnesium", "small_scale_t_yr")
    frac = np.clip(np.log(big / np.maximum(t_yr, 1e-9)) / np.log(big / small), 0, 1)
    cost = base_cost * (1 + (A("recover_magnesium", "small_scale_cost_multiplier") - 1) * frac)
    margin = (price - cost) / price
    net_usd = (price - cost) * t_yr
    feasible = (net_usd > 0) & (t_yr >= A("recover_magnesium", "min_production_t_yr"))
    score = np.where(feasible, np.clip(50 + 100 * margin, 0, 100), np.clip(30 + 30 * margin, 0, 30))
    fx = A("economics", "usd_to_sar")
    return pd.DataFrame({"recover_t_yr": t_yr, "recover_high_purity": high_purity, "recover_price_usd_t": price,
                         "recover_cost_usd_t": cost, "recover_net_sar_yr": net_usd * fx,
                         "recover_revenue_sar_yr": price * t_yr * fx, "recover_feasible": feasible,
                         "recover_score": score}, index=s.index)


def dispose_path(s):
    """Fallback: treat/dispose. Environmental burden from salt load (TDS x volume)."""
    q = s["reject_flow_m3d"].to_numpy(dtype=float)
    tds = chem.tds_from_ions(s)
    salt_kg_d = tds * q / 1000
    burden = 100 * np.clip(np.log10(np.maximum(salt_kg_d, 1)) / np.log10(A("dispose", "burden_ref_salt_kg_d")), 0, 1)
    return pd.DataFrame({"dispose_salt_kg_d": salt_kg_d, "dispose_burden": burden,
                         "dispose_cost_sar_yr": q * 365 * A("economics", "disposal_cost_sar_m3"),
                         "dispose_feasible": True, "dispose_score": float(A("dispose", "fallback_score"))},
                        index=s.index)
