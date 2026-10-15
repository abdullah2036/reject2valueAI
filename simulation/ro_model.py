"""RO mass balance: turn a feed water analysis into permeate + concentrate (reject) chemistry.

Per ion: salt passage SP = 1 - rejection; permeate ~ C_feed * SP;
concentrate C_c = C_feed * (1 - R*SP) / (1 - R).
Concentrate pH: CO2 passes the membrane while HCO3 concentrates, so
pH_c ~ pH_feed + log10(C_c,HCO3 / C_feed,HCO3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import A
from simulation import chemistry as chem

MW_HCO3 = chem.MW["hco3"]
ION_GROUP = {"ca": "rejection_divalent", "mg": "rejection_divalent", "so4": "rejection_divalent",
             "na": "rejection_monovalent", "k": "rejection_monovalent", "cl": "rejection_monovalent",
             "hco3": "rejection_hco3", "sio2": "rejection_sio2"}


def mean_rejection():
    """Mid-range rejection per ion (used when the membrane state is unknown)."""
    return {ion: float(np.mean(A("simulation", grp))) for ion, grp in ION_GROUP.items()}


def sample_rejection(n, rng):
    """Sample one rejection per scenario per ion group (ions in a group share the draw)."""
    draws = {grp: rng.uniform(*A("simulation", grp), size=n) for grp in set(ION_GROUP.values())}
    return {ion: draws[grp] for ion, grp in ION_GROUP.items()}


def concentrate(feed, recovery, rejection):
    """Concentrate ion concentrations (mg/L). `rejection` maps ion -> scalar or array."""
    r = np.asarray(recovery, dtype=float)
    out = {}
    for ion in chem.IONS:
        sp = 1.0 - np.asarray(rejection[ion], dtype=float)
        out[ion] = chem._get(feed, ion) * (1.0 - r * sp) / (1.0 - r)
    return pd.DataFrame(out)


def permeate(feed, rejection):
    return pd.DataFrame({ion: chem._get(feed, ion) * (1.0 - np.asarray(rejection[ion], dtype=float))
                         for ion in chem.IONS})


def concentrate_ph(ph_feed, hco3_feed, hco3_conc):
    ratio = np.maximum(np.asarray(hco3_conc, float), 1e-6) / np.maximum(np.asarray(hco3_feed, float), 1e-6)
    return np.clip(np.asarray(ph_feed, float) + np.log10(ratio), 6.0, 9.0)


def feed_from_concentrate(conc, recovery, rejection=None):
    """Invert the mass balance: estimate the feed from a concentrate composition."""
    rejection = rejection or mean_rejection()
    r = np.asarray(recovery, dtype=float)
    return pd.DataFrame({ion: chem._get(conc, ion) * (1.0 - r) / (1.0 - r * (1.0 - rejection[ion]))
                         for ion in chem.IONS})


def acid_dose(feed, ph_feed, ph_target):
    """H2SO4 dosing of the feed down to ph_target (closed carbonate system, CO3 neglected).
    Returns (dosed feed, dosed pH, H2SO4 dose mg/L). HCO3 -> CO2 consumes acid; sulfate is added."""
    pk1 = A("simulation", "carbonate_pk1")
    ph_feed, ph_target = np.asarray(ph_feed, float), np.asarray(ph_target, float)
    ph_new = np.minimum(ph_feed, ph_target)
    hco3_mmol = chem._get(feed, "hco3") / MW_HCO3
    ct = hco3_mmol * (1 + 10 ** (pk1 - ph_feed))
    hco3_new = ct / (1 + 10 ** (pk1 - ph_new))
    acid_meq = np.maximum(hco3_mmol - hco3_new, 0)
    dosed = feed.copy() if isinstance(feed, pd.DataFrame) else pd.DataFrame(feed)
    dosed["hco3"] = hco3_new * MW_HCO3
    dosed["so4"] = chem._get(feed, "so4") + acid_meq * 96.06 / 2
    return dosed, ph_new, acid_meq * 98.08 / 2


def concentrate_state(feed, ph_feed, temp_c, recovery, rejection=None):
    """Concentrate composition + scaling indicators at a given recovery (vectorised)."""
    rejection = rejection or mean_rejection()
    c = concentrate(feed, recovery, rejection)
    c["ph"] = concentrate_ph(ph_feed, chem._get(feed, "hco3"), c["hco3"])
    c["tds"] = chem.tds_from_ions(c)
    c["lsi"], c["lsi_valid"] = chem.lsi(c, c["ph"], temp_c, c["tds"])
    c["gypsum_sr"] = chem.gypsum_sr(c, temp_c)
    return c
