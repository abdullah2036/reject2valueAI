"""Pipeline tests. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest
import yaml

from simulation import chemistry as chem
from simulation import ro_model as ro
from simulation.load_usgs import PROFILES

KFUPM_TOL = 0.15  # acceptance: simulated concentrate within ~10-15% of measured reject


def kfupm_validation():
    p = yaml.safe_load(PROFILES.read_text(encoding="utf-8"))["kfupm_dhahran_brackish"]
    feed = pd.DataFrame({"ca": [p["feed"]["ca"]], "so4": [p["feed"]["so4"]]})
    conc = ro.concentrate(feed, p["recovery"], ro.mean_rejection())
    return {ion: {"simulated": float(conc[ion][0]), "measured": p["reject_at_70pct_recovery"][ion],
                  "error_pct": float((conc[ion][0] / p["reject_at_70pct_recovery"][ion] - 1) * 100)}
            for ion in ["ca", "so4"]}


def test_kfupm_reject_validation():
    for ion, r in kfupm_validation().items():
        assert abs(r["error_pct"]) <= KFUPM_TOL * 100, (ion, r)


def test_mass_balance_closes():
    feed = pd.DataFrame({i: [100.0] for i in chem.IONS})
    rej = ro.mean_rejection()
    r = 0.75
    c, p = ro.concentrate(feed, r, rej), ro.permeate(feed, rej)
    # feed = R * permeate + (1 - R) * concentrate
    np.testing.assert_allclose(r * p.to_numpy() + (1 - r) * c.to_numpy(), feed.to_numpy(), rtol=1e-9)


def test_concentrate_ph_rises():
    assert ro.concentrate_ph(7.5, 200, 800) == pytest.approx(7.5 + np.log10(4))


def test_seawater_sar_and_lsi_sane():
    sw = pd.DataFrame({"ca": [480.0], "mg": [1497.0], "na": [12570.0], "k": [465.0], "cl": [22563.0],
                       "so4": [3162.0], "hco3": [146.0], "sio2": [1.0]})
    assert 40 < chem.sar(sw)[0] < 80          # seawater SAR ~ 57
    assert 0.6 < chem.ionic_strength(sw)[0] < 1.0
    assert chem.gypsum_sr(sw, 25)[0] < 1.0     # seawater is undersaturated in gypsum


def test_ec_model_monotonic():
    base = pd.DataFrame({"ca": [80.0], "mg": [30.0], "na": [150.0], "k": [5.0], "cl": [200.0],
                         "so4": [150.0], "hco3": [300.0], "sio2": [20.0]})
    ecs = [chem.predict_ec(base * f)[0] for f in [1, 2, 5, 10, 40, 80]]
    assert all(b > a for a, b in zip(ecs, ecs[1:]))
    assert 0.55 < chem.tds_from_ions(base)[0] / ecs[0] < 0.8
