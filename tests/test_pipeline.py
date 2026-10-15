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


def test_acid_dosing_lowers_hco3_and_adds_sulfate():
    feed = pd.DataFrame({"hco3": [250.0], "so4": [300.0]})
    dosed, ph, acid = ro.acid_dose(feed, [7.5], [6.9])
    assert dosed["hco3"][0] < 250 and dosed["so4"][0] > 300 and ph[0] == pytest.approx(6.9) and acid[0] > 0
    _, ph2, acid2 = ro.acid_dose(feed, [6.5], [6.9])  # already below target: no dosing
    assert ph2[0] == pytest.approx(6.5) and acid2[0] == 0


MODELS_READY = (chem.MODELS_DIR / "soft_sensor.joblib").exists()


@pytest.mark.skipif(not MODELS_READY, reason="run scripts/run_full_pipeline.py first")
def test_engine_seawater_recovers_mg():
    from engine.decide import run_engine
    sw = pd.DataFrame([{"ca": 830, "mg": 2580, "na": 21700, "k": 800, "cl": 38900, "so4": 5450, "hco3": 250,
                        "sio2": 2.6, "reject_ph": 8.3, "temp_c": 30, "recovery": 0.42, "reject_flow_m3d": 5800}])
    e = run_engine(sw).iloc[0]
    assert e["recommended"] == "recover" and e["recover_t_yr"] > 1000


@pytest.mark.skipif(not MODELS_READY, reason="run scripts/run_full_pipeline.py first")
def test_api_examples():
    from fastapi.testclient import TestClient

    from api.main import EXAMPLES, app
    with TestClient(app) as c:
        assert c.get("/health").json()["models_loaded"]
        for ex in EXAMPLES.values():
            r = c.post("/analyze", json=ex["value"])
            assert r.status_code == 200, r.text
            j = r.json()
            assert 0 <= j["rvi"] <= 100 and len(j["ranked_paths"]) == 4
            mg = j["predicted_reject_composition"]["mg"]
            assert mg["p10"] <= mg["mg_l"] <= mg["p90"]
