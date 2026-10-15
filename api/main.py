"""REJECT2VALUE API (runs fully locally - no external calls).

Start:  uvicorn api.main:app --reload     then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from engine.decide import rank, run_engine
from ml.soft_sensor.predict import QUANTILE_TARGETS, TARGETS, SoftSensor
from simulation import chemistry as chem

ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD = ROOT / "ml" / "models" / "model_card.json"
STATE = {}
DEFAULT_LAB_SIO2 = {"brackish_well": 20.0, "seawater": 1.5}  # used only if the lab report has no silica

FEATURE_LABELS = {
    "m_reject_ec": "reject EC", "m_feed_ec": "feed EC", "m_reject_ph": "reject pH", "m_temp_c": "temperature",
    "m_recovery": "recovery", "is_seawater": "plant type", "cf": "concentration factor (from recovery)",
    "log_reject_ec": "reject EC", "log_feed_ec": "feed EC", "ec_ratio": "reject/feed EC ratio",
    "ec_ratio_over_cf": "EC ratio vs recovery consistency", "tds_fao_reject": "reject EC",
    "feed_drift": "feed EC drift since last lab", "lab_ph": "last lab pH",
    "est_dosed_feed_ph": "acid dosing (from reject pH)", "est_acid_mg_l": "acid dosing (from reject pH)",
}


@asynccontextmanager
async def lifespan(_app):
    STATE["sensor"] = SoftSensor()  # load models once
    yield
    STATE.clear()


app = FastAPI(title="REJECT2VALUE", version="0.1.0", lifespan=lifespan,
              description="Decide what to do with RO reject water: Reduce, Reuse, Recover (Mg) or Treat/Dispose. "
                          "Runs locally; plant data never leaves the machine.")


class FeedLab(BaseModel):
    ca: float = Field(..., gt=0, description="mg/L")
    mg: float = Field(..., gt=0)
    na: float = Field(..., gt=0)
    k: float = Field(0.0, ge=0)
    cl: float = Field(..., gt=0)
    so4: float = Field(..., gt=0)
    hco3: float = Field(..., gt=0)
    sio2: Optional[float] = Field(None, ge=0)
    ph: Optional[float] = Field(None, ge=4, le=11, description="Raw feed pH from the lab report (enables acid-dosing correction)")


class AnalyzeRequest(BaseModel):
    reject_ec: float = Field(..., gt=0, description="Reject (concentrate) conductivity, uS/cm at 25 C")
    reject_ph: float = Field(..., ge=4, le=11)
    temp_c: float = Field(..., ge=0, le=60)
    feed_ec: float = Field(..., gt=0, description="Feed conductivity, uS/cm at 25 C")
    recovery: Optional[float] = Field(None, gt=0.1, lt=0.98, description="Permeate / feed flow")
    feed_flow_m3d: Optional[float] = Field(None, gt=0)
    permeate_flow_m3d: Optional[float] = Field(None, gt=0)
    plant_type: Literal["brackish_well", "seawater"] = "brackish_well"
    reject_flow_m3d: Optional[float] = Field(None, gt=0)
    last_feed_lab: Optional[FeedLab] = Field(None, description="Last lab analysis of the FEED water (enables Mode B)")

    @model_validator(mode="after")
    def _flows(self):
        if self.recovery is None:
            if not (self.feed_flow_m3d and self.permeate_flow_m3d):
                raise ValueError("give `recovery` or both `feed_flow_m3d` and `permeate_flow_m3d`")
            self.recovery = self.permeate_flow_m3d / self.feed_flow_m3d
        if self.reject_flow_m3d is None:
            if self.feed_flow_m3d:
                self.reject_flow_m3d = self.feed_flow_m3d * (1 - self.recovery)
            else:
                raise ValueError("give `reject_flow_m3d` or `feed_flow_m3d`")
        return self


EXAMPLES = {
    "jeddah_bottled_water_well": {
        "summary": "Jeddah-area bottled-water plant (well water, acid-dosed, with last feed lab -> Mode B)",
        "value": {"reject_ec": 8195, "reject_ph": 7.49, "temp_c": 33, "feed_ec": 2339, "recovery": 0.75,
                  "plant_type": "brackish_well", "feed_flow_m3d": 300,
                  "last_feed_lab": {"ca": 150, "mg": 50, "na": 250, "k": 8, "cl": 450, "so4": 300,
                                    "hco3": 250, "sio2": 30, "ph": 7.5}}},
    "riyadh_brackish_plant": {
        "summary": "Riyadh brackish RO plant (Minjur/Wasia feed, sensors only -> Mode A)",
        "value": {"reject_ec": 17530, "reject_ph": 8.21, "temp_c": 36, "feed_ec": 3227,
                  "feed_flow_m3d": 1500, "permeate_flow_m3d": 1275, "plant_type": "brackish_well"}},
    "red_sea_swro": {
        "summary": "Red Sea seawater RO (with last feed lab -> Mode B)",
        "value": {"reject_ec": 90530, "reject_ph": 8.33, "temp_c": 30, "feed_ec": 54440, "recovery": 0.42,
                  "plant_type": "seawater", "feed_flow_m3d": 10000,
                  "last_feed_lab": {"ca": 480, "mg": 1497, "na": 12570, "k": 465, "cl": 22563, "so4": 3162,
                                    "hco3": 146, "sio2": 1.5, "ph": 8.1}}},
}


def _r(v, n=1):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def analyze_one(req: AnalyzeRequest):
    row = {"m_reject_ec": req.reject_ec, "m_reject_ph": req.reject_ph, "m_temp_c": req.temp_c,
           "m_feed_ec": req.feed_ec, "m_recovery": req.recovery, "plant_type": req.plant_type}
    if req.last_feed_lab:
        lab = req.last_feed_lab.model_dump()
        if lab["sio2"] is None:
            lab["sio2"] = DEFAULT_LAB_SIO2[req.plant_type]
        row.update({f"lab_{k}": v for k, v in lab.items()})
    x = pd.DataFrame([row])
    res = STATE["sensor"].predict(x, contributions=True)
    pred = res["pred"].iloc[0]
    comp = {t: float(pred[t]) for t in TARGETS}
    s = pd.DataFrame([{**comp, "reject_ph": req.reject_ph, "temp_c": req.temp_c, "recovery": req.recovery,
                       "reject_flow_m3d": req.reject_flow_m3d}])
    eng = run_engine(s).iloc[0]

    composition = {}
    for t in TARGETS:
        composition[t] = {"mg_l": _r(comp[t])}
        if t in QUANTILE_TARGETS:
            composition[t].update({"p10": _r(pred[f"{t}_p10"]), "p90": _r(pred[f"{t}_p90"])})

    contrib = res["contrib"]["mg"].iloc[0]
    agg = {}
    for f, v in contrib.items():
        name = FEATURE_LABELS.get(f) or (f"last lab {f.split('_')[1].upper()}" if f.startswith("lab_") else f)
        agg[name] = agg.get(name, 0.0) + float(v)
    top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:5]

    indicators = {"tds_mg_l": _r(eng["tds"], 0), "lsi": _r(eng["lsi"], 2), "lsi_formula_valid": bool(eng["lsi_valid"]),
                  "gypsum_saturation_ratio": _r(eng["gypsum_sr"], 2), "sar": _r(eng["sar"], 1),
                  "mg_oh2_potential_mg_l": _r(chem.mg_oh2_potential(s)[0], 0)}
    if chem.phreeqc_available():
        try:
            ph = chem.phreeqc_indices(s, req.reject_ph, req.temp_c).iloc[0]
            indicators["phreeqc"] = {"si_gypsum": _r(ph["si_gypsum"], 2), "si_calcite": _r(ph["si_calcite"], 2),
                                     "ionic_strength": _r(ph["ionic_strength"], 3)}
        except Exception:  # PHREEQC is optional; the calibrated Davies values above still apply
            pass

    warnings = []
    if not eng["lsi_valid"]:
        warnings.append("LSI formula is only valid for TDS < 10,000 mg/L; treat LSI as indicative.")
    if res["mode"] == "A":
        warnings.append("Mode A (sensors only): individual ions are uncertain; add the last feed lab analysis "
                        "for +/-15% accuracy (Mode B).")
    ranked = rank(eng)
    return {
        "mode": {"A": "A - sensors only", "B": "B - sensors + last feed lab"}[res["mode"]],
        "predicted_reject_composition": composition,
        "indicators": indicators,
        "rvi": _r(eng["rvi"]),
        "recommended_path": ranked[0]["path"],
        "ranked_paths": ranked,
        "impact": {"water_saved_m3_per_day": _r(eng["reduce_water_saved_m3d"]),
                   "reject_volume_reduction_pct": _r(eng["reduce_reduction"] * 100),
                   "mg_oh2_t_per_year": _r(eng["recover_t_yr"]),
                   "recover_net_sar_per_year": _r(eng["recover_net_sar_yr"], 0),
                   "disposal_cost_avoided_sar_per_year": _r(eng["reduce_water_saved_m3d"] * 365 *
                                                            eng["dispose_cost_sar_yr"] / max(req.reject_flow_m3d * 365, 1e-9), 0)},
        "top_contributing_inputs_for_mg": [{"input": k, "log_contribution": round(v, 3)} for k, v in top],
        "warnings": warnings,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest = Body(..., openapi_examples=EXAMPLES)):
    try:
        return analyze_one(req)
    except Exception as e:  # surface model/engine errors as 422 with the reason
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/model-card")
def model_card():
    if not MODEL_CARD.exists():
        raise HTTPException(404, "model_card.json not built yet - run scripts/run_full_pipeline.py")
    return json.loads(MODEL_CARD.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "sensor" in STATE, "phreeqc": chem.phreeqc_available()}
