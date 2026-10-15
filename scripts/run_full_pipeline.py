"""Run the whole REJECT2VALUE pipeline and write ml/models/model_card.json.

    python scripts/run_full_pipeline.py              # all steps
    python scripts/run_full_pipeline.py --from train # resume from a step (load, simulate, train, thresholds, card)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import confirm_items, load_assumptions  # noqa: E402
from simulation import chemistry as chem  # noqa: E402

MODELS = ROOT / "ml" / "models"
STEPS = ["load", "simulate", "train", "thresholds", "card"]


def _j(name):
    p = MODELS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def metrics_table(metrics):
    rows = []
    for mode, res in metrics.get("modes", {}).items():
        for sub in ["overall", "saudi_like_top20", "seawater"]:
            for t, m in res.get(sub, {}).items():
                rows.append({"mode": mode, "subset": sub, "target": t, "n": m["n"],
                             "within_15pct": round(m["within_15pct"], 1), "within_10pct": round(m["within_10pct"], 1),
                             "mape_pct": round(m["mape_pct"], 1), "median_ape_pct": round(m["median_ape_pct"], 1),
                             "r2_log": round(m["r2_log"], 3),
                             "baseline_within_15pct": round(m["baseline_within_15pct"], 1),
                             **({"p10_p90_coverage_pct": round(m["p10_p90_coverage_pct"], 1)}
                                if "p10_p90_coverage_pct" in m else {})})
    return rows


def build_model_card():
    from tests.test_pipeline import kfupm_validation
    usgs, sim, metrics, thr = _j("usgs_summary.json"), _j("simulation_summary.json"), _j("metrics.json"), _j("thresholds.json")
    profiles = yaml.safe_load((ROOT / "data" / "reference" / "saudi_profiles.yaml").read_text(encoding="utf-8"))
    kf = kfupm_validation()
    confirm = [{"item": f"assumptions.{c['item']}", "value": c["value"], "source": c["source"],
                "status": "pending chemist confirmation"} for c in confirm_items()]
    for k, v in profiles.items():
        if isinstance(v, dict) and v.get("status") == "CONFIRM":
            confirm.append({"item": f"saudi_profiles.{k}", "value": None, "source": v.get("source"),
                            "note": v.get("note", ""), "status": "pending chemist confirmation"})
    agree = thr.get("recommendation_agreement", {})
    card = {
        "name": "REJECT2VALUE soft sensor + decision engine",
        "version": "0.1.0", "built": dt.date.today().isoformat(),
        "intended_use": "Decision support for RO plant operators: estimate reject (concentrate) chemistry from online "
                        "sensors (+ the last feed lab analysis) and rank Reduce / Reuse / Recover (Mg) / Treat-Dispose. "
                        "Runs fully locally (no external APIs). Not a substitute for lab analysis before capital decisions.",
        "data_sources": [
            {"name": "USGS Brackish Groundwater major-ions dataset",
             "citation": "Qi, S.L., and Harris, A.C., 2017, Geochemical Database for the Brackish Groundwater Assessment "
                         "of the United States, U.S. Geological Survey data release",
             "doi": "https://doi.org/10.5066/F72F7KK1",
             "url": "https://www.sciencebase.gov/catalog/item/58937228e4b0fa1e59b73361"},
            *[{"name": k, "citation": v.get("source")} for k, v in profiles.items() if isinstance(v, dict)],
        ],
        "real_samples": {"raw_analyses": usgs.get("cleaning_log", {}).get("raw_rows"),
                         "after_cleaning": usgs.get("cleaning_log", {}).get("final_rows"),
                         "cleaning_log": usgs.get("cleaning_log"), "states": usgs.get("n_states"),
                         "aquifers": usgs.get("n_aquifers"), "saudi_like_top20_samples": usgs.get("saudi_top20_n"),
                         "saudi_weighting": "sample_weight = 0.05 + saudi_score^2; saudi_score from TDS range, ion "
                                            "dominance order and Ca/Mg, SO4/Cl, Na/Cl ratios (saudi_profiles.yaml)"},
        "simulation": {"rows": sim.get("rows"), "by_plant_type": sim.get("by_plant_type"),
                       "assumptions": load_assumptions()["simulation"],
                       "ec_model": _j("ec_model.json"),
                       "chemistry_engine": sim.get("phreeqc"),
                       "gypsum_calibration_vs_phreeqc": sim.get("gypsum_calibration"),
                       "phreeqc_check_vs_usgs_published_si": sim.get("check_vs_usgs_published_si_gypsum")},
        "validation_kfupm": {"case": "KFUPM Dhahran brackish RO, feed Ca 281.2 / SO4 611 mg/L at R = 0.70 "
                                     "(US patent 9481590)", "result": kf,
                             "passes_15pct": all(abs(v["error_pct"]) <= 15 for v in kf.values())},
        "soft_sensor": {"model": "LightGBM per ion on log-concentration, residual over a physics prior "
                                 "(Mode A: reject EC; Mode B: last-lab feed ion x concentration factor); "
                                 "conformalised quantile models (P10/P90) for Mg and Ca",
                        "split": {"type": "grouped holdout - whole US states + Arabian Gulf seawater never seen in training",
                                  "test_groups": metrics.get("test_groups"), "n_train": metrics.get("n_train"),
                                  "n_test": metrics.get("n_test")},
                        "acceptance": "within +/-15% (minimum), +/-10% (target)",
                        "metrics_table": metrics_table(metrics),
                        "top_features_mg": {m: r.get("top_features_mg") for m, r in metrics.get("modes", {}).items()}},
        "recommendation_agreement": {"definition": "engine recommendation from predicted composition + measured sensors "
                                                   "vs from true composition + true operation (held-out test set)",
                                     "acceptance": ">=90% minimum, 95% target", **agree},
        "discovered_thresholds": {"rules": thr.get("decision_tree", {}).get("rules"),
                                  "tree_fidelity_pct": thr.get("decision_tree", {}).get("fidelity_to_engine_pct"),
                                  "archetypes": [{k: a[k] for k in ["name", "share_pct", "tds_p50", "mg_p50", "top_path",
                                                                   "path_mix_pct"]}
                                                 for a in thr.get("clusters", {}).get("archetypes", [])],
                                  "path_quantiles": thr.get("path_quantiles"),
                                  "path_mix_pct": thr.get("path_mix_pct")},
        "known_limitations": [
            "Trained on US groundwater (USGS) re-weighted toward Saudi chemistry and passed through an RO simulator - "
            "no real Saudi reject samples yet. Next step: fine-tune on pilot-plant samples.",
            "Mode A (sensors only) cannot determine individual ions from conductivity alone; it is a screening mode. "
            "Mode B (with last feed lab) is the mode that meets +/-15%.",
            "Simulated 'last lab' staleness is 0-15% random drift; real feed changes (new wells, seasons) may be larger.",
            "Acid dosing is simulated for 60% of brackish plants (H2SO4 to pH 6.5-7.2); antiscalant chemistry is "
            "represented only through the limits in assumptions.yaml.",
            "LSI (Langelier) is only valid below 10,000 mg/L TDS; for seawater concentrate it is indicative only "
            "(Stiff-Davis would be more appropriate).",
            "Gypsum saturation uses Davies activities with a PHREEQC (phreeqc.dat) calibration; phreeqc.dat is "
            "itself approximate above ionic strength ~1 (seawater concentrate); Pitzer would be more accurate.",
            "P10-P90 bands are conformally calibrated on held-out US states; for seawater the Mg band under-covers "
            "(no seawater calibration group available).",
            "Conductivity beyond the fitted range (TDS > ~8,700 mg/L) is extrapolated with the FAO slope.",
            "Mg recovery economics use literature price/cost ranges and an assumed small-scale multiplier; no "
            "reagent-price sensitivity or site CAPEX is included.",
            "Reuse considers FAO TDS and SAR classes for surface irrigation only (no boron, chloride toxicity, crop "
            "tolerance or SAR x ECw infiltration interaction).",
        ],
        "confirm_items": confirm,
    }
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STEPS, default="load")
    args = ap.parse_args()
    t0 = time.time()
    todo = STEPS[STEPS.index(args.start):]
    if "load" in todo:
        print("== Phase 1: load + clean USGS data"), __import__("simulation.load_usgs", fromlist=["main"]).main()
    if "simulate" in todo:
        print("== Phase 2: RO simulation"), __import__("simulation.generate_dataset", fromlist=["main"]).main()
    if "train" in todo:
        print("== Phase 3: soft sensor"), __import__("ml.soft_sensor.train", fromlist=["main"]).main()
    if "thresholds" in todo:
        print("== Phase 4: engine thresholds + agreement"), __import__("ml.thresholds.discover", fromlist=["main"]).main()
    print("== Model card")
    card = build_model_card()
    print(f"KFUPM: {json.dumps(card['validation_kfupm']['result'])}")
    print(f"done in {time.time() - t0:.0f}s; PHREEQC available: {chem.phreeqc_available()}")


if __name__ == "__main__":
    main()
