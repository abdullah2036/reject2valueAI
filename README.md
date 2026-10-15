# REJECT2VALUE — AI core

Decision support for RO (reverse osmosis) plant operators: **what to do with your reject water (brine)** —
**Reduce, Reuse, Recover (magnesium), or Treat/Dispose**.
Built for Miyahthon (Saudi Water Innovation Center), track: Reuse & Circular Economy.

Everything runs **locally**. There are no external AI/LLM APIs and no cloud calls at runtime, because plant data is confidential.

## What it does

1. **Soft sensor (the AI):** estimates the reject's chemistry (Mg, Ca, SO4, HCO3, Na, Cl, K, SiO2) from what plants
   measure cheaply and continuously:
   - **Mode A — sensors only:** reject EC, reject pH, temperature, feed EC, recovery, plant type. Screening only.
   - **Mode B — sensors + the last feed lab analysis** (which can be weeks old): meets the ±15% target.
   It returns P10–P90 ranges for Mg and Ca and the inputs that drove each prediction.
2. **Decision engine:** scores the four paths (0–100) with plain-language reasons, computes the **Reject Value Index
   (RVI)** and the impact (m³/day saved, t/yr Mg(OH)2, SAR/yr). Every limit and price is in
   [engine/assumptions.yaml](engine/assumptions.yaml) with its source.
3. **Data-discovered thresholds:** clusters reject archetypes and distils the engine into slide-ready rules.

## Results (held-out US states + Arabian Gulf seawater never seen in training)

% of predictions within ±15% (acceptance: ±15% minimum, ±10% target):

| Mode | Subset | Mg | Ca | SO4 | HCO3 | Na | Cl |
|---|---|---|---|---|---|---|---|
| A (sensors only) | overall | 29 | 32 | 29 | 36 | 31 | 28 |
| A | Saudi-like top 20% | 12 | 17 | 17 | 20 | 30 | 6 |
| A | seawater | 98 | 99 | 94 | 80 | 75 | 99 |
| **B (sensors + last lab)** | overall | **98** | **98** | 86 | 96 | 98 | 98 |
| **B** | Saudi-like top 20% | **98** | **98** | 90 | 95 | 98 | 98 |
| **B** | seawater | **100** | **100** | 100 | 99 | 97 | 100 |

Naive-physics baseline (last lab × concentration factor), Mode B, Saudi-like subset: Mg 71%, Ca 72%, SO4 52%.

- **Recommendation agreement** (engine on predicted vs true composition): **Mode B 97.6%** overall, 96.5% on
  Saudi-like water (target 95%). Mode A: 76.2%.
- **KFUPM validation** (Dhahran brackish RO, R = 0.70): simulated reject Ca **+7.5%**, SO4 **−3.6%** vs measured.
- Full numbers: `GET /model-card` or [ml/models/model_card.json](ml/models/model_card.json). Parity plots:
  [ml/models/figures](ml/models/figures).

## How this was trained

Plants don't publish reject data. So we screened 124k real groundwater analyses (USGS) and kept the 51.8k that pass
charge-balance and salinity QC. We weighted them toward Saudi aquifer chemistry and passed them through an RO
mass-balance simulator, validated against a KFUPM plant measurement. That gave 115k reject scenarios: brackish wells
plus Red Sea / Gulf seawater, with acid dosing, sensor noise, drift and stale lab results. Scaling chemistry is
checked against PHREEQC. Next step: fine-tune on real samples from a pilot plant.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Download the USGS **"Major-ions dataset - comma delimited text file format"** zip (~54 MB) from
https://www.sciencebase.gov/catalog/item/58937228e4b0fa1e59b73361 (Qi & Harris 2017,
https://doi.org/10.5066/F72F7KK1) and put it in `data/raw/`.

```bash
python scripts/run_full_pipeline.py            # ~4-5 min on a laptop: clean -> simulate -> train -> thresholds -> model card
python scripts/run_full_pipeline.py --from train   # resume from a step: load | simulate | train | thresholds | card
pytest -q
```

Start the API, then open http://127.0.0.1:8000/docs. It has three ready-made examples: a Jeddah bottled-water well,
a Riyadh brackish plant, and a Red Sea SWRO plant.

```bash
uvicorn api.main:app --reload
```

| Endpoint | |
|---|---|
| `POST /analyze` | sensors (+ optional `last_feed_lab`) → predicted composition with P10–P90, LSI / gypsum / SAR (+ PHREEQC SI), RVI, ranked paths with reasons, impact, top contributing inputs |
| `GET /model-card` | data sources, metrics, validation, limitations, items pending chemist confirmation |
| `GET /health` | status |

## Layout

```
data/reference/saudi_profiles.yaml   Saudi water profiles (with sources)
simulation/   load_usgs.py, chemistry.py (TDS/EC, LSI, gypsum, SAR, PHREEQC), ro_model.py, generate_dataset.py
ml/           soft_sensor/ (train, predict), thresholds/discover.py, common/evaluation.py, models/ (card, metrics, figures)
engine/       assumptions.yaml, paths.py, rvi.py, decide.py
api/main.py   FastAPI
```

## Pending chemist confirmation

Every item flagged `confirm: true` in [engine/assumptions.yaml](engine/assumptions.yaml) and `CONFIRM` in
[saudi_profiles.yaml](data/reference/saudi_profiles.yaml) is listed under `confirm_items` in the model card. Among
them: membrane rejections, the gypsum limit (2.3), the disposal cost (placeholder 2 SAR/m³), the small-scale Mg cost
multiplier, acid-dosing practice, and the Lenntech Red Sea Mg/Ca values.
