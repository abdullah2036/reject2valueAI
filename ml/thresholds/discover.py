"""Let the data suggest structure and slide-ready thresholds (limits themselves stay from literature).

1. Run the engine over the simulated dataset.
2. Cluster reject compositions into archetypes (KMeans, k by silhouette, k <= 6; GMM BIC reported).
3. Distill the engine into a shallow decision tree -> human-readable rules.
4. P10/P50/P90 of key variables per recommended path.
5. Recommendation agreement: engine on soft-sensor-predicted vs true composition (held-out test set).

Run: python -m ml.thresholds.discover  ->  ml/models/thresholds.json (+ figures)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from engine.decide import LABELS, run_engine
from ml.soft_sensor.predict import TARGETS
from simulation import chemistry as chem

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "reject_dataset.parquet"
TEST_PRED = ROOT / "data" / "processed" / "test_predictions.parquet"
OUT = ROOT / "ml" / "models" / "thresholds.json"
FIG = ROOT / "ml" / "models" / "figures"
TREE_VARS = ["tds", "mg", "ca", "so4", "sio2", "sar", "lsi", "gypsum_sr", "reject_flow_m3d",
             "reject_ph", "temp_c"]
N_ENGINE = 40_000


def cluster_features(d):
    return pd.DataFrame({"log_tds": np.log10(d["tds"]), "log_mg_na": np.log10(d["mg"] / d["na"]),
                         "log_so4_cl": np.log10(d["so4"] / d["cl"]), "log_ca_cl": np.log10(d["ca"] / d["cl"]),
                         "log_hco3_cl": np.log10(d["hco3"] / d["cl"])})


def name_cluster(c):
    tds = c["tds_p50"]
    sal = ("low-salinity" if tds < 3000 else "brackish" if tds < 15000 else
           "high-salinity" if tds < 50000 else "seawater-concentrate")
    traits = []
    if c["gypsum_sr_p50"] > 1.0:
        traits.append("sulfate-rich, gypsum-scaling-prone")
    if c["mg_p50"] > 1000:
        traits.append("high-Mg recoverable")
    if c["lsi_p50"] > 1.8 and tds < 15000:
        traits.append("carbonate-scaling-limited")
    if c["hco3_cl_p50"] > 1.5:
        traits.append("bicarbonate-type")
    if not traits:
        traits.append("reusable" if c["top_path"] == "reuse" else "chloride-type")
    return f"{sal} {', '.join(traits)}"


def clusters(d, eng, seed=0):
    x = StandardScaler().fit_transform(cluster_features(d))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(x), size=min(8000, len(x)), replace=False)
    sil, bic = {}, {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(x[sub])
        sil[k] = float(silhouette_score(x[sub], km.labels_))
        bic[k] = float(GaussianMixture(k, random_state=seed).fit(x[sub]).bic(x[sub]))
    k = max(sil, key=sil.get)
    if k == 2:  # two clusters only separates seawater from groundwater; prefer the next best for archetypes
        k = max((kk for kk in sil if kk > 2), key=sil.get)
    lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(x).labels_
    arche = []
    for c in range(k):
        m = lab == c
        st = {"cluster": c, "n": int(m.sum()), "share_pct": round(float(m.mean() * 100), 1),
              "tds_p50": float(d.loc[m, "tds"].median()), "mg_p50": float(d.loc[m, "mg"].median()),
              "ca_p50": float(d.loc[m, "ca"].median()), "so4_p50": float(d.loc[m, "so4"].median()),
              "gypsum_sr_p50": float(eng.loc[m, "gypsum_sr"].median()), "lsi_p50": float(eng.loc[m, "lsi"].median()),
              "sar_p50": float(eng.loc[m, "sar"].median()),
              "hco3_cl_p50": float((d.loc[m, "hco3"] / d.loc[m, "cl"]).median()),
              "seawater_pct": round(float((d.loc[m, "plant_type"] == "seawater").mean() * 100), 1),
              "path_mix_pct": (eng.loc[m, "recommended"].value_counts(normalize=True) * 100).round(1).to_dict()}
        st["top_path"] = max(st["path_mix_pct"], key=st["path_mix_pct"].get)
        st["name"] = name_cluster(st)
        arche.append(st)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    pal = ["#1f5f99", "#e8a33d", "#3a9d5d", "#c4453c", "#7b5ea7", "#5aa3c8"]
    idx = rng.choice(len(d), size=min(8000, len(d)), replace=False)
    f = cluster_features(d)
    for a in arche:
        m = lab[idx] == a["cluster"]
        ax.scatter(f["log_tds"].to_numpy()[idx][m], f["log_mg_na"].to_numpy()[idx][m], s=3, alpha=0.4,
                   color=pal[a["cluster"] % 6], label=f"{a['name']} ({a['share_pct']}%)")
    ax.set_xlabel("log10 reject TDS (mg/L)"), ax.set_ylabel("log10 Mg/Na (mass)")
    ax.set_title("Reject water archetypes (KMeans on ion ratios + TDS)", fontsize=10)
    ax.legend(fontsize=6, markerscale=3, frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(), fig.savefig(FIG / "reject_archetypes.png"), plt.close(fig)
    return {"k": k, "silhouette": sil, "gmm_bic": bic, "archetypes": arche}


def distill_tree(d, eng, seed=0):
    x = pd.concat([d[["mg", "ca", "so4", "sio2", "reject_flow_m3d", "recovery", "reject_ph", "temp_c"]]
                   .reset_index(drop=True), eng[["tds", "sar", "lsi", "gypsum_sr"]]], axis=1)[TREE_VARS]
    y = eng["recommended"].to_numpy()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    best = None
    for depth in (3, 4):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=200, random_state=seed).fit(xtr, ytr)
        acc = float((t.predict(xte) == yte).mean())
        if best is None or acc > best[1] + 0.01:
            best = (t, acc, depth)
    tree, acc, depth = best
    rules = _leaf_rules(tree, TREE_VARS)
    fig, ax = plt.subplots(figsize=(16, 7), dpi=130)
    plot_tree(tree, feature_names=TREE_VARS, class_names=[LABELS[c] for c in tree.classes_], filled=True,
              impurity=False, proportion=True, rounded=True, fontsize=6, ax=ax)
    fig.tight_layout(), fig.savefig(FIG / "decision_tree.png"), plt.close(fig)
    return {"max_depth": depth, "fidelity_to_engine_pct": round(acc * 100, 1),
            "rules_text": export_text(tree, feature_names=TREE_VARS, decimals=2), "rules": rules}


def _leaf_rules(tree, names):
    t = tree.tree_
    out = []

    def label(node):  # class if the whole subtree predicts one class (redundant splits collapse)
        if t.children_left[node] == -1:
            return int(np.argmax(t.value[node][0]))
        a, b = label(t.children_left[node]), label(t.children_right[node])
        return a if a == b else None

    def walk(node, conds):
        if t.children_left[node] == -1 or label(node) is not None:
            v = t.value[node][0]
            cls = tree.classes_[int(np.argmax(v))]
            out.append({"if": _simplify(conds) or "always", "then": LABELS[cls],
                        "purity_pct": round(float(v.max() / v.sum() * 100), 1), "samples": int(t.n_node_samples[node])})
            return
        f, th = names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], conds + [(f, "<=", th)])
        walk(t.children_right[node], conds + [(f, ">", th)])

    walk(0, [])
    return sorted(out, key=lambda r: -r["samples"])


def _simplify(conds):
    """Keep only the tightest bound per feature and direction, e.g. 'tds > 5,677 and tds > 57,615' -> 'tds > 57,615'."""
    tight = {}
    for f, op, th in conds:
        k = (f, op)
        tight[k] = th if k not in tight else (max(tight[k], th) if op == ">" else min(tight[k], th))
    fmt = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"  # noqa: E731
    return " and ".join(f"{f} {op} {fmt(th)}" for (f, op), th in tight.items())


def path_quantiles(d, eng):
    v = pd.concat([d[["mg", "ca", "so4", "reject_flow_m3d", "recovery"]].reset_index(drop=True),
                   eng[["tds", "sar", "lsi", "gypsum_sr", "recommended"]]], axis=1)
    out = {}
    for p, g in v.groupby("recommended"):
        out[LABELS[p]] = {"n": int(len(g)), **{c: [round(float(q), 2) for q in g[c].quantile([.1, .5, .9])]
                                                 for c in ["tds", "mg", "ca", "so4", "sar", "lsi", "gypsum_sr",
                                                           "reject_flow_m3d", "recovery"]}}
    return out


def agreement():
    """Engine on predicted composition + measured sensors vs engine on true composition + true operation."""
    t = pd.read_parquet(TEST_PRED).reset_index(drop=True)
    truth = t[TARGETS + ["reject_ph", "temp_c", "recovery", "reject_flow_m3d"]].copy()
    eng_true = run_engine(truth)
    res = {}
    for mode in ["A", "B"]:
        p = pd.DataFrame({ion: t[f"pred{mode}_{ion}"] for ion in TARGETS})
        p["reject_ph"], p["temp_c"] = t["m_reject_ph"], t["m_temp_c"]
        p["recovery"], p["reject_flow_m3d"] = t["m_recovery"], t["m_reject_flow_m3d"]
        ok = p.notna().all(axis=1).to_numpy()
        e = run_engine(p[ok])
        agree = e["recommended"].to_numpy() == eng_true.loc[ok, "recommended"].to_numpy()
        sub = {"overall": np.ones(ok.sum(), bool), "saudi_like_top20": t.loc[ok, "saudi_top20"].to_numpy(),
               "brackish": (t.loc[ok, "plant_type"] != "seawater").to_numpy(),
               "seawater": (t.loc[ok, "plant_type"] == "seawater").to_numpy()}
        res[mode] = {k: {"n": int(m.sum()), "agreement_pct": round(float(agree[m].mean() * 100), 2)} for k, m in sub.items()}
        res[mode]["excluded_rows_missing_sensor"] = int((~ok).sum())
        res[mode]["confusion"] = pd.crosstab(eng_true.loc[ok, "recommended"].to_numpy(), e["recommended"].to_numpy(),
                                             rownames=["true"], colnames=["pred"]).to_dict()
    res["true_path_mix_pct"] = (eng_true["recommended"].value_counts(normalize=True) * 100).round(1).to_dict()
    return res


def main(seed=0):
    df = pd.read_parquet(DATA)
    d = df.sample(min(N_ENGINE, len(df)), random_state=seed).reset_index(drop=True)
    eng = run_engine(d[chem.IONS + ["reject_ph", "temp_c", "recovery", "reject_flow_m3d"]])
    out = {"n_engine_rows": len(d),
           "path_mix_pct": (eng["recommended"].value_counts(normalize=True) * 100).round(1).to_dict(),
           "reduce_feasible_pct": round(float(eng["reduce_feasible"].mean() * 100), 1),
           "reduce_reduction_p50_when_feasible": round(float(eng.loc[eng["reduce_feasible"], "reduce_reduction"].median()), 3),
           "reduce_limiting_factors_pct": (eng["reduce_limiting"].value_counts(normalize=True) * 100).round(1).to_dict()}
    out["clusters"] = clusters(d, eng, seed)
    out["decision_tree"] = distill_tree(d, eng, seed)
    out["path_quantiles"] = path_quantiles(d, eng)
    out["recommendation_agreement"] = agreement()
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(out["decision_tree"]["rules_text"])
    print("tree fidelity:", out["decision_tree"]["fidelity_to_engine_pct"])
    for a in out["clusters"]["archetypes"]:
        print(f"  archetype {a['cluster']}: {a['name']} ({a['share_pct']}%) top path {a['top_path']}")
    print("agreement:", json.dumps({m: {k: v for k, v in r.items() if k != 'confusion'}
                                    for m, r in out["recommendation_agreement"].items() if m in "AB"}, indent=1))
    return out


if __name__ == "__main__":
    main()
