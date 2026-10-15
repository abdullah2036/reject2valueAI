"""Evaluation helpers: grouped split, metrics, parity plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "models" / "figures"
HELDOUT_SEAWATER = "SW_GULF"  # train on Red Sea seawater, test on Arabian Gulf seawater


def group_split(df, test_frac=0.2, seed=7):
    """Hold out whole groups (US states / seawater body) so test water is never seen in training.
    Every scenario from the same well shares a group, so there is no leakage between scenarios."""
    br = df.loc[df["plant_type"] != "seawater", "group"]
    sizes = br.value_counts().sample(frac=1.0, random_state=seed)
    test_groups, n = [], 0
    for g, s in sizes.items():
        if n >= test_frac * len(br):
            break
        if n + s <= 1.5 * test_frac * len(br):
            test_groups.append(g)
            n += s
    test_groups.append(HELDOUT_SEAWATER)
    is_test = df["group"].isin(test_groups).to_numpy()
    return ~is_test, is_test, sorted(test_groups)


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    t, p = y_true[ok], y_pred[ok]
    rel = np.abs(p - t) / t
    ss_res, ss_tot = np.sum((t - p) ** 2), np.sum((t - t.mean()) ** 2)
    lt, lp = np.log10(t), np.log10(np.maximum(p, 1e-9))
    return {"n": int(ok.sum()), "mape_pct": float(np.mean(rel) * 100), "median_ape_pct": float(np.median(rel) * 100),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "r2_log": float(1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2)),
            "within_15pct": float(np.mean(rel <= 0.15) * 100), "within_10pct": float(np.mean(rel <= 0.10) * 100)}


def parity_plot(y_true, y_pred, title, path, p10=None, p90=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=min(6000, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    if p10 is not None:
        ax.vlines(y_true[idx[:400]], np.asarray(p10)[idx[:400]], np.asarray(p90)[idx[:400]],
                  color="#9db8d9", lw=0.6, alpha=0.6, label="P10-P90 (subset)")
    ax.scatter(y_true[idx], y_pred[idx], s=4, alpha=0.35, color="#1f5f99", label="prediction")
    lo, hi = np.nanpercentile(y_true, 0.5), np.nanpercentile(y_true, 99.5)
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color="#222", lw=1, label="1:1")
    ax.fill_between(xs, xs * 0.85, xs * 1.15, color="#e8a33d", alpha=0.18, label="+/-15%")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel("true reject concentration (mg/L)"), ax.set_ylabel("predicted (mg/L)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
