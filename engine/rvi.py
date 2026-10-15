"""Reject Value Index (0-100): weighted sum of path sub-scores (weights in assumptions.yaml)."""
from __future__ import annotations

import numpy as np

from engine import load_assumptions


def weights():
    w = load_assumptions()["rvi_weights"]
    return {k: float(v) for k, v in w.items() if k != "source"}


def rvi(recover_score, reuse_score, reduce_score, burden):
    w = weights()
    return (w["resource_value"] * np.asarray(recover_score)
            + w["reusability"] * np.asarray(reuse_score)
            + w["reduction_potential"] * np.asarray(reduce_score)
            + w["environmental_burden_inverse"] * (100 - np.asarray(burden)))
