from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef


THRESHOLDS = np.round(np.arange(0.05, 0.9500001, 0.005), 3)


def select_validation_threshold(val_patient: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    required = {"class_label", "probability"}
    missing = required - set(val_patient.columns)
    if missing:
        raise ValueError(f"Validation patient frame missing {sorted(missing)}")
    y = val_patient.class_label.to_numpy(dtype=int)
    p = val_patient.probability.to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Validation threshold selection requires both classes")
    rows = []
    for t in THRESHOLDS:
        yh = (p >= float(t)).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "balanced_accuracy": balanced_accuracy_score(y, yh),
                "mcc": matthews_corrcoef(y, yh),
                "f1": f1_score(y, yh, zero_division=0),
                "proximity_to_0_5": -abs(float(t) - 0.5),
            }
        )
    curve = pd.DataFrame(rows)
    # Historical study tie hierarchy, plus explicit deterministic lower-threshold fallback.
    ranked = curve.sort_values(
        ["balanced_accuracy", "mcc", "f1", "proximity_to_0_5", "threshold"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return float(ranked.iloc[0].threshold), ranked


def audit_selected_threshold(curve: pd.DataFrame, selected: float, atol: float = 1e-12) -> bool:
    required = {"threshold", "balanced_accuracy", "mcc", "f1", "proximity_to_0_5"}
    if required - set(curve.columns):
        raise ValueError("Threshold curve schema mismatch")
    ranked = curve.sort_values(
        ["balanced_accuracy", "mcc", "f1", "proximity_to_0_5", "threshold"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return abs(float(ranked.iloc[0].threshold) - float(selected)) <= atol
