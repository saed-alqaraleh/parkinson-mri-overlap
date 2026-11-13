from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True) -> pd.DataFrame:
    model.eval()
    rows = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for x, y, metas in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x).reshape(-1)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            labels = y.numpy().astype(int)
            for meta, label, prob in zip(metas, labels, probs):
                rows.append({**meta, "class_label": int(label), "probability": float(prob)})
    return pd.DataFrame(rows)


def aggregate_patient(slice_predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"anonymous_subject_id", "class_label", "probability"}
    missing = required - set(slice_predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame missing {sorted(missing)}")
    conflicts = slice_predictions.groupby("anonymous_subject_id").class_label.nunique()
    if (conflicts > 1).any():
        raise ValueError("A subject has conflicting labels in predictions")
    return (
        slice_predictions.groupby(["anonymous_subject_id", "class_label"], as_index=False)
        .agg(probability=("probability", "mean"), n_slices=("probability", "size"))
    )


def metric_row(
    predictions: pd.DataFrame,
    threshold: float,
    level: str,
    architecture: str,
    fold: int,
    overlap_pct: int,
    replicate: int,
    training_seed: int,
    population: str = "all",
) -> dict:
    if predictions is None or len(predictions) == 0:
        return {
            "architecture": architecture,
            "outer_fold": fold,
            "overlap_pct": overlap_pct,
            "training_replicate": replicate,
            "training_seed": training_seed,
            "evaluation_level": level,
            "population": population,
            "n_subjects": np.nan,
            "n_slices": np.nan,
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "f1": np.nan,
            "mcc": np.nan,
            "selected_threshold": threshold,
            "warning": "NA population",
        }
    y = predictions.class_label.to_numpy(dtype=int)
    p = predictions.probability.to_numpy(dtype=float)
    yh = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    one_class = len(np.unique(y)) < 2
    if level == "slice":
        n_subjects = predictions.anonymous_subject_id.nunique()
        n_slices = len(predictions)
    elif level == "patient":
        n_subjects = len(predictions)
        n_slices = int(predictions.n_slices.sum()) if "n_slices" in predictions.columns else np.nan
    else:
        raise ValueError("level must be slice or patient")
    return {
        "architecture": architecture,
        "outer_fold": int(fold),
        "overlap_pct": int(overlap_pct),
        "training_replicate": int(replicate),
        "training_seed": int(training_seed),
        "evaluation_level": level,
        "population": population,
        "n_subjects": int(n_subjects),
        "n_pd": int((predictions.class_label == 1).sum()) if level == "patient" else int(predictions[predictions.class_label == 1].anonymous_subject_id.nunique()),
        "n_non_pd": int((predictions.class_label == 0).sum()) if level == "patient" else int(predictions[predictions.class_label == 0].anonymous_subject_id.nunique()),
        "n_slices": int(n_slices) if not (isinstance(n_slices, float) and math.isnan(n_slices)) else np.nan,
        "roc_auc": np.nan if one_class else roc_auc_score(y, p),
        "average_precision": np.nan if one_class else average_precision_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, yh),
        "sensitivity": recall_score(y, yh, pos_label=1, zero_division=0),
        "specificity": recall_score(y, yh, pos_label=0, zero_division=0),
        "accuracy": accuracy_score(y, yh),
        "precision": precision_score(y, yh, pos_label=1, zero_division=0),
        "f1": f1_score(y, yh, zero_division=0),
        "mcc": matthews_corrcoef(y, yh),
        "selected_threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "warning": "one class; rank metrics NA" if one_class else "",
    }


def annotate_test_predictions(
    predictions: pd.DataFrame,
    architecture: str,
    fold: int,
    replicate: int,
    training_seed: int,
    nominal_overlap_pct: int,
    realized_overlap_pct: float,
    represented_subjects: set[str],
    selected_threshold: float,
    checkpoint_identifier: str,
) -> pd.DataFrame:
    out = predictions.copy()
    out.insert(0, "architecture", architecture)
    out.insert(1, "outer_fold", int(fold))
    out.insert(2, "training_replicate", int(replicate))
    out.insert(3, "training_seed", int(training_seed))
    out.insert(4, "nominal_overlap_pct", int(nominal_overlap_pct))
    out.insert(5, "realized_overlap_pct", float(realized_overlap_pct))
    out["partition"] = "test"
    out["subject_represented_in_training"] = out.anonymous_subject_id.astype(str).isin(represented_subjects)
    out["common_held_out_intersection"] = False
    out["selected_validation_threshold"] = float(selected_threshold)
    out["checkpoint_identifier"] = checkpoint_identifier
    return out


def common_held_out_analysis(frames: Iterable[pd.DataFrame]) -> list[pd.DataFrame]:
    frames = list(frames)
    if not frames:
        return []
    common = set.intersection(*(set(f.file_identifier.astype(str)) for f in frames))
    result = []
    for f in frames:
        z = f[f.file_identifier.astype(str).isin(common)].copy()
        z["common_held_out_intersection"] = True
        result.append(z)
    return result
