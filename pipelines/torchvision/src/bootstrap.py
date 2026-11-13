from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ANALYSIS_SEED = 20260910


def _patient_auc(df: pd.DataFrame) -> float:
    y = df.class_label.to_numpy(dtype=int)
    p = df.probability.to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def paired_subject_bootstrap_endpoint(
    predictions_0: pd.DataFrame,
    predictions_100: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = ANALYSIS_SEED,
) -> Dict[str, object]:
    """Paired, diagnosis-stratified subject bootstrap for one fitted model pair.

    Both frames must be patient-level, with one row per subject. The same sampled
    subjects (within diagnosis) are applied to the 0% and 100% conditions.
    """
    req = {"anonymous_subject_id", "class_label", "probability"}
    for name, df in [("0%", predictions_0), ("100%", predictions_100)]:
        if req - set(df.columns):
            raise ValueError(f"{name} patient predictions missing {sorted(req - set(df.columns))}")
        if df.anonymous_subject_id.duplicated().any():
            raise ValueError(f"{name} patient predictions are not one row per subject")
    s0 = set(predictions_0.anonymous_subject_id.astype(str))
    s1 = set(predictions_100.anonymous_subject_id.astype(str))
    if s0 != s1:
        raise ValueError("Paired bootstrap requires identical subject identity sets")
    left = predictions_0.set_index("anonymous_subject_id").sort_index()
    right = predictions_100.set_index("anonymous_subject_id").sort_index()
    if not np.array_equal(left.class_label.to_numpy(), right.class_label.to_numpy()):
        raise ValueError("Diagnosis labels differ between paired conditions")
    labels = left.class_label.astype(int)
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    ids_by_class = {cls: labels[labels == cls].index.to_numpy() for cls in sorted(labels.unique())}
    for _ in range(int(n_bootstrap)):
        sampled_ids = []
        for cls, ids in ids_by_class.items():
            sampled_ids.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
        a = left.loc[sampled_ids].reset_index()
        b = right.loc[sampled_ids].reset_index()
        diffs.append(_patient_auc(b) - _patient_auc(a))
    diffs_arr = np.asarray(diffs, dtype=float)
    finite = diffs_arr[np.isfinite(diffs_arr)]
    observed = _patient_auc(right.reset_index()) - _patient_auc(left.reset_index())
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "observed_difference": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_bootstrap": int(n_bootstrap),
        "analysis_seed": int(seed),
        "bootstrap_unit": "subject within diagnosis",
        "conditioning_statement": "Interval conditions on the fitted model pair and fixed subject partition.",
    }


def aggregate_repeated_seed_endpoint(
    patient_predictions: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = ANALYSIS_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate repeated-seed endpoint results without treating seeds as cohorts.

    Input columns: architecture, outer_fold, training_replicate, overlap_pct,
    anonymous_subject_id, class_label, probability.

    Returns fold/seed deltas and seed-averaged fold deltas. Bootstrap intervals
    are intended to be computed per fitted pair or by a manuscript-specific
    aggregation layer; this function preserves the non-independence structure.
    """
    required = {
        "architecture", "outer_fold", "training_replicate", "overlap_pct",
        "anonymous_subject_id", "class_label", "probability"
    }
    missing = required - set(patient_predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows = []
    for (arch, fold, rep), g in patient_predictions.groupby(["architecture", "outer_fold", "training_replicate"]):
        g0 = g[g.overlap_pct == 0]
        g1 = g[g.overlap_pct == 100]
        if g0.empty or g1.empty:
            continue
        result = paired_subject_bootstrap_endpoint(g0, g1, n_bootstrap=n_bootstrap, seed=seed + int(fold) * 1000 + int(rep))
        rows.append({
            "architecture": arch,
            "outer_fold": int(fold),
            "training_replicate": int(rep),
            **result,
        })
    pair = pd.DataFrame(rows)
    if pair.empty:
        return pair, pair
    fold = pair.groupby(["architecture", "outer_fold"], as_index=False).agg(
        seed_averaged_difference=("observed_difference", "mean"),
        seed_sd=("observed_difference", "std"),
        n_training_replicates=("observed_difference", "size"),
    )
    return pair, fold


def conditional_collection_bootstrap(
    patient_predictions: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = ANALYSIS_SEED,
) -> Dict[str, object]:
    """Manuscript-compatible paired, diagnosis-stratified subject bootstrap.

    For each bootstrap replicate:
    1. Within each outer fold and diagnosis, resample test subjects with replacement.
    2. Apply exactly the same sampled subject multiplicities to paired 0% and 100%
       predictions for every training replicate.
    3. Compute the 100%-0% patient ROC-AUC difference per training replicate.
    4. Average training-replicate differences within fold.
    5. Average the five fold values with equal fold weight.

    The interval therefore respects paired overlap conditions and does not count
    training seeds as independent cohorts. It conditions on the fitted model
    collection and fixed outer partitions.
    """
    required = {
        "architecture", "outer_fold", "training_replicate", "overlap_pct",
        "anonymous_subject_id", "class_label", "probability"
    }
    missing = required - set(patient_predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    architectures = patient_predictions.architecture.unique().tolist()
    if len(architectures) != 1:
        raise ValueError("Run conditional_collection_bootstrap separately per architecture")
    df = patient_predictions[patient_predictions.overlap_pct.isin([0, 100])].copy()
    rng = np.random.default_rng(seed)

    # Validate pairing and build a lookup keyed by fold/rep/overlap.
    lookup = {}
    folds = sorted(df.outer_fold.unique())
    for fold in folds:
        fdf = df[df.outer_fold == fold]
        reps = sorted(fdf.training_replicate.unique())
        for rep in reps:
            r = fdf[fdf.training_replicate == rep]
            a = r[r.overlap_pct == 0].copy()
            b = r[r.overlap_pct == 100].copy()
            if a.empty or b.empty:
                raise ValueError(f"Missing paired endpoint fold={fold}, replicate={rep}")
            if set(a.anonymous_subject_id.astype(str)) != set(b.anonymous_subject_id.astype(str)):
                raise ValueError(f"Subject-set mismatch fold={fold}, replicate={rep}")
            aa = a.set_index("anonymous_subject_id").sort_index()
            bb = b.set_index("anonymous_subject_id").sort_index()
            if not np.array_equal(aa.class_label.to_numpy(), bb.class_label.to_numpy()):
                raise ValueError(f"Label mismatch fold={fold}, replicate={rep}")
            lookup[(int(fold), int(rep), 0)] = aa
            lookup[(int(fold), int(rep), 100)] = bb

    # Observed seed-averaged fold deltas.
    observed_fold = []
    for fold in folds:
        reps = sorted(df[df.outer_fold == fold].training_replicate.unique())
        ds = []
        for rep in reps:
            a = lookup[(int(fold), int(rep), 0)].reset_index()
            b = lookup[(int(fold), int(rep), 100)].reset_index()
            ds.append(_patient_auc(b) - _patient_auc(a))
        observed_fold.append(float(np.mean(ds)))
    observed = float(np.mean(observed_fold))

    boots = []
    for _ in range(int(n_bootstrap)):
        fold_values = []
        for fold in folds:
            reps = sorted(df[df.outer_fold == fold].training_replicate.unique())
            # Subject labels are identical across replicate/overlap; take first pair.
            base = lookup[(int(fold), int(reps[0]), 0)]
            ids_by_class = {
                cls: base[base.class_label.astype(int) == cls].index.to_numpy()
                for cls in sorted(base.class_label.astype(int).unique())
            }
            sampled = []
            for cls, ids in ids_by_class.items():
                sampled.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
            rep_deltas = []
            for rep in reps:
                a = lookup[(int(fold), int(rep), 0)].loc[sampled].reset_index()
                b = lookup[(int(fold), int(rep), 100)].loc[sampled].reset_index()
                rep_deltas.append(_patient_auc(b) - _patient_auc(a))
            fold_values.append(float(np.mean(rep_deltas)))
        boots.append(float(np.mean(fold_values)))
    b = np.asarray(boots, dtype=float)
    b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {
        "architecture": architectures[0],
        "observed_mean_seed_averaged_fold_delta": observed,
        "observed_seed_averaged_fold_deltas": observed_fold,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_replicates": int(n_bootstrap),
        "analysis_seed": int(seed),
        "resampling_unit": "subject within diagnosis and outer fold",
        "pairing": "same resampled subjects applied to 0% and 100% and all training replicates",
        "aggregation": "training-replicate delta mean within fold, then unweighted mean across folds",
        "conditioning_statement": "Interval conditions on the fitted collection of models and the fixed subject partitions; it does not include alternative outer partitions.",
    }
