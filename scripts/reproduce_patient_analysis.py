"""Recompute patient-level research tables from released pseudonymized scores."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

MODELS = ("ResNet50V2", "VGG19", "ConvNeXt-Tiny", "Swin-Tiny")
KEYS = ["architecture", "outer_fold", "overlap_pct"]
ANALYSIS_SEED = 20260910
BOOTSTRAP_REPLICATES = 10000


def outcome(y, prob, threshold):
    pred = prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        roc_auc=roc_auc_score(y, prob),
        average_precision=average_precision_score(y, prob),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        sensitivity=tp / (tp + fn),
        specificity=tn / (tn + fp),
    )


def cond_metrics(scores, thresholds, fixed=False):
    rows = []
    for (architecture, fold, overlap), group in scores.groupby(KEYS):
        t = float(thresholds.loc[(architecture, fold, 0 if fixed else overlap)])
        rows.append(
            dict(
                architecture=architecture,
                outer_fold=fold,
                overlap_pct=overlap,
                threshold=t,
                **outcome(group.class_label, group.probability, t),
            )
        )
    return pd.DataFrame(rows)


def conditional_bootstrap(scores, architecture, realized):
    subset = scores[scores.architecture == architecture]
    folds = sorted(subset.outer_fold.unique())
    levels = sorted(subset.overlap_pct.unique())
    rng = np.random.default_rng(ANALYSIS_SEED)
    modern = architecture in MODELS[2:]
    if modern:
        draws = {}
        for f in folds:
            base = subset[(subset.outer_fold == f) & (subset.overlap_pct == 0)]
            counts = base.class_label.value_counts().to_dict()
            draws[f] = {c: np.zeros((BOOTSTRAP_REPLICATES, counts[c])) for c in (0, 1)}
        for i in range(BOOTSTRAP_REPLICATES):
            for f in folds:
                for c in (0, 1):
                    weights = draws[f][c]
                    n = weights.shape[1]
                    weights[i] = np.bincount(rng.choice(n, n, replace=True), minlength=n) / n
    estimates = {k: np.zeros(BOOTSTRAP_REPLICATES) for k in levels}
    slope = np.zeros(BOOTSTRAP_REPLICATES)
    for f in folds:
        g = subset[subset.outer_fold == f]
        base = g[g.overlap_pct == 0].sort_values("frozen_subject_order")
        y = base.class_label.to_numpy()
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        if modern:
            w1, w0 = draws[f][1], draws[f][0]
        else:
            w1 = rng.multinomial(n1, [1 / n1] * n1, size=BOOTSTRAP_REPLICATES) / n1
            w0 = rng.multinomial(n0, [1 / n0] * n0, size=BOOTSTRAP_REPLICATES) / n0
        foldvals = []
        for k in levels:
            d = g[g.overlap_pct == k].sort_values("frozen_subject_order")
            assert d.subject_token.tolist() == base.subject_token.tolist()
            assert np.array_equal(d.class_label, y)
            p = d.probability.to_numpy()
            pair = (p[y == 1, None] > p[None, y == 0]).astype(float) + 0.5 * (p[y == 1, None] == p[None, y == 0])
            result = np.einsum("bi,ij,bj->b", w1, pair, w0)
            estimates[k] += result / len(folds)
            foldvals.append(result)
        if architecture == "ResNet50V2":
            x = np.array([realized.loc[(architecture, f, k)] / 100 for k in levels])
            x = x - x.mean()
            slope += (x @ np.asarray(foldvals)) / (x @ x) / len(folds)
    lo, hi = np.percentile(estimates[100] - estimates[0], [2.5, 97.5])
    slope_bounds = np.percentile(slope, [2.5, 97.5]) if architecture == "ResNet50V2" else [np.nan, np.nan]
    return dict(
        architecture=architecture, ci_low=lo, ci_high=hi, slope_ci_low=slope_bounds[0], slope_ci_high=slope_bounds[1]
    )


def compare(actual, expected, cols):
    a = actual.set_index(KEYS).sort_index()
    b = expected.set_index(KEYS).sort_index()
    assert a.index.equals(b.index), "Run keys differ"
    np.testing.assert_allclose(a[cols], b[cols], rtol=0, atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/patient_reconstruction"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    data = args.data_dir
    args.output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(data / "patient_scores_original.csv")
    common = pd.read_csv(data / "patient_scores_original_common.csv")
    recorded = pd.read_csv(data / "reference_run_metrics.csv")
    expected = recorded.loc[recorded.evaluation_level == "patient"]
    assert len(expected) == 55
    th = expected.set_index(KEYS).selected_threshold
    actual = cond_metrics(scores, th)
    saved_keys = ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity"]
    compare(actual, expected, saved_keys)
    fixed = cond_metrics(scores, th, fixed=True)
    paired = actual.pivot(index=["architecture", "outer_fold"], columns="overlap_pct", values="roc_auc")
    endpoint = paired[[0, 100]].rename(columns={0: "auc_0", 100: "auc_100"}).copy()
    endpoint["delta"] = endpoint.auc_100 - endpoint.auc_0
    p = cond_metrics(common, th).pivot(index=["architecture", "outer_fold"], columns="overlap_pct", values="roc_auc")
    endpoint["common_file_delta"] = p[100] - p[0]
    endpoint["balanced_accuracy_fixed_delta"] = fixed.pivot(
        index=["architecture", "outer_fold"], columns="overlap_pct", values="balanced_accuracy"
    ).pipe(lambda df: df[100] - df[0])
    archived_ends = (
        pd.read_csv(data / "reference_fold_endpoints.csv").set_index(["architecture", "outer_fold"]).sort_index()
    )
    np.testing.assert_allclose(
        endpoint[["auc_0", "auc_100", "delta"]], archived_ends[["auc_0", "auc_100", "delta"]], atol=1e-10, rtol=0
    )
    realized = expected.set_index(KEYS).realized_overlap_pct
    cis = pd.DataFrame(conditional_bootstrap(scores, model, realized) for model in MODELS)
    ex = pd.read_csv(data / "reference_endpoint_intervals.csv").set_index("architecture").loc[list(MODELS)]
    np.testing.assert_allclose(cis[["ci_low", "ci_high"]], ex[["ci_low", "ci_high"]], rtol=0, atol=1e-12)
    slope = pd.read_csv(data / "reference_legacy_conditional_intervals.csv").query(
        'scope=="original" and architecture=="ResNet50V2" and analysis=="slope" and metric=="roc_auc"'
    )
    np.testing.assert_allclose(
        cis.loc[0, ["slope_ci_low", "slope_ci_high"]].to_numpy(dtype=float),
        slope[["ci_low", "ci_high"]].iloc[0].to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    actual.to_csv(args.output / "patient_metrics.csv", index=False)
    fixed.to_csv(args.output / "fixed_threshold_metrics.csv", index=False)
    endpoint.to_csv(args.output / "fold_endpoint_contrasts.csv")
    cis.to_csv(args.output / "conditional_intervals.csv", index=False)
    print(
        "Verified patient metrics for 55 runs, all 20 paired endpoint results, four conditional intervals, and the ResNet50V2 slope interval."
    )
    print(
        endpoint.groupby(level="architecture")[["delta", "common_file_delta", "balanced_accuracy_fixed_delta"]]
        .mean()
        .round(6)
        .to_string()
    )


if __name__ == "__main__":
    main()
