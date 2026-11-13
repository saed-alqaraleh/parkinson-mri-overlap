"""Reproduce observed-study statistics from released pseudonymized predictions.

This script does not fit any model. Its bootstrap resamples recorded subjects
within diagnosis and outer fold, conditional on the 95 saved fitted models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.reproduce_patient_analysis import (
    ANALYSIS_SEED,
    BOOTSTRAP_REPLICATES,
    KEYS,
    MODELS,
    cond_metrics,
    conditional_bootstrap,
)


def _paired_auc(group: pd.DataFrame) -> dict[int, float]:
    return {
        int(level): float(roc_auc_score(frame.class_label, frame.probability))
        for level, frame in group.groupby("overlap_pct")
    }


def _point_slope(scores: pd.DataFrame, realized: pd.Series) -> float:
    slopes = []
    for fold, group in scores.query('architecture == "ResNet50V2"').groupby("outer_fold"):
        auc = _paired_auc(group)
        levels = sorted(auc)
        x = np.array([realized.loc[("ResNet50V2", fold, k)] / 100 for k in levels])
        x -= x.mean()
        slopes.append(float(x @ np.array([auc[k] for k in levels]) / (x @ x)))
    return float(np.mean(slopes))


def shared_seed_average_ci(
    original: pd.DataFrame,
    second_seed: pd.DataFrame,
    architecture: str,
    n_reps: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Apply the same within-fold, within-diagnosis resample to both fitted seeds."""
    a = original.query("architecture == @architecture")
    b = second_seed.query("architecture == @architecture")
    rng = np.random.default_rng(ANALYSIS_SEED)
    folds = sorted(a.outer_fold.unique())
    assert len(folds) in (4, 5)
    # TorchVision archival interval procedure draws by fold/diagnosis first.
    # This preserves that ordering for comparisons with archived endpoint CIs.
    modern = architecture in MODELS[2:]
    precomputed = {}
    if modern:
        for fold in folds:
            base = a[(a.outer_fold == fold) & (a.overlap_pct == 0)]
            counts = base.class_label.value_counts().to_dict()
            precomputed[fold] = {cls: np.zeros((n_reps, counts[cls])) for cls in (0, 1)}
        for iteration in range(n_reps):
            for fold in folds:
                for cls in (0, 1):
                    weights = precomputed[fold][cls]
                    n = weights.shape[1]
                    weights[iteration] = np.bincount(rng.choice(n, n, replace=True), minlength=n) / n
    aggregate = np.zeros(n_reps)
    for fold in folds:
        ga = a[a.outer_fold == fold]
        gb = b[b.outer_fold == fold]
        base = ga[ga.overlap_pct == 0].sort_values("frozen_subject_order")
        labels = base.class_label.to_numpy(int)
        n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
        if modern:
            wpos, wneg = precomputed[fold][1], precomputed[fold][0]
        else:
            wpos = rng.multinomial(n_pos, [1 / n_pos] * n_pos, size=n_reps) / n_pos
            wneg = rng.multinomial(n_neg, [1 / n_neg] * n_neg, size=n_reps) / n_neg
        for seed_data in (ga, gb):
            for level, sign in ((0, -1), (100, 1)):
                frame = seed_data[seed_data.overlap_pct == level].sort_values("frozen_subject_order")
                assert frame.subject_token.tolist() == base.subject_token.tolist()
                assert np.array_equal(frame.class_label.to_numpy(int), labels)
                pred = frame.probability.to_numpy(float)
                pair = (pred[labels == 1, None] > pred[None, labels == 0]).astype(float)
                pair += 0.5 * (pred[labels == 1, None] == pred[None, labels == 0])
                aggregate += sign * np.einsum("bi,ij,bj->b", wpos, pair, wneg) / (2 * len(folds))
    return tuple(float(v) for v in np.percentile(aggregate, [2.5, 97.5]))


def reproduce(data: Path, out: Path) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(data / "patient_scores_original.csv")
    b = pd.read_csv(data / "patient_scores_second_seed.csv")
    a_common = pd.read_csv(data / "patient_scores_original_common.csv")
    b_common = pd.read_csv(data / "patient_scores_second_seed_common.csv")
    reference = pd.read_csv(data / "reference_run_metrics.csv")
    per_run = reference.query('evaluation_level == "patient"')
    realized = per_run.set_index(KEYS).realized_overlap_pct
    thresholds = per_run.set_index(KEYS).selected_threshold
    assert a.groupby(KEYS).ngroups == 55 and b.groupby(KEYS).ngroups == 40
    # Recalculate 55 recorded metrics rather than trusting a copied table.
    actual = cond_metrics(a, thresholds)
    previous = per_run.set_index(KEYS).sort_index()
    observed = actual.set_index(KEYS).sort_index()
    np.testing.assert_allclose(
        observed[["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity"]],
        previous[["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity"]],
        rtol=0,
        atol=1e-10,
    )
    result_rows = []
    fold_rows = []
    for model in MODELS:
        df0 = a.query("architecture == @model")
        df1 = b.query("architecture == @model")
        for scope, frame in (
            ("original", df0),
            ("second_seed", df1),
            ("original_common_files", a_common.query("architecture == @model")),
            ("second_seed_common_files", b_common.query("architecture == @model")),
            ("original_excluding_fold4", df0[df0.outer_fold != 4]),
            ("second_seed_excluding_fold4", df1[df1.outer_fold != 4]),
        ):
            by_fold = []
            for fold, gf in frame.groupby("outer_fold"):
                auc = _paired_auc(gf)
                delta = auc[100] - auc[0]
                by_fold.append(delta)
                fold_rows.append(
                    dict(
                        architecture=model,
                        scope=scope,
                        outer_fold=int(fold),
                        auc_0=auc[0],
                        auc_100=auc[100],
                        delta=delta,
                    )
                )
            interval = conditional_bootstrap(frame, model, realized)
            result_rows.append(
                dict(
                    architecture=model,
                    scope=scope,
                    mean_delta=float(np.mean(by_fold)),
                    ci_low=float(interval["ci_low"]),
                    ci_high=float(interval["ci_high"]),
                    positive_folds=sum(d > 1e-12 for d in by_fold),
                    tied_folds=sum(abs(d) <= 1e-12 for d in by_fold),
                    negative_folds=sum(d < -1e-12 for d in by_fold),
                    n_folds=len(by_fold),
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=ANALYSIS_SEED,
                )
            )
        low, high = shared_seed_average_ci(a, b, model)
        delta0 = np.mean(
            [
                row["mean_delta"]
                for row in result_rows
                if row["architecture"] == model and row["scope"] in ("original", "second_seed")
            ]
        )
        result_rows.append(
            dict(
                architecture=model,
                scope="seed_average",
                mean_delta=float(delta0),
                ci_low=low,
                ci_high=high,
                positive_folds=np.nan,
                tied_folds=np.nan,
                negative_folds=np.nan,
                n_folds=5,
                replicates=BOOTSTRAP_REPLICATES,
                seed=ANALYSIS_SEED,
            )
        )
        # Apply the same held-out subject draw to both seeds after omitting the audited fold.
        remaining_a = df0[df0.outer_fold != 4]
        remaining_b = df1[df1.outer_fold != 4]
        remaining_low, remaining_high = shared_seed_average_ci(remaining_a, remaining_b, model)
        remaining_delta_a = next(
            r["mean_delta"]
            for r in result_rows
            if r["architecture"] == model and r["scope"] == "original_excluding_fold4"
        )
        remaining_delta_b = next(
            r["mean_delta"]
            for r in result_rows
            if r["architecture"] == model and r["scope"] == "second_seed_excluding_fold4"
        )
        result_rows.append(
            dict(
                architecture=model,
                scope="seed_average_excluding_fold4",
                mean_delta=float((remaining_delta_a + remaining_delta_b) / 2),
                ci_low=remaining_low,
                ci_high=remaining_high,
                positive_folds=np.nan,
                tied_folds=np.nan,
                negative_folds=np.nan,
                n_folds=4,
                replicates=BOOTSTRAP_REPLICATES,
                seed=ANALYSIS_SEED,
            )
        )
    stats = pd.DataFrame(result_rows)
    folds = pd.DataFrame(fold_rows)
    ref_intervals = pd.read_csv(data / "reference_endpoint_intervals.csv").set_index("architecture")
    orig_ci = stats.query('scope == "original"').set_index("architecture").loc[list(MODELS)]
    np.testing.assert_allclose(
        orig_ci[["ci_low", "ci_high"]], ref_intervals.loc[list(MODELS), ["ci_low", "ci_high"]], atol=1e-12, rtol=0
    )
    # Printed Table 5 interval endpoints use independently archived bootstrap draws.
    # Verify differences transparently; never silently substitute printed endpoints.
    table5_reference = pd.read_csv(data / "reference_table5_intervals.csv")
    for row in table5_reference.itertuples(index=False):
        observed_row = stats.query("architecture == @row.architecture and scope == @row.scope").iloc[0]
        assert abs(observed_row.ci_low - row.printed_ci_low) <= 0.003
        assert abs(observed_row.ci_high - row.printed_ci_high) <= 0.003
    # Original fold-4 omission check against the verified independent archived sensitivity.
    expected_fold4 = {
        "ResNet50V2": (-0.03125, 0.11944444444444446),
        "VGG19": (0.043055555555555625, 0.24791666666666662),
        "ConvNeXt-Tiny": (0.0, 0.07777777777777783),
        "Swin-Tiny": (-0.021527777777777812, 0.11875000000000002),
    }
    for model, expected_ci in expected_fold4.items():
        row = stats.query('scope == "original_excluding_fold4" and architecture == @model').iloc[0]
        np.testing.assert_allclose([row.ci_low, row.ci_high], expected_ci, atol=1e-12, rtol=0)
    original_slope = conditional_bootstrap(a, "ResNet50V2", realized)
    slope = pd.DataFrame(
        [
            dict(
                architecture="ResNet50V2",
                scope="original",
                roc_auc_slope=_point_slope(a, realized),
                ci_low=original_slope["slope_ci_low"],
                ci_high=original_slope["slope_ci_high"],
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                bootstrap_seed=ANALYSIS_SEED,
            )
        ]
    )
    saved_slope = pd.read_csv(data / "reference_legacy_conditional_intervals.csv").query(
        'scope == "original" and architecture == "ResNet50V2" and analysis == "slope" and metric == "roc_auc"'
    )
    np.testing.assert_allclose(slope[["ci_low", "ci_high"]], saved_slope[["ci_low", "ci_high"]], atol=1e-12, rtol=0)
    fixed = cond_metrics(a, thresholds, fixed=True)
    combined = cond_metrics(a_common, thresholds)
    stats.to_csv(out / "bootstrap_intervals.csv", index=False)
    folds.to_csv(out / "fold_endpoint_deltas.csv", index=False)
    slope.to_csv(out / "graded_slope.csv", index=False)
    fixed.to_csv(out / "fixed_threshold_metrics.csv", index=False)
    combined.to_csv(out / "common_held_out_metrics.csv", index=False)
    # Provide an evidence-only, per-architecture summary for Table 3, including slice AUC.
    for name in ("original_slice_metrics.csv", "second_seed_slice_metrics.csv"):
        source = pd.read_csv(data / name)
        assert len(source) in (40, 55)
        cols = ["roc_auc", "balanced_accuracy"]
        g = source.groupby(["architecture", "overlap_pct"])[cols].agg(["mean", "std"])
        g.to_csv(out / name.replace("metrics", "summary"))
    return {"intervals": stats, "slope": slope, "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, default=Path("outputs/reproduced"))
    args = parser.parse_args()
    r = reproduce(args.data_dir, args.out)
    print(r["intervals"]["architecture scope mean_delta ci_low ci_high".split()].to_string(index=False))
    print(r["slope"].to_string(index=False))


if __name__ == "__main__":
    main()
