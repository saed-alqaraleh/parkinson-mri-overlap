"""Generate table-ready study estimates without refitting the backbones."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reproduce_full_statistics import reproduce

MODELS = ("ResNet50V2", "VGG19", "ConvNeXt-Tiny", "Swin-Tiny")


def table3(data: Path) -> pd.DataFrame:
    patient = pd.read_csv(data / "original_run_metrics.csv")
    slices = pd.read_csv(data / "original_slice_metrics.csv")
    metrics = ["roc_auc", "average_precision", "balanced_accuracy"]
    p = patient.groupby(["architecture", "overlap_pct"])[metrics].agg(["mean", "std"])
    p.columns = [f"patient_{key}_{stat}" for key, stat in p.columns]
    q = slices.groupby(["architecture", "overlap_pct"])["roc_auc"].agg(["mean", "std"])
    q.columns = ["slice_roc_auc_mean", "slice_roc_auc_sd"]
    return p.join(q, how="inner").reset_index()


def table4(stats: pd.DataFrame) -> pd.DataFrame:
    orig = stats.query('scope == "original"')
    common = stats.query('scope == "original_common_files"')[["architecture", "mean_delta"]]
    common = common.rename(columns={"mean_delta": "common_held_out_delta"})
    return orig.merge(common, on="architecture", validate="one_to_one")


def table5(stats: pd.DataFrame, fold: pd.DataFrame) -> pd.DataFrame:
    original = stats.query('scope == "original"')[["architecture", "mean_delta"]].rename(
        columns={"mean_delta": "original_delta"}
    )
    repeat = stats.query('scope == "second_seed"')[
        ["architecture", "mean_delta", "ci_low", "ci_high", "positive_folds", "tied_folds", "negative_folds"]
    ].rename(columns={"mean_delta": "second_seed_delta"})
    avg = stats.query('scope == "seed_average"')[["architecture", "mean_delta", "ci_low", "ci_high"]].rename(
        columns={
            "mean_delta": "seed_averaged_delta",
            "ci_low": "seed_average_ci_low",
            "ci_high": "seed_average_ci_high",
        }
    )
    p = fold.pivot(index=["architecture", "outer_fold"], columns="scope", values="delta")
    gaps = (p.original - p.second_seed).abs().groupby(level="architecture").agg(["mean", "max"])
    gaps.columns = ["mean_absolute_seed_gap", "max_absolute_fold_seed_gap"]
    return (
        original.merge(repeat, on="architecture", validate="one_to_one")
        .merge(avg, on="architecture")
        .merge(gaps.reset_index(), on="architecture")
    )


def reproduce_tables(data: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    analysis = reproduce(data, output / "statistics")
    tables = {
        "Table_3_metrics.csv": table3(data),
        "Table_4_original_contrasts.csv": table4(analysis["intervals"]),
        "Table_5_seed_sensitivity.csv": table5(analysis["intervals"], analysis["folds"]),
        "Table_6_content_audit.csv": pd.read_csv(data / "content_audit_main_table.csv"),
        "Supplementary_original_identity_auc.csv": pd.read_csv(data / "original_representation_summary.csv"),
        "Supplementary_diagnosis_matched_identity_auc.csv": pd.read_csv(
            data / "diagnosis_matched_representation_summary.csv"
        ),
    }
    assert np.isclose(
        tables["Table_3_metrics.csv"]
        .query('architecture == "ResNet50V2" and overlap_pct == 0')
        .patient_roc_auc_mean.iloc[0],
        0.3572222222222222,
        atol=0.001,
    )
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    print("Regenerated Tables 3–6 and supplementary summary tables from archived study evidence.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()
    reproduce_tables(args.data_dir, args.out)


if __name__ == "__main__":
    main()
