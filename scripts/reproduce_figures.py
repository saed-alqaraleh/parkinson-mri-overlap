"""Regenerate the manuscript-style Figures 2–4 from archived evaluation evidence.

Figure 2 contains four ResNet50V2 panels evaluated at realized subject overlap.
Figures 3–4 compare conditional endpoint effects and fold-4 omission sensitivity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.reproduce_full_statistics import reproduce

ORDER = ("ResNet50V2", "VGG19", "ConvNeXt-Tiny", "Swin-Tiny")
COLORS = ("#1C5279", "#D07B2B", "#2B7F73")


def clean_axis(ax, *, xlabel: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, color="0.75", linewidth=0.6)
    if xlabel:
        ax.set_xlabel(xlabel)


def figure2(data: Path, out: Path) -> None:
    """Render patient/slice ROC-AUC and balanced accuracy with realized-overlap x-axis."""
    patient = pd.read_csv(data / "original_run_metrics.csv")
    slice_level = pd.read_csv(data / "original_slice_metrics.csv")
    panels = (
        (patient, "roc_auc", "(a)  Patient ROC-AUC"),
        (slice_level, "roc_auc", "(b)  Slice ROC-AUC"),
        (patient, "balanced_accuracy", "(c)  Patient balanced accuracy"),
        (slice_level, "balanced_accuracy", "(d)  Slice balanced accuracy"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True, sharey=True, layout="constrained")
    for ax, (source, metric, heading) in zip(axes.flat, panels):
        table = source.loc[source.architecture == "ResNet50V2"]
        for fold, rows in table.groupby("outer_fold"):
            rows = rows.sort_values("realized_overlap_pct")
            ax.plot(
                rows.realized_overlap_pct,
                rows[metric],
                color="#9FAAB2",
                lw=1.05,
                marker="o",
                markersize=2.9,
                alpha=0.85,
                label="Individual folds" if fold == 0 else None,
            )
        means = (
            table.groupby("overlap_pct", sort=True)
            .agg(x=("realized_overlap_pct", "mean"), value=(metric, "mean"))
            .sort_values("x")
        )
        ax.plot(means.x, means.value, color=COLORS[0], lw=2.5, marker="o", markersize=5.5, label="Fold mean")
        ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlim(-3, 103)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(heading, loc="left", weight="bold", fontsize=10.5)
        clean_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score")
    for ax in axes[1, :]:
        ax.set_xlabel("Realized subject overlap (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles[::-1], labels[::-1], loc="upper left", fontsize=8.5, frameon=False)
    fig.savefig(out / "Figure_2_graded_response.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / "Figure_2_graded_response.pdf", bbox_inches="tight")
    plt.close(fig)


def _series_plot(ax, frames: list[tuple[str, pd.DataFrame, str, float, str]]) -> None:
    for name, frame, marker, dy, color in frames:
        selected = frame.set_index("architecture")
        for i, model in enumerate(ORDER):
            row = selected.loc[model]
            y = i + dy
            ax.plot([row.ci_low, row.ci_high], [y, y], lw=1.75, color=color)
            ax.plot(
                row.mean_delta,
                y,
                marker=marker,
                ms=6.2,
                color=color,
                mfc="white" if marker == "D" else color,
                label=name if i == 0 else None,
            )
    ax.set_yticks(range(len(ORDER)), ORDER)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.axvline(0, color="0.50", linestyle="--", lw=1)
    ax.set_xlim(-0.11, 0.32)
    clean_axis(ax, xlabel="Paired change in patient ROC-AUC (100% minus 0%)")
    ax.legend(frameon=False, ncol=1, fontsize=8.5, loc="upper right")


def figure3(stats: pd.DataFrame, out: Path) -> None:
    original = stats.query('scope == "original"')
    omission = stats.query('scope == "original_excluding_fold4"')
    fig, ax = plt.subplots(figsize=(8.6, 4.45), layout="constrained")
    _series_plot(
        ax,
        [
            ("Original five folds", original, "o", -0.12, COLORS[0]),
            ("Fold 4 omitted after audit", omission, "D", 0.12, COLORS[1]),
        ],
    )
    ax.set_title("Original-seed, audit-informed endpoint sensitivity", weight="bold", loc="left")
    fig.savefig(out / "Figure_3_endpoint_contrasts.png", dpi=240)
    fig.savefig(out / "Figure_3_endpoint_contrasts.pdf")
    plt.close(fig)


def figure4(stats: pd.DataFrame, out: Path) -> None:
    """Display all three seed summaries in both five-fold and four-fold panels."""
    five = (
        ("Original seed", "original", "o", -0.19, COLORS[0]),
        ("Second seed", "second_seed", "s", 0.0, COLORS[1]),
        ("Seed average", "seed_average", "D", 0.19, COLORS[2]),
    )
    four = (
        ("Original seed", "original_excluding_fold4", "o", -0.19, COLORS[0]),
        ("Second seed", "second_seed_excluding_fold4", "s", 0.0, COLORS[1]),
        ("Seed average", "seed_average_excluding_fold4", "D", 0.19, COLORS[2]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.7, 4.8), sharex=True, sharey=True, layout="constrained")
    for ax, configurations, title in (
        (axes[0], five, "(A)  All five outer folds"),
        (axes[1], four, "(B)  Fold 4 omitted after content audit"),
    ):
        series = [
            (label, stats.query("scope == @scope"), marker, dy, color)
            for label, scope, marker, dy, color in configurations
        ]
        _series_plot(ax, series)
        ax.set_title(title, fontsize=11, weight="bold", loc="left", pad=12)
    fig.savefig(out / "Figure_4_seed_and_fold_sensitivity.png", dpi=240)
    fig.savefig(out / "Figure_4_seed_and_fold_sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = reproduce(args.data_dir, args.out / "statistics")
    figure2(args.data_dir, args.out)
    figure3(summary["intervals"], args.out)
    figure4(summary["intervals"], args.out)
    print("Reproduced Figures 2–4 with full panel layouts, shared intervals and fold-4 omission.")


if __name__ == "__main__":
    main()
