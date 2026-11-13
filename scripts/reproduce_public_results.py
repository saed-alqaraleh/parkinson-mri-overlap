"""Recreate shipped aggregate research tables from saved, non-identifying run metrics."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ORDER = ["ResNet50V2", "VGG19", "ConvNeXt-Tiny", "Swin-Tiny"]


def reproduce(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    o = pd.read_csv(root / "data/original_run_metrics.csv")
    s = pd.read_csv(root / "data/second_seed_run_metrics.csv")
    assert len(o) == 55 and len(s) == 40
    assert len(o[o.overlap_pct.isin([0, 100])]) == 40
    assert not o.duplicated(["architecture", "outer_fold", "overlap_pct"]).any()
    assert not s.duplicated(["architecture", "outer_fold", "overlap_pct"]).any()
    a = []
    for group, df in [("original", o), ("second_seed", s)]:
        p = df.pivot(index=["architecture", "outer_fold"], columns="overlap_pct", values="roc_auc")
        assert len(p) == 20 and p[[0, 100]].notna().all().all()
        tab = p[[0, 100]].rename(columns={0: "auc_0", 100: "auc_100"}).reset_index()
        tab["delta"] = tab.auc_100 - tab.auc_0
        tab["seed_group"] = group
        a.append(tab)
    fold = pd.concat(a, ignore_index=True).sort_values(["seed_group", "architecture", "outer_fold"])
    fold.to_csv(output / "fold_paired_endpoints.csv", index=False)
    summ = fold.groupby(["architecture", "seed_group"])["delta"].mean().unstack()
    summ["seed_average"] = (summ["original"] + summ["second_seed"]) / 2
    summ = summ.loc[ORDER].reset_index()
    summ.to_csv(output / "endpoint_summary.csv", index=False)
    expected = pd.read_csv(root / "data/endpoint_summary.csv").set_index("architecture")
    for x, y in [
        ("original", "original_delta"),
        ("second_seed", "second_seed_delta"),
        ("seed_average", "seed_average_delta"),
    ]:
        np.testing.assert_allclose(
            summ.set_index("architecture")[x].values, expected.loc[ORDER, y].values, rtol=0, atol=1e-12
        )
    # Original intervals are recorded conditional-bootstrap outputs, not re-estimated from aggregate data.
    ci = pd.read_csv(root / "data/original_endpoint_intervals.csv")
    assert len(ci) == 4
    ci.to_csv(output / "original_conditional_intervals_archived.csv", index=False)
    m = pd.read_csv(root / "data/diagnosis_matched_representation_summary.csv")
    assert m.shape[0] == 4
    m.to_csv(output / "diagnosis_matched_representation.csv", index=False)
    for name in ["content_audit_main_table.csv", "content_audit_fold_summary.csv", "content_audit_summary.csv"]:
        p = root / "data" / name
        if p.exists():
            pd.read_csv(p).to_csv(output / name, index=False)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=Path("outputs/derived"))
    args = ap.parse_args()
    print(reproduce(args.repo, args.out).to_string(index=False))
