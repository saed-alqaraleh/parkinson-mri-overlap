from __future__ import annotations

from typing import Dict, Iterable, Sequence

import pandas as pd

from .manifests import FrozenManifests, build_partitions, expected_cohort_audit


def audit_reference_manifests(m: FrozenManifests) -> Dict[str, object]:
    cohort = expected_cohort_audit(m)
    errors = []
    if cohort != {
        "image_records": 42147,
        "unique_file_identifiers": 42147,
        "subjects": 68,
        "pd_subjects": 48,
        "non_pd_subjects": 20,
    }:
        errors.append(f"cohort mismatch: {cohort}")

    if m.master.relative_path.duplicated().any():
        errors.append("duplicate relative_path in master manifest")
    if m.master.groupby("anonymous_subject_id").class_label.nunique().gt(1).any():
        errors.append("subject with conflicting diagnosis labels")

    nested = True
    for fold in sorted(m.split.outer_fold.unique()):
        sets = {
            ov: set(
                m.contamination[
                    (m.contamination.outer_fold == fold) & (m.contamination.overlap_pct == ov)
                ].anonymous_subject_id.astype(str)
            )
            for ov in [0, 25, 50, 75, 100]
        }
        expected_test = set(
            m.split[(m.split.outer_fold == fold) & (m.split.role == "outer_testing")].anonymous_subject_id.astype(str)
        )
        if not (sets[0] == set() and sets[25] <= sets[50] <= sets[75] <= sets[100] == expected_test):
            nested = False
            errors.append(f"nested overlap set violation in fold {fold}")

        # Same injected file set for a subject at every higher dose where selected.
        for sid in expected_test:
            seen = []
            for ov in [25, 50, 75, 100]:
                z = m.contamination[
                    (m.contamination.outer_fold == fold)
                    & (m.contamination.overlap_pct == ov)
                    & (m.contamination.anonymous_subject_id.astype(str) == sid)
                ]
                if len(z):
                    seen.append(set(z.injected_relative_path.astype(str)))
            if len(seen) > 1 and any(s != seen[0] for s in seen[1:]):
                errors.append(f"non-identical injected slices across doses fold={fold} subject={sid}")

    return {"status": "PASS" if not errors else "FAIL", "cohort": cohort, "nested_overlap_pass": nested, "errors": errors}


def audit_partition(
    m: FrozenManifests,
    fold: int,
    overlap_pct: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    contaminated: set[str],
) -> Dict[str, object]:
    ptrain, pval, ptest = set(train.relative_path), set(val.relative_path), set(test.relative_path)
    strain, sval, stest = set(train.anonymous_subject_id.astype(str)), set(val.anonymous_subject_id.astype(str)), set(test.anonymous_subject_id.astype(str))
    expected_test = set(
        m.split[(m.split.outer_fold == fold) & (m.split.role == "outer_testing")].anonymous_subject_id.astype(str)
    )
    expected_train_test = set(contaminated)
    original_train_ids = set(
        m.split[(m.split.outer_fold == fold) & (m.split.role == "outer_training")].anonymous_subject_id.astype(str)
    )
    original_val_ids = set(
        m.split[(m.split.outer_fold == fold) & (m.split.role == "inner_validation")].anonymous_subject_id.astype(str)
    )

    train_cls = train.class_label.value_counts().to_dict()
    base_train, _, _, _ = build_partitions(m, fold, 0)
    base_cls = base_train.class_label.value_counts().to_dict()

    checks = {
        "path_train_val_overlap": len(ptrain & pval),
        "path_train_test_overlap": len(ptrain & ptest),
        "path_val_test_overlap": len(pval & ptest),
        "subject_train_val_overlap": len(strain & sval),
        "subject_val_test_overlap": len(sval & stest),
        "subject_train_test_overlap": len(strain & stest),
        "expected_train_test_subject_overlap": len(expected_train_test),
        "train_test_overlap_exact_match": (strain & stest) == expected_train_test,
        "test_subject_identity_set_preserved": stest == expected_test,
        "validation_subject_identity_set_preserved": sval == original_val_ids,
        "original_train_subjects_preserved_except_intentional_test_additions": original_train_ids <= strain,
        "train_slice_budget_equal_to_zero_overlap": len(train) == len(base_train),
        "class_specific_train_slice_budget_equal": train_cls == base_cls,
    }
    pass_bool = (
        checks["path_train_val_overlap"] == 0
        and checks["path_train_test_overlap"] == 0
        and checks["path_val_test_overlap"] == 0
        and checks["subject_train_val_overlap"] == 0
        and checks["subject_val_test_overlap"] == 0
        and checks["train_test_overlap_exact_match"]
        and checks["test_subject_identity_set_preserved"]
        and checks["validation_subject_identity_set_preserved"]
        and checks["original_train_subjects_preserved_except_intentional_test_additions"]
        and checks["train_slice_budget_equal_to_zero_overlap"]
        and checks["class_specific_train_slice_budget_equal"]
    )
    return {
        "outer_fold": int(fold),
        "overlap_pct": int(overlap_pct),
        "actual_overlap_pct": 100.0 * len(expected_train_test) / len(expected_test),
        "train_subjects": len(strain),
        "validation_subjects": len(sval),
        "test_subjects": len(stest),
        "train_slices": len(train),
        "validation_slices": len(val),
        "test_slices": len(test),
        "contaminated_subject_ids": "|".join(sorted(expected_train_test)),
        **checks,
        "audit_status": "PASS" if pass_bool else "FAIL",
    }


def assert_partition_pass(audit: Dict[str, object]) -> None:
    if audit.get("audit_status") != "PASS":
        raise RuntimeError(f"Partition audit failed: {audit}")


def audit_all_reference_partitions(m: FrozenManifests, overlaps: Sequence[int] = (0, 25, 50, 75, 100)) -> pd.DataFrame:
    rows = []
    for fold in sorted(m.split.outer_fold.unique()):
        for ov in overlaps:
            train, val, test, contaminated = build_partitions(m, int(fold), int(ov))
            rows.append(audit_partition(m, int(fold), int(ov), train, val, test, contaminated))
    return pd.DataFrame(rows)
