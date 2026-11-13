"""Deterministic integrity checks for authorized frozen image-record manifests.

The module checks saved assignments and does not create new subject splits.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class ProtocolLimits:
    max_per_subject: int = 50
    max_fraction: float = 0.2
    minimum_held_out: int = 2


def inspect_frozen_assignments(master: pd.DataFrame, split: pd.DataFrame,
                               injected: pd.DataFrame, removed: pd.DataFrame,
                               limits: ProtocolLimits = ProtocolLimits()) -> pd.DataFrame:
    """Raise on subject/diagnosis, budget, and nested-manifest violations."""
    master = master.copy()
    split = split.copy()
    injected = injected.copy()
    removed = removed.copy()
    for col in ('anonymous_subject_id', 'relative_path'):
        if master[col].isna().any():
            raise ValueError(f'Missing master {col}')
        master[col] = master[col].astype(str)
    assert not master.relative_path.duplicated().any(), 'Duplicate master path'
    assert master.groupby('anonymous_subject_id').class_label.nunique().eq(1).all(), 'Conflicting patient diagnoses'
    by_path = master.set_index('relative_path')
    subject_path_counts = master.groupby('anonymous_subject_id').size().to_dict()
    rows = []
    for fold in sorted(split.outer_fold.unique()):
        s = split.loc[split.outer_fold == fold].copy()
        roles = {role: set(s.loc[s.role == role, 'anonymous_subject_id'].astype(str))
                 for role in ('outer_training', 'inner_validation', 'outer_testing')}
        assert not (roles['outer_training'] & roles['inner_validation']), 'Training/validation identity intersection'
        assert not (roles['inner_validation'] & roles['outer_testing']), 'Validation/test identity intersection'
        assert not (roles['outer_training'] & roles['outer_testing']), 'Base training/test identity intersection'
        train_paths = set(master[master.anonymous_subject_id.isin(roles['outer_training'])].relative_path)
        test_paths = set(master[master.anonymous_subject_id.isin(roles['outer_testing'])].relative_path)
        val_paths = set(master[master.anonymous_subject_id.isin(roles['inner_validation'])].relative_path)
        previous_subjects: set[str] = set()
        previous_by_subject: dict[str,set[str]] = {}
        for level in (0,25,50,75,100):
            i = injected[(injected.outer_fold == fold) & (injected.overlap_pct == level)].copy()
            r = removed[(removed.outer_fold == fold) & (removed.overlap_pct == level)].copy()
            injected_paths = i.injected_relative_path.astype(str).tolist()
            removed_paths = r.removed_original_training_relative_path.astype(str).tolist()
            assert len(injected_paths) == len(set(injected_paths)), 'Repeated injection path'
            assert len(removed_paths) == len(set(removed_paths)), 'Repeated removal path'
            assert set(injected_paths) <= test_paths, 'Injected image did not come from outer test partition'
            assert set(removed_paths) <= train_paths, 'Removed image was not in initial training partition'
            assert not (set(injected_paths) & val_paths), 'Validation image used in injection'
            assert len(injected_paths) == len(removed_paths), 'Overall slice-budget mismatch'
            by_subject: dict[str,set[str]] = {}
            cls_in = {0:0,1:0}
            cls_out = {0:0,1:0}
            for ipath in injected_paths:
                meta = by_path.loc[ipath]
                sid = str(meta.anonymous_subject_id)
                assert sid in roles['outer_testing'], 'Injected subject is not an outer-test subject'
                by_subject.setdefault(sid, set()).add(ipath)
                cls_in[int(meta.class_label)] += 1
            for rpath in removed_paths:
                cls_out[int(by_path.loc[rpath].class_label)] += 1
            assert cls_in == cls_out, 'Diagnosis-matched replacement violates class budget'
            represented = set(by_subject)
            for sid, paths in by_subject.items():
                n = int(subject_path_counts[sid])
                cap = min(limits.max_per_subject, int(limits.max_fraction*n), n-limits.minimum_held_out)
                assert 0 < len(paths) <= cap, f'Per-subject injection cap exceeded for {sid}'
            assert previous_subjects <= represented, 'Overlap tiers are not nested'
            for sid in previous_subjects:
                assert previous_by_subject[sid] == by_subject[sid], 'Injected slices changed at higher tier'
            if level == 0:
                assert len(represented) == len(injected_paths) == 0, 'Base tier must be identity-disjoint'
            if level == 100:
                assert represented == roles['outer_testing'], '100% tier missing test identities'
            rows.append(dict(outer_fold=int(fold),overlap_pct=level,
                             represented_test_subjects=len(represented),
                             test_subjects=len(roles['outer_testing']),
                             injected_slices=len(injected_paths),removed_slices=len(removed_paths),
                             diagnosis_matched=True,budget_fixed=True,validation_disjoint=True))
            previous_subjects = represented
            previous_by_subject = by_subject
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Validate authorized frozen partitions and slice-budget invariants')
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/protocol_integrity.csv'))
    a = parser.parse_args()
    m = a.manifest_dir
    needed = ['master_subject_index_relative.csv','outer_inner_split_manifest.csv',
              'contamination_manifest_relative.csv','removed_training_slices_manifest_relative.csv']
    dfs = [pd.read_csv(m/x) for x in needed]
    result = inspect_frozen_assignments(*dfs)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    print(f'Validated {len(result)} fold × overlap assignments')

if __name__ == '__main__':
    main()
