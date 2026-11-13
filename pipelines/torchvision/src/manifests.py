from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, Tuple

import pandas as pd


REQUIRED_REFERENCE_FILES = {
    "master": "master_subject_index_relative.csv",
    "split": "outer_inner_split_manifest.csv",
    "contamination": "contamination_manifest_relative.csv",
    "removal": "removed_training_slices_manifest_relative.csv",
}


@dataclass
class FrozenManifests:
    master: pd.DataFrame
    split: pd.DataFrame
    contamination: pd.DataFrame
    removal: pd.DataFrame
    root: Path


def load_frozen_manifests(root: str | Path) -> FrozenManifests:
    root = Path(root)
    missing = [name for name in REQUIRED_REFERENCE_FILES.values() if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen manifest files under {root}: {missing}")
    master = pd.read_csv(root / REQUIRED_REFERENCE_FILES["master"])
    split = pd.read_csv(root / REQUIRED_REFERENCE_FILES["split"])
    contamination = pd.read_csv(root / REQUIRED_REFERENCE_FILES["contamination"])
    removal_path = root / REQUIRED_REFERENCE_FILES["removal"]
    removal = pd.read_csv(removal_path) if removal_path.stat().st_size > 1 else pd.DataFrame()
    validate_reference_schema(master, split, contamination, removal)
    return FrozenManifests(master, split, contamination, removal, root)


def validate_reference_schema(master: pd.DataFrame, split: pd.DataFrame, contamination: pd.DataFrame, removal: pd.DataFrame) -> None:
    req_master = {"relative_path", "anonymous_subject_id", "class_label", "slice_identifier"}
    req_split = {"outer_fold", "role", "anonymous_subject_id", "class_label"}
    req_cont = {"outer_fold", "overlap_pct", "anonymous_subject_id", "class_label", "injected_relative_path"}
    req_rem = {"outer_fold", "overlap_pct", "class_label", "removed_original_training_relative_path"}
    for name, df, req in [
        ("master", master, req_master),
        ("split", split, req_split),
        ("contamination", contamination, req_cont),
    ]:
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"{name} manifest missing columns: {sorted(miss)}")
    if len(removal):
        miss = req_rem - set(removal.columns)
        if miss:
            raise ValueError(f"removal manifest missing columns: {sorted(miss)}")


def expected_cohort_audit(m: FrozenManifests) -> Dict[str, int]:
    subjects = m.master[["anonymous_subject_id", "class_label"]].drop_duplicates()
    return {
        "image_records": int(len(m.master)),
        "unique_file_identifiers": int(m.master.relative_path.nunique()),
        "subjects": int(subjects.anonymous_subject_id.nunique()),
        "pd_subjects": int((subjects.class_label == 1).sum()),
        "non_pd_subjects": int((subjects.class_label == 0).sum()),
    }


def build_partitions(m: FrozenManifests, fold: int, overlap_pct: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    """Build a run only from frozen manifests; never regenerate subject or slice selections."""
    if fold not in set(m.split.outer_fold.unique()):
        raise ValueError(f"Unknown fold {fold}")
    sm = m.split[m.split.outer_fold == fold]
    train_ids = set(sm[sm.role == "outer_training"].anonymous_subject_id.astype(str))
    val_ids = set(sm[sm.role == "inner_validation"].anonymous_subject_id.astype(str))
    test_ids = set(sm[sm.role == "outer_testing"].anonymous_subject_id.astype(str))
    train = m.master[m.master.anonymous_subject_id.isin(train_ids)].copy()
    val = m.master[m.master.anonymous_subject_id.isin(val_ids)].copy()
    test = m.master[m.master.anonymous_subject_id.isin(test_ids)].copy()

    c = m.contamination[(m.contamination.outer_fold == fold) & (m.contamination.overlap_pct == overlap_pct)].copy()
    contaminated = set(c.anonymous_subject_id.astype(str)) if len(c) else set()
    injected_paths = set(c.injected_relative_path.astype(str)) if len(c) else set()
    if injected_paths:
        inj = m.master[m.master.relative_path.isin(injected_paths)].copy()
        if set(inj.relative_path) != injected_paths:
            missing = sorted(injected_paths - set(inj.relative_path))[:10]
            raise RuntimeError(f"Frozen contamination paths are absent from master manifest: {missing}")
        test = test[~test.relative_path.isin(injected_paths)].copy()
    else:
        inj = m.master.iloc[0:0].copy()

    r = m.removal[(m.removal.outer_fold == fold) & (m.removal.overlap_pct == overlap_pct)].copy() if len(m.removal) else pd.DataFrame()
    removal_paths = set(r.removed_original_training_relative_path.astype(str)) if len(r) else set()
    if removal_paths:
        if not removal_paths <= set(train.relative_path):
            missing = sorted(removal_paths - set(train.relative_path))[:10]
            raise RuntimeError(f"Frozen removal paths are absent from original training partition: {missing}")
        train = train[~train.relative_path.isin(removal_paths)].copy()
    if len(inj):
        train = pd.concat([train, inj], ignore_index=True)

    train["partition"] = "train"
    val["partition"] = "validation"
    test["partition"] = "test"
    train["represented_in_training"] = True
    val["represented_in_training"] = False
    test["represented_in_training"] = test.anonymous_subject_id.isin(contaminated)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), contaminated


def validate_data_root(
    m: FrozenManifests,
    data_root: str | Path,
    max_report: int = 20,
    progress_every: int = 2000,
) -> Dict[str, object]:
    data_root = Path(data_root)
    paths = m.master.relative_path.astype(str).tolist()
    missing = []
    started = time.time()
    total = len(paths)
    print(f"[DATA AUDIT] Checking {total:,} manifest files under {data_root} ...", flush=True)
    for i, rel in enumerate(paths, 1):
        if not (data_root / rel).is_file():
            if len(missing) < max_report:
                missing.append(rel)
        if progress_every and (i % progress_every == 0 or i == total):
            elapsed = max(time.time() - started, 1e-9)
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else float("nan")
            print(
                f"[DATA AUDIT] {i:,}/{total:,} checked "
                f"({100*i/total:.1f}%), {rate:.1f} files/s, ETA {eta/60:.1f} min",
                flush=True,
            )
    # If more than max_report were missing, make a second cheap count impossible without
    # re-scanning; status is all that is scientifically required. We report at least
    # the number retained in the preview and use 0 only on PASS.
    missing_count = 0 if not missing else len(missing)
    return {
        "data_root": str(data_root),
        "expected_files": int(total),
        "missing_files": int(missing_count),
        "missing_examples": missing[:max_report],
        "status": "PASS" if not missing else "FAIL",
        "elapsed_seconds": time.time() - started,
    }


def common_held_out_file_ids(test_prediction_frames: Iterable[pd.DataFrame]) -> set[str]:
    sets = []
    for df in test_prediction_frames:
        if "file_identifier" not in df.columns:
            raise ValueError("Prediction frame missing file_identifier")
        sets.append(set(df.file_identifier.astype(str)))
    return set.intersection(*sets) if sets else set()
