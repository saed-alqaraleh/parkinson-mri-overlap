"""Independent release checks for the frozen cohort and reported analyses."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.regenerate_partitions import build, records
from scripts.reproduce_full_statistics import reproduce

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_source_inventory_and_all_partitions() -> None:
    data = ROOT / "data"
    inventory = records(data)
    assert len(inventory) == 42147
    assert inventory.subject_token.nunique() == 68
    subjects = inventory[["subject_token", "class_label"]].drop_duplicates()
    assert subjects.class_label.value_counts().to_dict() == {1: 48, 0: 20}
    assert inventory.dataset_relative_path.str.match(r"^(?:PD Patients|Non PD Patients)/").all()
    assert not inventory.dataset_relative_path.str.startswith("/").any()
    all_ops = build(data)
    frozen = pd.read_csv(data / "frozen_operations.csv.gz")
    cols = list(frozen.columns)
    pd.testing.assert_frame_equal(
        all_ops[cols].sort_values(cols).reset_index(drop=True),
        frozen[cols].sort_values(cols).reset_index(drop=True),
    )
    assert len(frozen) == 15710
    assert len(frozen.query("overlap_pct == 100")) == 6318


def test_released_scores_95_cases_and_interval_regeneration(tmp_path: Path) -> None:
    data = ROOT / "data"
    a = pd.read_csv(data / "patient_scores_original.csv")
    b = pd.read_csv(data / "patient_scores_second_seed.csv")
    assert a.groupby(["architecture", "outer_fold", "overlap_pct"]).ngroups == 55
    assert b.groupby(["architecture", "outer_fold", "overlap_pct"]).ngroups == 40
    for scores in (a, b):
        assert scores.subject_token.str.fullmatch(r"P\d{3}").all()
        assert not scores.duplicated(["architecture", "outer_fold", "overlap_pct", "subject_token"]).any()
        assert scores.probability.between(0, 1).all()
    results = reproduce(data, tmp_path)
    original = results["intervals"].query('scope == "original"').set_index("architecture")
    slope = results["slope"].iloc[0]
    assert np.isclose(original.loc["ResNet50V2", "mean_delta"], 0.075, atol=1e-9)
    assert np.isclose(slope.roc_auc_slope, 0.097996, atol=0.00001)
    assert np.isclose(slope.ci_low, 0.0170954205436964, atol=1e-12)
    assert np.isclose(slope.ci_high, 0.1924384692574348, atol=1e-12)
    # The manuscript's four-fold, two-seed point estimates, not just the five-fold series.
    four = results["intervals"].query('scope == "seed_average_excluding_fold4"').set_index("architecture")
    for model, expected in (
        ("ResNet50V2", 0.086),
        ("VGG19", 0.168),
        ("ConvNeXt-Tiny", 0.032),
        ("Swin-Tiny", 0.037),
    ):
        assert np.isclose(four.loc[model, "mean_delta"], expected, atol=0.0005)
    assert all(four.n_folds == 4)
    # All six printed interval bounds differing by up to 0.003 are compared
    # against independently regenerated estimates inside reproduce().
    assert len(pd.read_csv(data / "reference_table5_intervals.csv")) == 6


def test_recorded_batch_augmentation_pipeline_structure() -> None:
    # Python-version-independent structural assertions: ast.dump() can change
    # across interpreter versions without a change in the training program.
    source = ROOT / "pipelines/keras/runner_batched.py"
    script = source.read_text(encoding="utf-8")
    program = ast.parse(script)
    make_ds = next(node for node in program.body if isinstance(node, ast.FunctionDef) and node.name == "make_ds")
    body = ast.unparse(make_ds)
    assert "ds = ds.shuffle(min(len(df), 10000), seed=int(run_seed), reshuffle_each_iteration=True)" in body
    assert "ds = ds.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE)" in body
    assert "options.experimental_deterministic = True" in body
    assert "options.threading.max_intra_op_parallelism = 0" in body
    assert "ds = ds.with_options(options)" in body
    assert "ds = ds.batch(batch_size).map(transform, num_parallel_calls=1).prefetch(tf.data.AUTOTUNE)" in body
    assert "x = augment(x, training=True)" in body
    assert "if TARGET_OUTER_FOLD not in (2, 3, 4)" in script
    assert "PDMRI_SEED_OFFSET" in script
