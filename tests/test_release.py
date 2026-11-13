import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines/torchvision"))


def mock_manifest():
    from src.manifests import FrozenManifests

    master = []
    split = []
    cont = []
    rem = []
    for role, subject, dx in [
        ("outer_training", "A", 0),
        ("outer_training", "B", 1),
        ("inner_validation", "V0", 0),
        ("inner_validation", "V1", 1),
        ("outer_testing", "T0", 0),
        ("outer_testing", "T1", 1),
    ]:
        split.append(dict(outer_fold=0, role=role, anonymous_subject_id=subject, class_label=dx))
        for j in range(10):
            master.append(
                dict(
                    relative_path=f"{subject}/{j}.png", anonymous_subject_id=subject, class_label=dx, slice_identifier=j
                )
            )
    # On the 100% tier, one test image per test subject enters training; one same-label training image is removed.
    for subject, dx, base in [("T0", 0, "A"), ("T1", 1, "B")]:
        cont.append(
            dict(
                outer_fold=0,
                overlap_pct=100,
                anonymous_subject_id=subject,
                class_label=dx,
                injected_relative_path=f"{subject}/0.png",
            )
        )
        rem.append(
            dict(outer_fold=0, overlap_pct=100, class_label=dx, removed_original_training_relative_path=f"{base}/0.png")
        )
    return FrozenManifests(pd.DataFrame(master), pd.DataFrame(split), pd.DataFrame(cont), pd.DataFrame(rem), ROOT)


def test_fold_disjointness_and_overlap():
    from src.manifests import build_partitions
    from src.audits import audit_partition

    m = mock_manifest()
    for overlap, n in [(0, 0), (100, 2)]:
        tr, val, te, ids = build_partitions(m, 0, overlap)
        assert len(ids) == n
        assert set(tr.relative_path).isdisjoint(set(val.relative_path))
        assert set(tr.relative_path).isdisjoint(set(te.relative_path))
        assert set(val.anonymous_subject_id).isdisjoint(set(te.anonymous_subject_id))
        assert audit_partition(m, 0, overlap, tr, val, te, ids)["audit_status"] == "PASS"
        assert tr.class_label.value_counts().to_dict() == {0: 10, 1: 10}
        assert len(tr) == 20


def test_subject_mean_and_auc():
    frame = pd.DataFrame(
        {
            "subject": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "truth": [0, 0, 1, 1, 0, 0, 1, 1],
            "p": [0.1, 0.3, 0.8, 0.6, 0.2, 0.4, 0.7, 0.9],
        }
    )
    s = frame.groupby("subject").agg(truth=("truth", "first"), p=("p", "mean"))
    assert np.isclose(s.loc["a", "p"], 0.2)
    assert roc_auc_score(s.truth, s.p) == 1.0


def test_reported_deltas(tmp_path):
    from scripts.reproduce_public_results import reproduce

    d = reproduce(ROOT, tmp_path)
    assert np.allclose(d.original.to_numpy(), [0.075, 0.160, 0.048, 0.023], atol=0.0015)
    assert np.allclose(d.second_seed.to_numpy(), [0.135, 0.181, 0.053, 0.019], atol=0.0015)
    assert np.allclose(d.seed_average.to_numpy(), [0.105, 0.170, 0.050, 0.021], atol=0.0015)


def test_diagnosis_matched_budget_and_subject_cap():
    from pipelines.protocol_checks import inspect_frozen_assignments

    source = mock_manifest()
    # Extend synthetic records to all five tiers, keeping each test subject's chosen slice fixed.
    cont = []
    rem = []
    for level in (25, 50, 75, 100):
        for subject, dx, base in [("T0", 0, "A"), ("T1", 1, "B")][: 1 if level < 75 else 2]:
            cont.append(
                dict(
                    outer_fold=0,
                    overlap_pct=level,
                    anonymous_subject_id=subject,
                    class_label=dx,
                    injected_relative_path=f"{subject}/0.png",
                )
            )
            rem.append(
                dict(
                    outer_fold=0,
                    overlap_pct=level,
                    class_label=dx,
                    removed_original_training_relative_path=f"{base}/0.png",
                )
            )
    c = pd.DataFrame(cont)
    r = pd.DataFrame(rem)
    result = inspect_frozen_assignments(source.master, source.split, c, r)
    assert result.shape[0] == 5
    assert result.set_index("overlap_pct").loc[100, "represented_test_subjects"] == 2
    # Three of ten held-out slices breaches the 20% cap (even though n-2 would allow it).
    extra = source.master.copy()
    bad = pd.concat(
        [
            c,
            pd.DataFrame(
                [
                    dict(
                        outer_fold=0,
                        overlap_pct=100,
                        anonymous_subject_id="T0",
                        class_label=0,
                        injected_relative_path=f"T0/{i}.png",
                    )
                    for i in [1, 2]
                ]
            ),
        ],
        ignore_index=True,
    )
    rr = pd.concat(
        [
            r,
            pd.DataFrame(
                [
                    dict(
                        outer_fold=0,
                        overlap_pct=100,
                        class_label=0,
                        removed_original_training_relative_path=f"A/{i}.png",
                    )
                    for i in [1, 2]
                ]
            ),
        ],
        ignore_index=True,
    )
    import pytest

    with pytest.raises(AssertionError, match="cap exceeded"):
        inspect_frozen_assignments(extra, source.split, bad, rr)


def test_diagnosis_mismatch_is_detected():
    from pipelines.protocol_checks import inspect_frozen_assignments

    source = mock_manifest()
    c = pd.concat([source.contamination.assign(overlap_pct=level) for level in (25, 50, 75, 100)], ignore_index=True)
    r = pd.concat([source.removal.assign(overlap_pct=level) for level in (25, 50, 75, 100)], ignore_index=True)
    r.loc[0, "removed_original_training_relative_path"] = "A/0.png"
    r.loc[1, "removed_original_training_relative_path"] = "A/1.png"
    import pytest

    with pytest.raises(AssertionError, match="Diagnosis-matched"):
        inspect_frozen_assignments(source.master, source.split, c, r)


def test_torch_architectures_synthetic_cpu_forward():
    import pytest
    import os

    if os.environ.get("PDMRI_TORCH_CPU_SMOKE") != "1":
        pytest.skip("Enable the full TorchVision CPU integration check in its framework environment")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.models import build_model

    torch.set_num_threads(2)
    x = torch.zeros(1, 3, 64, 64)
    for arch in ("convnext_tiny", "swin_tiny"):
        with torch.no_grad():
            output = build_model(arch, pretrained=False, progress=False).cpu().eval()(x)
        assert tuple(output.shape) == (1, 1)
        assert torch.isfinite(output).all()


def test_training_seed_separation_from_partitions():
    import ast
    import os

    source = (ROOT / "pipelines/keras/runner.py").read_text()
    tree = ast.parse(source)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "run_seed_for")
    namespace = {"MANUSCRIPT_FIXED_SEEDS": [42, 101, 202, 303, 404], "_runtime_os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT / "pipelines/keras/runner.py"), "exec"), namespace)
    prev = os.environ.get("PDMRI_SEED_OFFSET")
    try:
        for offset in (0, 100000):
            os.environ["PDMRI_SEED_OFFSET"] = str(offset)
            for fold, base in enumerate((42, 101, 202, 303, 404)):
                assert namespace["run_seed_for"](fold) == base + offset
    finally:
        if prev is None:
            os.environ.pop("PDMRI_SEED_OFFSET", None)
        else:
            os.environ["PDMRI_SEED_OFFSET"] = prev
