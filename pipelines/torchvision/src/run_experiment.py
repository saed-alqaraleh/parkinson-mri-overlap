from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch

from .audits import assert_partition_pass, audit_partition, audit_reference_manifests
from .data import MRISliceDataset, collate_with_metadata
from .evaluate import aggregate_patient, annotate_test_predictions, metric_row, predict_loader
from .manifests import build_partitions, load_frozen_manifests, validate_data_root
from .models import assert_official_pretrained_config, build_model, get_model_spec, model_metadata
from .thresholds import audit_selected_threshold, select_validation_threshold
from .train import make_loader, suggested_micro_batch_sizes, train_two_stage
from .utils import (
    FileLease,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_hash,
    environment_report,
    audit_io_roots,
    cuda_health_check,
    freeze_framework_environment,
    load_yaml,
    now_utc_iso,
    run_key,
    seed_everything,
    sha256_file,
    validate_completed_run,
)


FOLD_BASE_SEEDS = {0: 42, 1: 101, 2: 202, 3: 303, 4: 404}


def resolve_seed(fold: int, replicate: int, offsets: list[int]) -> int:
    if replicate < 0 or replicate >= len(offsets):
        raise ValueError("replicate index outside configured offsets")
    return int(FOLD_BASE_SEEDS[int(fold)] + int(offsets[replicate]))


def freeze_experiment_plan(config: dict, config_path: Path, output_root: Path) -> tuple[dict, str, Path]:
    plan = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "architectures": list(config["architectures"]),
        "folds": list(map(int, config["folds"])),
        "overlap_levels": list(map(int, config["overlap_levels"])),
        "replicates": int(config["replicates"]),
        "replicate_seed_offsets": list(map(int, config["replicate_seed_offsets"])),
        "fold_base_seeds": FOLD_BASE_SEEDS,
        "training": config["training"],
        "analysis": config["analysis"],
        "preprocessing": config["preprocessing"],
        "pretrained": bool(config.get("pretrained", True)),
        "source_config_sha256": sha256_file(config_path),
    }
    plan_hash = canonical_json_hash(plan)
    plan["plan_sha256"] = plan_hash
    plan_path = output_root / "FROZEN_EXPERIMENT_PLAN.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old.get("plan_sha256") != plan_hash:
            raise RuntimeError(
                f"Frozen plan exists but does not match current configuration. Refusing to alter official plan: {plan_path}"
            )
    else:
        atomic_write_json(plan_path, plan)
        (output_root / "FROZEN_EXPERIMENT_PLAN.sha256").write_text(plan_hash + "\n")
    return plan, plan_hash, plan_path


def make_prediction_loaders(train, val, test, data_root, arch, seed, micro_batch, workers):
    # The entire DATA_ROOT has already passed the official top-level data audit.
    # Do not repeat tens of thousands of remote filesystem metadata calls for every run.
    train_ds = MRISliceDataset(train, data_root, arch, True, seed, verify_files=False)
    val_ds = MRISliceDataset(val, data_root, arch, False, seed, verify_files=False)
    test_ds = MRISliceDataset(test, data_root, arch, False, seed, verify_files=False)
    return (
        make_loader(train_ds, micro_batch, True, workers, collate_with_metadata, seed),
        make_loader(val_ds, micro_batch, False, workers, collate_with_metadata, seed),
        make_loader(test_ds, micro_batch, False, workers, collate_with_metadata, seed),
    )


def _run_config_hash(payload: dict) -> str:
    return canonical_json_hash(payload)


def execute_single_run(
    manifests,
    data_root: Path,
    output_root: Path,
    arch: str,
    fold: int,
    overlap: int,
    replicate: int,
    seed: int,
    config: dict,
    plan_hash: str,
    device: torch.device,
) -> dict:
    spec = get_model_spec(arch)
    run_payload = {
        "architecture": arch,
        "outer_fold": fold,
        "overlap_pct": overlap,
        "training_replicate": replicate,
        "training_seed": seed,
        "plan_hash": plan_hash,
        "model_weight_identifier": spec.weight_identifier,
        "training": config["training"],
        "preprocessing": config["preprocessing"],
    }
    config_hash = _run_config_hash(run_payload)
    rk = run_key(arch, fold, overlap, replicate, seed)
    run_dir = output_root / "runs" / rk
    ok, why = validate_completed_run(run_dir, config_hash) if run_dir.exists() else (False, "missing")
    if ok:
        print(f"SKIP valid completed run: {rk}", flush=True)
        return json.loads((run_dir / "RUN_COMPLETE.json").read_text())
    if run_dir.exists() and any(run_dir.iterdir()):
        quarantine = run_dir.with_name(run_dir.name + "__INCOMPLETE__" + str(int(time.time())))
        os.replace(run_dir, quarantine)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "run_configuration.json", {**run_payload, "config_hash": config_hash})
    atomic_write_json(run_dir / "RUN_STATUS.json", {
        "status": "RUNNING", "run_key": rk, "started_utc": now_utc_iso(),
        "config_hash": config_hash, "plan_hash": plan_hash,
    })

    with FileLease(run_dir / "RUNNING.lock"):
        train, val, test, contaminated = build_partitions(manifests, fold, overlap)
        audit = audit_partition(manifests, fold, overlap, train, val, test, contaminated)
        atomic_write_csv(run_dir / "partition_audit.csv", pd.DataFrame([audit]))
        assert_partition_pass(audit)
        realized = float(audit["actual_overlap_pct"])

        # Persist the exact scientific partitions used by this run. These are
        # relative-path/anonymous-ID manifests only; no raw image bytes are copied.
        manifest_cols = ["relative_path", "anonymous_subject_id", "class_label", "partition", "represented_in_training"]
        atomic_write_csv(run_dir / "train_partition_manifest.csv", train[manifest_cols])
        atomic_write_csv(run_dir / "validation_partition_manifest.csv", val[manifest_cols])
        atomic_write_csv(run_dir / "test_partition_manifest.csv", test[manifest_cols])
        csub = manifests.contamination[(manifests.contamination.outer_fold == fold) & (manifests.contamination.overlap_pct == overlap)].copy()
        rsub = manifests.removal[(manifests.removal.outer_fold == fold) & (manifests.removal.overlap_pct == overlap)].copy()
        atomic_write_csv(run_dir / "contamination_manifest.csv", csub)
        atomic_write_csv(run_dir / "removed_training_slices_manifest.csv", rsub)

        print(
            f"[RUN] {rk}: train={len(train):,}, val={len(val):,}, test={len(test):,}, "
            f"realized overlap={realized:.2f}%",
            flush=True,
        )
        seed_report = seed_everything(seed, deterministic=True)
        atomic_write_json(run_dir / "determinism_report.json", seed_report)
        print(f"[RUN] {rk}: building {arch} and loading pretrained weights ...", flush=True)
        model = build_model(arch, pretrained=bool(config.get("pretrained", True)))
        print(f"[RUN] {rk}: model ready; beginning training.", flush=True)
        atomic_write_json(run_dir / "model_metadata.json", model_metadata(model, arch))
        model.to(device)

        tcfg = config["training"]
        workers = int(tcfg.get("num_workers", 2))
        effective = int(tcfg["effective_batch_size"])
        use_amp = bool(tcfg.get("mixed_precision", True))
        selected_micro = None
        training_summary = None
        last_oom = None
        for micro in suggested_micro_batch_sizes(effective):
            if micro > int(tcfg.get("max_micro_batch_size", effective)):
                continue
            try:
                train_loader, val_loader, test_loader = make_prediction_loaders(
                    train, val, test, data_root, arch, seed, micro, workers
                )
                training_summary = train_two_stage(
                    model,
                    arch,
                    train_loader,
                    val_loader,
                    device,
                    run_dir,
                    head_epochs=int(tcfg["head_epochs"]),
                    finetune_epochs=int(tcfg["finetune_epochs"]),
                    patience=int(tcfg["patience"]),
                    head_lr=float(tcfg["head_lr"]),
                    finetune_lr=float(tcfg["finetune_lr"]),
                    effective_batch_size=effective,
                    micro_batch_size=micro,
                    use_amp=use_amp,
                )
                selected_micro = micro
                break
            except torch.cuda.OutOfMemoryError as e:  # pragma: no cover - requires GPU pressure
                last_oom = repr(e)
                print(f"OOM with micro-batch {micro}; retrying with a smaller micro-batch while keeping effective batch {effective}.", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Rebuild model from pretrained initialization with the same seed so failed attempts cannot alter the official run.
                seed_everything(seed, deterministic=True)
                model = build_model(arch, pretrained=bool(config.get("pretrained", True))).to(device)
        if training_summary is None:
            raise RuntimeError(f"Unable to train with effective batch {effective}; last OOM={last_oom}")
        atomic_write_json(run_dir / "batching_report.json", {
            "effective_batch_size": effective,
            "micro_batch_size": selected_micro,
            "gradient_accumulation_steps": effective // selected_micro,
            "last_oom_before_success": last_oom,
        })

        # Loaders are reconstructed without training augmentation for validation/test predictions.
        _, val_loader, test_loader = make_prediction_loaders(
            train, val, test, data_root, arch, seed, selected_micro, workers
        )
        val_slice = predict_loader(model, val_loader, device, use_amp)
        val_patient = aggregate_patient(val_slice)
        threshold, curve = select_validation_threshold(val_patient)
        atomic_write_csv(run_dir / "validation_threshold_curve.csv", curve)
        if not audit_selected_threshold(curve, threshold):
            raise RuntimeError("Selected threshold failed validation-curve audit")
        atomic_write_json(run_dir / "selected_threshold.json", {
            "selected_threshold": threshold,
            "source": "inner-validation patient mean probabilities only",
            "test_used": False,
        })

        test_slice = predict_loader(model, test_loader, device, use_amp)
        test_slice = annotate_test_predictions(
            test_slice, arch, fold, replicate, seed, overlap, realized, contaminated,
            threshold, training_summary["selected_checkpoint_sha256"],
        )
        test_patient = aggregate_patient(test_slice)
        # Propagate run metadata to patient frame.
        for col, value in [
            ("architecture", arch), ("outer_fold", fold), ("training_replicate", replicate),
            ("training_seed", seed), ("overlap_pct", overlap), ("realized_overlap_pct", realized),
            ("selected_threshold", threshold),
        ]:
            test_patient[col] = value

        atomic_write_csv(run_dir / "file_predictions.csv", test_slice)
        atomic_write_csv(run_dir / "patient_predictions.csv", test_patient)
        atomic_write_csv(run_dir / "validation_file_predictions.csv", val_slice)
        atomic_write_csv(run_dir / "validation_patient_predictions.csv", val_patient)

        rows = [
            metric_row(test_slice, threshold, "slice", arch, fold, overlap, replicate, seed, "all"),
            metric_row(test_patient, threshold, "patient", arch, fold, overlap, replicate, seed, "all"),
        ]
        if overlap > 0:
            seen_s = test_slice[test_slice.anonymous_subject_id.isin(contaminated)]
            unseen_s = test_slice[~test_slice.anonymous_subject_id.isin(contaminated)]
            seen_p = aggregate_patient(seen_s) if len(seen_s) else pd.DataFrame()
            unseen_p = aggregate_patient(unseen_s) if len(unseen_s) else pd.DataFrame()
            rows.extend([
                metric_row(seen_p, threshold, "patient", arch, fold, overlap, replicate, seed, "seen"),
                metric_row(unseen_p, threshold, "patient", arch, fold, overlap, replicate, seed, "unseen"),
            ])
        atomic_write_csv(run_dir / "metrics.csv", pd.DataFrame(rows))

        summary = {
            "architecture": arch,
            "outer_fold": fold,
            "overlap_pct": overlap,
            "realized_overlap_pct": realized,
            "training_replicate": replicate,
            "training_seed": seed,
            "selected_threshold": threshold,
            "selected_stage": training_summary["selected_stage"],
            "selected_epoch": training_summary["selected_epoch"],
            "selected_val_loss": training_summary["selected_val_loss"],
            "config_hash": config_hash,
            "plan_hash": plan_hash,
            "completed_utc": now_utc_iso(),
        }
        required_names = [
            "run_configuration.json", "partition_audit.csv", "model_metadata.json",
            "train_partition_manifest.csv", "validation_partition_manifest.csv", "test_partition_manifest.csv",
            "contamination_manifest.csv", "removed_training_slices_manifest.csv",
            "training_summary.json", "stage1_history.csv", "stage2_history.csv",
            "selected_threshold.json", "validation_threshold_curve.csv",
            "validation_file_predictions.csv", "validation_patient_predictions.csv",
            "file_predictions.csv", "patient_predictions.csv", "metrics.csv",
            "batching_report.json", "determinism_report.json",
        ]
        required_hashes = {name: sha256_file(run_dir / name) for name in required_names}
        marker = {**summary, "required_outputs": required_hashes}
        atomic_write_json(run_dir / "RUN_COMPLETE.json", marker)
        ok2, why2 = validate_completed_run(run_dir, config_hash)
        if not ok2:
            (run_dir / "RUN_COMPLETE.json").unlink(missing_ok=True)
            atomic_write_json(run_dir / "RUN_STATUS.json", {
                "status": "FAILED_INTEGRITY", "run_key": rk, "finished_utc": now_utc_iso(),
                "config_hash": config_hash, "reason": why2,
            })
            raise RuntimeError(f"Post-completion integrity check failed: {why2}")
        atomic_write_json(run_dir / "RUN_STATUS.json", {
            "status": "COMPLETE", "run_key": rk, "finished_utc": now_utc_iso(),
            "config_hash": config_hash, "plan_hash": plan_hash,
            "selected_checkpoint_sha256": training_summary["selected_checkpoint_sha256"],
        })
        return marker


def execute_config(config_path: str | Path, data_root: str | Path, output_root: str | Path, manifest_root: str | Path) -> None:
    config_path = Path(config_path)
    data_root = Path(data_root)
    base_output = Path(output_root)

    # DATA_ROOT may be a read-only mounted or shared read-only folder.
    # No output, cache, checkpoint, lock, or temporary result is ever written there.
    io_audit = audit_io_roots(data_root, base_output)
    if io_audit["status"] != "PASS":
        raise RuntimeError(
            "Read-only DATA_ROOT / writable OUTPUT_ROOT preflight failed. "
            f"Details: {io_audit}"
        )

    print("[OFFICIAL] Loading frozen configuration and manifests ...", flush=True)
    config = load_yaml(config_path)
    assert_official_pretrained_config(config)
    manifests = load_frozen_manifests(manifest_root)
    reference_audit = audit_reference_manifests(manifests)
    if reference_audit["status"] != "PASS":
        raise RuntimeError(f"Frozen reference manifest audit failed: {reference_audit}")
    print("[OFFICIAL] Starting one-time DATA_ROOT integrity audit.", flush=True)
    data_audit = validate_data_root(manifests, data_root)
    print(f"[OFFICIAL] DATA_ROOT audit status: {data_audit['status']}", flush=True)
    if data_audit["status"] != "PASS":
        raise RuntimeError(
            "Raw MRI data are not available at the configured DATA_ROOT with the frozen relative paths. "
            f"Missing {data_audit['missing_files']} of {data_audit['expected_files']}. "
            f"Examples: {data_audit['missing_examples'][:5]}"
        )
    root = base_output / config["experiment_name"]
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "io_root_audit.json", io_audit)
    env = environment_report()
    atomic_write_json(root / "environment_report.json", env)
    cuda_report = cuda_health_check(require_cuda=bool(config.get("require_cuda", False)))
    atomic_write_json(root / "cuda_health_report.json", cuda_report)
    if cuda_report["status"] == "FAIL":
        raise RuntimeError(
            "CUDA/cuDNN health gate failed before official training. Do not continue. "
            f"Report: {cuda_report}"
        )
    # Freeze the framework environment on the first run and reject incompatible resumes.
    freeze_framework_environment(root)
    atomic_write_json(root / "reference_manifest_audit.json", reference_audit)
    atomic_write_json(root / "data_root_audit.json", data_audit)
    plan, plan_hash, plan_path = freeze_experiment_plan(config, config_path, root)
    os.environ.setdefault("TORCH_HOME", str(base_output / "pretrained_cache"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = []
    for arch in plan["architectures"]:
        for fold in plan["folds"]:
            for rep in range(plan["replicates"]):
                seed = resolve_seed(fold, rep, plan["replicate_seed_offsets"])
                for ov in plan["overlap_levels"]:
                    jobs.append((arch, fold, ov, rep, seed))
    print(f"Frozen plan: {plan_path}\nPlan hash: {plan_hash}\nTotal scheduled jobs: {len(jobs)}", flush=True)
    failures = []
    experiment_started = time.time()
    for i, (arch, fold, ov, rep, seed) in enumerate(jobs, 1):
        job_started = time.time()
        print(f"[{i}/{len(jobs)}] {arch} fold={fold} overlap={ov} replicate={rep} seed={seed}", flush=True)
        try:
            execute_single_run(manifests, data_root, root, arch, fold, ov, rep, seed, config, plan_hash, device)
        except Exception as e:
            err = {"architecture": arch, "fold": fold, "overlap": ov, "replicate": rep, "seed": seed, "error": repr(e), "failed_utc": now_utc_iso()}
            failures.append(err)
            atomic_write_csv(root / "failed_runs.csv", pd.DataFrame(failures))
            rk = run_key(arch, fold, ov, rep, seed)
            rd = root / "runs" / rk
            if rd.exists():
                atomic_write_json(rd / "RUN_STATUS.json", {
                    "status": "FAILED", "run_key": rk, "failed_utc": now_utc_iso(),
                    "plan_hash": plan_hash, "error": repr(e),
                })
            raise
        elapsed = time.time() - experiment_started
        avg = elapsed / i
        eta = avg * (len(jobs) - i)
        print(f"Completed {i}/{len(jobs)}; last job {time.time()-job_started:.1f}s; estimated remaining {eta/3600:.2f}h.", flush=True)
    if failures:
        atomic_write_csv(root / "failed_runs.csv", pd.DataFrame(failures))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run frozen contemporary Parkinson overlap baselines")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--manifest-root", required=True)
    args = ap.parse_args(argv)
    execute_config(args.config, args.data_root, args.output_root, args.manifest_root)


if __name__ == "__main__":
    main()
