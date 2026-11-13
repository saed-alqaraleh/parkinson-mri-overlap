from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .bootstrap import conditional_collection_bootstrap
from .evaluate import aggregate_patient, common_held_out_analysis, metric_row
from .figures import endpoint_paired_plot, full_overlap_curve_plot, roc_auc_by_overlap_plot
from .utils import atomic_write_csv, atomic_write_json, atomic_write_text, sha256_file, validate_completed_run


def _read_run_marker(marker: Path) -> dict:
    d = json.loads(marker.read_text(encoding="utf-8"))
    d["run_dir"] = str(marker.parent)
    return d


def _validated_markers(output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    markers = []
    for marker in _validated_markers(root):
        if "__INCOMPLETE__" in str(marker) or "INCOMPLETE_ARCHIVE" in str(marker):
            continue
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
            ok, _ = validate_completed_run(marker.parent, meta.get("config_hash"))
            if ok:
                markers.append(marker)
        except Exception:
            continue
    return sorted(markers)


def assert_frozen_plan_complete(output_root: str | Path) -> dict:
    """Refuse publication aggregation unless every frozen-plan run is integrity-valid."""
    root = Path(output_root)
    plan_path = root / "FROZEN_EXPERIMENT_PLAN.json"
    if not plan_path.exists():
        raise RuntimeError(f"Missing frozen experiment plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = {int(k): int(v) for k, v in plan["fold_base_seeds"].items()}
    offsets = [int(x) for x in plan["replicate_seed_offsets"]]
    expected = set()
    for arch in plan["architectures"]:
        for fold in map(int, plan["folds"]):
            for rep in range(int(plan["replicates"])):
                seed = base[fold] + offsets[rep]
                for ov in map(int, plan["overlap_levels"]):
                    expected.add((str(arch), fold, ov, rep, seed))
    observed = set()
    plan_hashes = set()
    for marker in _validated_markers(root):
        d = json.loads(marker.read_text(encoding="utf-8"))
        observed.add((str(d["architecture"]), int(d["outer_fold"]), int(d["overlap_pct"]), int(d["training_replicate"]), int(d["training_seed"])))
        plan_hashes.add(d.get("plan_hash"))
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"Frozen-plan completeness failed. Missing={missing}; extra={extra}")
    if plan_hashes != {plan.get("plan_sha256")}:
        raise RuntimeError(f"Run plan hashes do not match frozen plan: {plan_hashes} vs {plan.get('plan_sha256')}")
    return {
        "expected_runs": len(expected),
        "validated_completed_runs": len(observed),
        "plan_sha256": plan.get("plan_sha256"),
        "status": "PASS",
    }


def collect_run_records(output_root: str | Path) -> pd.DataFrame:
    rows = []
    for marker in _validated_markers(output_root):
        try:
            rows.append(_read_run_marker(marker))
        except Exception:
            continue
    return pd.DataFrame(rows)


def collect_completed_runs(output_root: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(output_root)
    metric_frames: List[pd.DataFrame] = []
    patient_frames: List[pd.DataFrame] = []
    training_rows: List[dict] = []
    audit_frames: List[pd.DataFrame] = []
    for marker in _validated_markers(root):
        run_dir = marker.parent
        try:
            meta = json.loads(marker.read_text())
        except Exception:
            continue
        mp = run_dir / "metrics.csv"
        pp = run_dir / "patient_predictions.csv"
        ap = run_dir / "partition_audit.csv"
        tsp = run_dir / "training_summary.json"
        if mp.exists():
            m = pd.read_csv(mp)
            m["run_dir"] = str(run_dir)
            metric_frames.append(m)
        if pp.exists():
            p = pd.read_csv(pp)
            if "architecture" not in p.columns:
                p["architecture"] = meta["architecture"]
            if "outer_fold" not in p.columns:
                p["outer_fold"] = meta["outer_fold"]
            if "training_replicate" not in p.columns:
                p["training_replicate"] = meta["training_replicate"]
            if "overlap_pct" not in p.columns:
                p["overlap_pct"] = meta["overlap_pct"]
            p["run_dir"] = str(run_dir)
            patient_frames.append(p)
        if ap.exists():
            a = pd.read_csv(ap)
            a["architecture"] = meta["architecture"]
            a["training_replicate"] = meta["training_replicate"]
            a["training_seed"] = meta["training_seed"]
            audit_frames.append(a)
        if tsp.exists():
            t = json.loads(tsp.read_text())
            bp = run_dir / "batching_report.json"
            batch = json.loads(bp.read_text()) if bp.exists() else {}
            training_rows.append({
                "architecture": meta["architecture"],
                "outer_fold": int(meta["outer_fold"]),
                "overlap_pct": int(meta["overlap_pct"]),
                "training_replicate": int(meta["training_replicate"]),
                "training_seed": int(meta["training_seed"]),
                "selected_stage": int(t["selected_stage"]),
                "selected_epoch": int(t["selected_epoch"]),
                "selected_val_loss": float(t["selected_val_loss"]),
                "stage1_epochs_executed": int(t["stage1"]["epochs_executed"]),
                "stage2_epochs_executed": int(t["stage2"]["epochs_executed"]),
                "stage1_duration_seconds": float(t["stage1"]["duration_seconds"]),
                "stage2_duration_seconds": float(t["stage2"]["duration_seconds"]),
                "training_duration_seconds": float(t["stage1"]["duration_seconds"] + t["stage2"]["duration_seconds"]),
                "selected_checkpoint_sha256": t["selected_checkpoint_sha256"],
                "selected_threshold": float(meta["selected_threshold"]),
                "effective_batch_size": batch.get("effective_batch_size"),
                "micro_batch_size": batch.get("micro_batch_size"),
                "gradient_accumulation_steps": batch.get("gradient_accumulation_steps"),
                "oom_retry_before_success": bool(batch.get("last_oom_before_success")),
                "last_oom_before_success": batch.get("last_oom_before_success"),
            })
    return (
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame(),
        pd.concat(patient_frames, ignore_index=True) if patient_frames else pd.DataFrame(),
        pd.DataFrame(training_rows),
        pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame(),
    )


def endpoint_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    p = metrics[(metrics.evaluation_level == "patient") & (metrics.population == "all") & metrics.overlap_pct.isin([0, 100])].copy()
    return p.rename(columns={"roc_auc": "patient_roc_auc", "average_precision": "patient_average_precision"})


def paired_endpoint_tables(endpoint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if endpoint.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    for (arch, fold, rep), g in endpoint.groupby(["architecture", "outer_fold", "training_replicate"]):
        z = g.set_index("overlap_pct")
        if 0 not in z.index or 100 not in z.index:
            continue
        rows.append({
            "architecture": arch,
            "outer_fold": int(fold),
            "training_replicate": int(rep),
            "roc_auc_0": float(z.loc[0, "patient_roc_auc"]),
            "roc_auc_100": float(z.loc[100, "patient_roc_auc"]),
            "delta_patient_roc_auc": float(z.loc[100, "patient_roc_auc"] - z.loc[0, "patient_roc_auc"]),
        })
    pair = pd.DataFrame(rows)
    if pair.empty:
        return pair, pair, pair
    fold = pair.groupby(["architecture", "outer_fold"], as_index=False).agg(
        seed_averaged_delta=("delta_patient_roc_auc", "mean"),
        seed_sd=("delta_patient_roc_auc", "std"),
        n_replicates=("delta_patient_roc_auc", "size"),
    )
    overall = fold.groupby("architecture", as_index=False).agg(
        mean_delta_across_folds=("seed_averaged_delta", "mean"),
        sd_delta_across_folds=("seed_averaged_delta", "std"),
        n_folds=("seed_averaged_delta", "size"),
        positive_folds=("seed_averaged_delta", lambda s: int((s > 0).sum())),
        tied_folds=("seed_averaged_delta", lambda s: int((s == 0).sum())),
        negative_folds=("seed_averaged_delta", lambda s: int((s < 0).sum())),
    )
    return pair, fold, overall


def common_held_out_sensitivity(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_root)
    groups: Dict[tuple, List[tuple[int, Path, dict]]] = {}
    for marker in _validated_markers(root):
        meta = json.loads(marker.read_text())
        key = (meta["architecture"], int(meta["outer_fold"]), int(meta["training_replicate"]))
        groups.setdefault(key, []).append((int(meta["overlap_pct"]), marker.parent, meta))
    rows = []
    predictions_out = []
    for (arch, fold, rep), items in groups.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda x: x[0])
        frames = []
        metas = []
        for ov, run_dir, meta in items:
            p = pd.read_csv(run_dir / "file_predictions.csv")
            frames.append(p)
            metas.append((ov, run_dir, meta))
        common_frames = common_held_out_analysis(frames)
        for cf, (ov, run_dir, meta) in zip(common_frames, metas):
            th = float(meta["selected_threshold"])
            cf["architecture"] = arch
            cf["outer_fold"] = fold
            cf["training_replicate"] = rep
            cf["overlap_pct"] = ov
            predictions_out.append(cf)
            pat = aggregate_patient(cf)
            rows.extend([
                metric_row(cf, th, "slice", arch, fold, ov, rep, int(meta["training_seed"]), "common_held_out"),
                metric_row(pat, th, "patient", arch, fold, ov, rep, int(meta["training_seed"]), "common_held_out"),
            ])
    return pd.DataFrame(rows), (pd.concat(predictions_out, ignore_index=True) if predictions_out else pd.DataFrame())


def overlap_curve_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    p = metrics[(metrics.evaluation_level == "patient") & (metrics.population == "all")]
    return p.groupby(["architecture", "overlap_pct"], as_index=False).agg(
        patient_roc_auc_mean=("roc_auc", "mean"),
        patient_roc_auc_sd=("roc_auc", "std"),
        patient_average_precision_mean=("average_precision", "mean"),
        patient_average_precision_sd=("average_precision", "std"),
        n_fold_seed_runs=("roc_auc", "size"),
    )


def write_results_summary(result_dir: Path, overall: pd.DataFrame, bootstrap_rows: list[dict], run_count: int) -> None:
    lines = [
        "# RESULTS_SUMMARY",
        "",
        "This file is generated only from completed, integrity-validated official runs.",
        f"Completed run count: **{run_count}**.",
        "",
    ]
    if overall.empty:
        lines.append("No complete paired endpoint summary is available yet.")
    else:
        lines.append("## Verified paired endpoint effects")
        for _, r in overall.iterrows():
            lines.append(
                f"- {r.architecture}: mean seed-averaged fold Δ patient ROC-AUC = "
                f"{r.mean_delta_across_folds:+.4f} (SD {r.sd_delta_across_folds:.4f}); "
                f"positive/tied/negative folds = {int(r.positive_folds)}/{int(r.tied_folds)}/{int(r.negative_folds)}."
            )
    if bootstrap_rows:
        lines.extend(["", "## Conditional paired subject-bootstrap intervals"])
        for b in bootstrap_rows:
            lines.append(
                f"- {b['architecture']}: Δ = {b['observed_mean_seed_averaged_fold_delta']:+.4f}, "
                f"95% conditional bootstrap CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]."
            )
        lines.append("Intervals condition on the fitted model collection and fixed subject partitions.")
    lines.extend([
        "",
        "## Output paths",
        "- `run_level_metrics.csv`",
        "- `paired_endpoint_differences.csv`",
        "- `seed_averaged_fold_differences.csv`",
        "- `conditional_subject_bootstrap.csv`",
        "- `common_held_out_metrics.csv`",
        "- `training_checkpoint_summary.csv`",
        "- `partition_audits.csv`",
        "- `figures/`",
    ])
    atomic_write_text(result_dir / "RESULTS_SUMMARY.md", "\n".join(lines))


def aggregate_outputs(output_root: str | Path, result_dir: str | Path, bootstrap_replicates: int = 10_000, analysis_seed: int = 20260910) -> dict:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    completeness = assert_frozen_plan_complete(output_root)
    atomic_write_json(result_dir / "aggregation_completeness_audit.json", completeness)
    metrics, patients, training, audits = collect_completed_runs(output_root)
    if audits.empty or not (audits.audit_status.astype(str).str.upper() == "PASS").all():
        raise RuntimeError("Aggregation refused: one or more validated runs has a missing/failed partition audit.")
    atomic_write_csv(result_dir / "run_level_metrics.csv", metrics)
    atomic_write_csv(result_dir / "patient_predictions_all_runs.csv", patients)
    atomic_write_csv(result_dir / "training_checkpoint_summary.csv", training)
    atomic_write_csv(result_dir / "partition_audits.csv", audits)

    endpoint = endpoint_summary(metrics)
    pair, fold, overall = paired_endpoint_tables(endpoint)
    atomic_write_csv(result_dir / "endpoint_metrics.csv", endpoint)
    atomic_write_csv(result_dir / "paired_endpoint_differences.csv", pair)
    atomic_write_csv(result_dir / "seed_averaged_fold_differences.csv", fold)
    atomic_write_csv(result_dir / "endpoint_summary_across_folds.csv", overall)

    bootstrap_rows = []
    if not patients.empty:
        for arch, g in patients.groupby("architecture"):
            if {0, 100} <= set(g.overlap_pct.unique()):
                try:
                    bootstrap_rows.append(conditional_collection_bootstrap(g, bootstrap_replicates, analysis_seed))
                except ValueError:
                    pass
    atomic_write_csv(result_dir / "conditional_subject_bootstrap.csv", pd.DataFrame(bootstrap_rows))

    common_metrics, common_predictions = common_held_out_sensitivity(output_root)
    atomic_write_csv(result_dir / "common_held_out_metrics.csv", common_metrics)
    atomic_write_csv(result_dir / "common_held_out_file_predictions.csv", common_predictions)

    curve = overlap_curve_summary(metrics)
    atomic_write_csv(result_dir / "overlap_curve_summary.csv", curve)
    endpoint_paired_plot(endpoint, result_dir / "figures")
    roc_auc_by_overlap_plot(metrics, result_dir / "figures", "patient")
    roc_auc_by_overlap_plot(metrics, result_dir / "figures", "slice")
    full_overlap_curve_plot(curve, result_dir / "figures")
    write_results_summary(result_dir, overall, bootstrap_rows, len(collect_run_records(output_root)))

    manifest = {
        "n_completed_runs": int(len(collect_run_records(output_root))),
        "frozen_plan_completeness": completeness,
        "files": {},
    }
    for p in sorted(result_dir.rglob("*")):
        if p.is_file() and p.name != "RESULTS_MANIFEST.json":
            manifest["files"][str(p.relative_to(result_dir))] = sha256_file(p)
    atomic_write_json(result_dir / "RESULTS_MANIFEST.json", manifest)

    # Create a compact final results archive beside RESULT_DIR and hash it.
    archive_path = result_dir.parent / (result_dir.name + ".zip")
    partial_archive = archive_path.with_name(archive_path.name + ".partial")
    if partial_archive.exists():
        partial_archive.unlink()
    with zipfile.ZipFile(partial_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(result_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(result_dir))
    partial_archive.replace(archive_path)
    archive_hash = sha256_file(archive_path)
    atomic_write_text(archive_path.with_suffix(archive_path.suffix + ".sha256"), archive_hash + "\n")
    manifest["results_archive"] = {
        "path": str(archive_path),
        "sha256": archive_hash,
        "bytes": int(archive_path.stat().st_size),
    }
    atomic_write_json(result_dir / "RESULTS_MANIFEST.json", manifest)
    return manifest
