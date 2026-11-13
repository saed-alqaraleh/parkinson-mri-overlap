from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(payload)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def atomic_write_csv(path: str | Path, df, **to_csv_kwargs) -> None:
    """Atomically persist a pandas DataFrame as CSV.

    The temporary file is created beside the destination so os.replace is atomic
    on the same filesystem. The file is flushed and fsynced before replacement.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False, **to_csv_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_copy(src: str | Path, dst: str | Path) -> None:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(obj, sort_keys=False)
    atomic_write_text(path, text)


def seed_everything(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic_report = {"requested": bool(deterministic), "enabled": False, "warning": ""}
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            deterministic_report["enabled"] = True
        except Exception as e:  # pragma: no cover - version/platform dependent
            deterministic_report["warning"] = repr(e)
    return deterministic_report


def environment_report() -> Dict[str, Any]:
    """Return environment information without crashing on a broken CUDA/cuDNN runtime.

    Installed CUDA runtime libraries may differ from the framework's bundled versions.
    Record cuDNN inspection errors and apply `cuda_health_check()` as the hard
    compatibility gate before training.
    """
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": None,
        "cudnn_error": None,
        "device_count": 0,
        "devices": [],
    }
    try:
        if torch.backends.cudnn.is_available():
            report["cudnn"] = torch.backends.cudnn.version()
    except Exception as e:  # broken/mixed CUDA library environment
        report["cudnn_error"] = repr(e)

    try:
        report["device_count"] = int(torch.cuda.device_count())
        for i in range(report["device_count"]):
            p = torch.cuda.get_device_properties(i)
            report["devices"].append({
                "index": i, "name": p.name, "total_memory_bytes": int(p.total_memory),
                "major": p.major, "minor": p.minor,
            })
    except Exception as e:
        report["cuda_device_query_error"] = repr(e)

    try:
        import torchvision
        report["torchvision"] = torchvision.__version__
    except Exception as e:  # pragma: no cover
        report["torchvision_error"] = repr(e)
    return report


def cuda_health_check(require_cuda: bool = False) -> Dict[str, Any]:
    """Force a tiny CUDA/cuDNN operation and return a clear compatibility report."""
    report: Dict[str, Any] = {"required": bool(require_cuda), "status": "PASS", "error": None}
    if not torch.cuda.is_available():
        report.update({"cuda_available": False, "status": "FAIL" if require_cuda else "CPU_ONLY"})
        if require_cuda:
            report["error"] = "CUDA GPU is required by the official configuration but torch.cuda.is_available() is False."
        return report
    report["cuda_available"] = True
    try:
        report["cudnn_version"] = torch.backends.cudnn.version()
        # Conv2d forces the cuDNN path used by ConvNeXt and exercises CUDA allocation.
        x = torch.randn(1, 3, 16, 16, device="cuda", dtype=torch.float32)
        conv = torch.nn.Conv2d(3, 4, 3, padding=1).cuda()
        with torch.no_grad():
            y = conv(x)
        torch.cuda.synchronize()
        report["probe_sum"] = float(y.float().sum().cpu())
        del x, y, conv
        torch.cuda.empty_cache()
    except Exception as e:
        report["status"] = "FAIL"
        report["error"] = repr(e)
    return report


def framework_fingerprint() -> Dict[str, Any]:
    """Versions that must remain fixed across resumed official runs."""
    import torchvision
    return {
        "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def freeze_framework_environment(experiment_root: str | Path) -> Dict[str, Any]:
    """Freeze PyTorch/TorchVision versions on first official run and enforce them on resume."""
    experiment_root = Path(experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    path = experiment_root / "FROZEN_FRAMEWORK_ENVIRONMENT.json"
    current = framework_fingerprint()
    current_hash = canonical_json_hash(current)
    payload = {**current, "fingerprint_sha256": current_hash}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        keys = ["python_major_minor", "torch", "torchvision", "cuda_runtime"]
        mismatch = {k: {"frozen": old.get(k), "current": current.get(k)} for k in keys if old.get(k) != current.get(k)}
        if mismatch:
            raise RuntimeError(
                "Framework environment changed since the official experiment started. "
                f"Refusing to mix environments across paired runs. Mismatch: {mismatch}"
            )
        return old
    atomic_write_json(path, payload)
    return payload



def audit_io_roots(data_root: str | Path, output_root: str | Path) -> Dict[str, Any]:
    """Validate the read-only-data / writable-output contract.

    DATA_ROOT is *never* modified by this function. It is only traversed/read.
    OUTPUT_ROOT is required to be writable and is tested with a temporary probe.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)

    report: Dict[str, Any] = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "data_root_exists": data_root.exists(),
        "data_root_is_dir": data_root.is_dir(),
        "data_root_readable": False,
        "output_root_writable": False,
        "status": "FAIL",
        "note": "DATA_ROOT is treated as read-only; all writes go to OUTPUT_ROOT.",
    }

    if not data_root.exists() or not data_root.is_dir():
        return report

    # Read-only check: list a small part of the directory; do not create anything.
    try:
        with os.scandir(data_root) as it:
            next(it, None)
        report["data_root_readable"] = True
    except Exception as e:
        report["data_root_read_error"] = repr(e)
        return report

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / f".write_probe_{os.getpid()}_{int(time.time())}"
        atomic_write_text(probe, "write-test\n")
        probe.unlink()
        report["output_root_writable"] = True
    except Exception as e:
        report["output_root_write_error"] = repr(e)
        return report

    report["status"] = "PASS"
    return report


def now_utc_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def run_key(architecture: str, fold: int, overlap_pct: int, replicate: int, seed: int) -> str:
    return f"{safe_slug(architecture)}__fold{fold}__ov{int(overlap_pct):03d}__rep{replicate}__seed{seed}"


def validate_completed_run(run_dir: str | Path, expected_config_hash: str | None = None) -> tuple[bool, str]:
    run_dir = Path(run_dir)
    marker = run_dir / "RUN_COMPLETE.json"
    if not marker.exists():
        return False, "RUN_COMPLETE.json missing"
    try:
        meta = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"invalid completion marker: {e}"
    if expected_config_hash is not None and meta.get("config_hash") != expected_config_hash:
        return False, "configuration hash mismatch"
    required = meta.get("required_outputs", {})
    if not isinstance(required, dict) or not required:
        return False, "completion marker has no required_outputs hashes"
    for rel, expected in required.items():
        p = run_dir / rel
        if not p.exists():
            return False, f"missing required output: {rel}"
        actual = sha256_file(p)
        if actual != expected:
            return False, f"hash mismatch for {rel}"
    return True, "PASS"


class FileLease:
    """Simple single-writer lease using exclusive file creation.

    It is intentionally conservative. If a stale lease is detected, the caller must
    explicitly allow stale replacement by setting stale_after_seconds.
    """

    def __init__(self, path: str | Path, stale_after_seconds: int = 12 * 3600):
        self.path = Path(path)
        self.stale_after_seconds = stale_after_seconds
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_after_seconds:
                self.path.unlink()
            else:
                raise RuntimeError(f"Active run lease exists: {self.path}")
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "created": now_utc_iso()}))
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False
        return False
