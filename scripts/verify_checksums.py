"""Verify every distributed artifact against the published SHA-256 inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OMIT = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "outputs", ".git"}


def releasable() -> set[str]:
    return {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & OMIT) and p.name != "SHA256SUMS.txt"
    }


def verify() -> None:
    listed = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        assert relative not in listed, f"Repeated checksum path: {relative}"
        listed[relative] = digest
    actual = releasable()
    if set(listed) != actual:
        missing = sorted(actual - set(listed))
        superfluous = sorted(set(listed) - actual)
        raise RuntimeError(f"Checksum file list differs (missing={missing}; stale={superfluous})")
    errors = []
    for relative, digest in listed.items():
        candidate = ROOT / relative
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            errors.append(relative)
    if errors:
        raise RuntimeError("Digest mismatch: " + ", ".join(errors))
    print(f"Verified {len(listed)} release file checksums")


if __name__ == "__main__":
    verify()
