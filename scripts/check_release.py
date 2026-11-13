"""Privacy and release hygiene scan for files intended for version control."""

from pathlib import Path
import gzip
import re

ROOT = Path(__file__).resolve().parents[1]
# Join marker fragments so the self-test does not match its own source text.
DENY = [
    r"(?i)\b(?:to" + "do|fix" + "me|rev" + "iewer|hand" + "off|manuscript" + r"\.number)\b",
    r"(?i)(?:I" + "2T1|I" + "6_RESUMABLE|I" + "7_|V" + "4C|Q" + "1_LEAKAGE)",
    "/content" + "/drive",
    "/Users" + "/",
    r"[A-Za-z]:\\Users\\",
    r"(?i)AKIA[0-9A-Z]{16}",
    r"(?i)-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
    # Require a real looking email domain rather than treating Python matmul as an address.
    r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.(?:com|org|net|edu|gov|io|ac\.[a-z]{2}|edu\.[a-z]{2})\b",
]
BAD_EXT = {
    ".ipynb",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".docx",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".pickle",
    ".pkl",
    ".zip",
    ".log",
}
problems = []
for p in ROOT.rglob("*"):
    if not p.is_file() or any(
        x in p.parts for x in (".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "outputs")
    ):
        continue
    rel = p.relative_to(ROOT).as_posix()
    if rel == "SHA256SUMS.txt":
        continue
    if p.suffix.lower() in BAD_EXT or p.stat().st_size > 2_000_000:
        problems.append(f"Binary, oversized or disallowed asset: {rel}")
        continue
    if ".ds_store" in rel.lower() or rel.endswith(".env"):
        problems.append("Disallowed path: " + rel)
    try:
        if rel.endswith(".csv.gz"):
            with gzip.open(p, "rt", encoding="utf-8") as source:
                content = source.read()
        else:
            content = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, gzip.BadGzipFile):
        problems.append("Unexpected binary: " + rel)
        continue
    for patt in DENY:
        if re.search(patt, content):
            problems.append(f"{rel}: {patt}")
if problems:
    raise SystemExit("\n".join(problems))
print("Release hygiene checks passed")
