"""Regenerate SOURCE_SHA256SUMS, the identity launchers verify before any GPU or paid work.

Run after every source change, before copying SOURCE_SHA256SUMS into a new result
directory:  python3 training-code/provenance.py  (stdlib only; `--check` verifies).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_SUFFIXES = {".py", ".sh", ".json", ".c"}  # .c: the peer-access shim preloaded in containers
# Outside training-code but executed by launch-training.sh.
EXTERNAL_SOURCES = ("../diagnostic-tools/merge_checkpoint.py",)
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "fixtures", "testing", "private"}


def maintained_files(root: Path = ROOT) -> list[str]:
    """Executable source and reviewed configuration; tests, fixtures and docs are excluded."""
    local = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and not path.name.startswith("test_")
        and not IGNORED_PARTS & set(path.relative_to(root).parts)
    )
    return local + list(EXTERNAL_SOURCES)


def checksums(root: Path = ROOT) -> str:
    lines = []
    for name in maintained_files(root):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the checksums are stale")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    path = args.root / "SOURCE_SHA256SUMS"
    if args.check:
        stale = not path.is_file() or path.read_text() != checksums(args.root)
        print("stale" if stale else "current")
        return int(stale)
    path.write_text(checksums(args.root))
    print(f"Updated SOURCE_SHA256SUMS for {len(maintained_files(args.root))} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
