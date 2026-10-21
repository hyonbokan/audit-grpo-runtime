"""Download every locked wheel, hash-checked, into one directory so image builds run offline.

One pip process per package, several at a time: each wheel is saved as soon as it is
complete, so an interrupted run resumes without fetching finished wheels again.
Usage: python3 images/fetch_wheels.py WHEEL_DIRECTORY [--jobs N]   (stdlib only)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGES = Path(__file__).resolve().parent
# Target interpreter per lock.
LOCKS = {
    "judge-requirements.lock": "3.14",
    "gpu-requirements-cu129.lock": "3.12",
    "vllm-requirements-cu129.lock": "3.12",
}
# Pin pip to this interpreter: uvx otherwise takes the first Python on PATH, which may lack SSL.
PIP = ("uvx", "--python", sys.executable, "--from", "pip==25.3", "pip")
# pip matches platform tags exactly: every manylinux level Ubuntu 24.04's glibc accepts.
PLATFORMS = [
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
    "linux_x86_64",
    "any",
    *(f"manylinux_2_{minor}_x86_64" for minor in range(5, 40)),
]
DEFAULT_JOBS = 6
HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    args = parser.parse_args(argv)
    if importlib.util.find_spec("_ssl") is None or not ssl_loads():
        print(
            f"{sys.executable} cannot load SSL; run with an interpreter that can", file=sys.stderr
        )
        return 2
    args.wheels.mkdir(parents=True, exist_ok=True)
    work = args.wheels / ".requirements"
    work.mkdir(exist_ok=True)
    present = {sha256(path) for path in args.wheels.glob("*.whl")}
    tasks, seen = [], set()
    for lock, version in LOCKS.items():
        for path in split_lock(IMAGES / lock, work / lock):
            # The two Python 3.12 locks share most pins; fetch each identical one once.
            pin = (version, pin_lines(path))
            if pin not in seen and not present & set(HASH.findall(pin[1])):
                seen.add(pin)
                tasks.append((path, version, args.wheels))
    with ThreadPoolExecutor(args.jobs) as pool:
        failures = [name for name, ok in pool.map(lambda t: download(*t), tasks) if not ok]
    wheels = len(list(args.wheels.glob("*.whl")))
    print(f"{len(tasks)} requirements, {len(failures)} failed, {wheels} wheels in {args.wheels}")
    for name in failures:
        print(f"failed: {name} (rerun to retry)", file=sys.stderr)
    return int(bool(failures))


# --- Helpers


def ssl_loads() -> bool:
    try:
        importlib.import_module("ssl")
    except ImportError:
        return False
    return True


def split_lock(lock: Path, out: Path) -> list[Path]:
    """Write one requirement file per locked package, each with the lock's index options."""
    options, blocks = [], []
    for line in lock.read_text().splitlines():
        if line.startswith("--"):
            options.append(line)
        elif line and not line[0].isspace() and not line.startswith("#"):
            blocks.append([line])
        elif line.strip().startswith("--hash") and blocks:
            blocks[-1].append(line)
    out.mkdir(exist_ok=True)
    paths = []
    for block in blocks:
        name = block[0].split("==")[0].split(" ")[0]
        path = out / f"{name}.txt"
        path.write_text("\n".join([*options, *block]) + "\n")
        paths.append(path)
    return paths


def sha256(path: Path) -> str:
    # Chunked rather than hashlib.file_digest: host Python may be 3.10.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_lines(requirements: Path) -> str:
    return "\n".join(line for line in requirements.read_text().splitlines() if line[:2] != "--")


def download(requirements: Path, python: str, wheels: Path) -> tuple[str, bool]:
    platforms = [option for tag in PLATFORMS for option in ("--platform", tag)]
    command = [
        *PIP,
        "download",
        "--quiet",
        "--dest",
        str(wheels),
        "--require-hashes",
        "--no-deps",
        # uv pip sync installs locked wheels regardless of Requires-Python (beanie 2.0.1
        # declares <3.14 and is in the 3.14 judge lock); the hashes still pin every file.
        "--ignore-requires-python",
        "--only-binary",
        ":all:",
        "--python-version",
        python,
        "--implementation",
        "cp",
        "--abi",
        f"cp{python.replace('.', '')}",
        "--abi",
        "abi3",
        "--abi",
        "none",
        *platforms,
        "--timeout",
        "120",
        "--retries",
        "10",
        "--requirement",
        str(requirements),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    label = f"{requirements.parent.name}:{requirements.stem}"
    if result.returncode:
        print(f"{label}: {result.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
    return label, result.returncode == 0


if __name__ == "__main__":
    sys.exit(main())
