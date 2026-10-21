"""Fail unless the running environment holds exactly the distributions pinned in a lock file."""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

# Pinned in every lock; evaluates platform markers the same way the installer does.
from packaging.markers import Marker

PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)\s*(?:;\s*([^\\]+?))?\s*\\?$")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: dependency_check.py LOCK_FILE", file=sys.stderr)
        return 2
    expected = locked_versions(Path(argv[0]))
    installed = {
        normalize(dist.metadata["Name"]): dist.version
        for dist in importlib.metadata.distributions()
    }
    problems = [
        f"{name}: locked {version}, installed {installed.get(name, 'missing')}"
        for name, version in sorted(expected.items())
        if installed.get(name) != version
    ]
    unlocked = sorted(installed.keys() - expected.keys())
    problems += [f"{name}: installed {installed[name]}, not locked" for name in unlocked]
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{argv[0]}: {len(expected)} locked, {len(problems)} differences")
    return int(bool(problems))


# --- Helpers


def locked_versions(lock: Path) -> dict[str, str]:
    pins = {}
    for line in lock.read_text().splitlines():
        match = PIN.match(line)
        if match and (match[3] is None or Marker(match[3]).evaluate()):
            pins[normalize(match[1])] = match[2]
    if not pins:
        raise ValueError(f"No pins found in {lock}")
    return pins


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
