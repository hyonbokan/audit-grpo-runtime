"""Mechanical checks for the AGENTS.md code rules; legacy debt may only shrink.

Regenerate the ratchet after removing debt:  python3 test_conventions.py --update
"""

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "testing/conventions_baseline.json"
FROZEN_BACKEND = Path("/workspace/code/backend")
# Reviewed contract files; a new one needs a reason, not a copy of schedules.json.
SCOPE_FILES = {
    "benchmark_scope.json",
    "cycle_scope.json",
    "evaluation_scope.json",
    "native_context_scope.json",
}
# Patterns that belong in runtime_io.py; elsewhere they are duplicated helpers.
DUPLICATED_HELPERS = {
    "read_json": r"json\.loads\([^\n]*read_text\(\)\)",
    "write_json": r"write_text\(\s*json\.dumps",
    "sha256": r"hashlib\.(sha256|file_digest)",
    "clock": r"datetime\.now\(",
}


def test_tests_do_not_import_other_test_modules():
    offenders = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in test_files()
        for line, module in imported_modules(path)
        if module.split(".")[-1].startswith("test_")
    ]
    assert offenders == [], "Move shared builders into testing/"


def test_tests_assert_behavior_not_attribute_presence():
    needle = "assert " + "hasattr("  # Split so this file does not match itself.
    offenders = [str(p.relative_to(ROOT)) for p in test_files() if needle in text(p)]
    assert offenders == []


def test_module_names_do_not_shadow_frozen_backend_modules():
    # The frozen grader puts probes/ on sys.path; a same-named module here replaces its import.
    if not FROZEN_BACKEND.is_dir():
        pytest.fail(f"Frozen backend not mounted at {FROZEN_BACKEND}; run via run-tests*.sh")
    finetuning = FROZEN_BACKEND / "scripts/finetuning"
    theirs = {p.stem for d in ("probes", "tools") for p in (finetuning / d).glob("*.py")}
    theirs |= {p.name for p in FROZEN_BACKEND.iterdir() if p.is_dir()}
    theirs |= {p.stem for p in FROZEN_BACKEND.glob("*.py")}
    ours = {p.stem for p in [*ROOT.glob("*.py"), *(ROOT / "inference").glob("*.py")]}
    ours |= {"grpo", "testing"}
    assert sorted(ours & theirs - {"__init__", "conftest"}) == []


@pytest.mark.parametrize(
    "launcher", sorted(p.name for p in ROOT.glob("launch-*.sh") if p.name != "launch-lib.sh")
)
def test_launchers_fail_fast_and_verify_source_before_any_container(launcher):
    lines = text(ROOT / launcher).splitlines()
    assert "set -euo pipefail" in lines
    verify = next(i for i, line in enumerate(lines) if "sha256sum --check" in line)
    first_docker = next(i for i, line in enumerate(lines) if re.search(r"\bdocker\b", line))
    assert verify < first_docker


def test_schedules_live_only_in_the_registry():
    assert {p.name for p in ROOT.glob("*_scope.json")} == SCOPE_FILES
    readers = [p.name for p in production_files() if '"schedules.json"' in text(p)]
    assert readers == ["schedules.py"]


def test_duplicated_helper_and_inline_import_debt_only_shrinks():
    current, allowed = debt(), json.loads(BASELINE.read_text())
    grown = {
        f"{rule} {name}": f"{count} > {allowed.get(rule, {}).get(name, 0)}"
        for rule, files in current.items()
        for name, count in files.items()
        if count > allowed.get(rule, {}).get(name, 0)
    }
    assert grown == {}, "Use runtime_io helpers or hoist the import; see AGENTS.md"
    assert current == allowed, "Debt went down: run `python3 test_conventions.py --update`"


# --- Scanning


def debt() -> dict[str, dict[str, int]]:
    result: dict[str, Counter] = {
        rule: Counter() for rule in [*DUPLICATED_HELPERS, "inline_import"]
    }
    for path in production_files():
        name, source = str(path.relative_to(ROOT)), text(path)
        if path.name != "runtime_io.py":
            for rule, pattern in DUPLICATED_HELPERS.items():
                result[rule][name] += len(re.findall(pattern, source))
        result["inline_import"][name] += inline_imports(path)
    return {
        rule: dict(sorted((k, v) for k, v in counts.items() if v))
        for rule, counts in result.items()
    }


def inline_imports(path: Path) -> int:
    tree = ast.parse(text(path))
    module_level = set(map(id, tree.body))
    return sum(
        isinstance(node, ast.Import | ast.ImportFrom) and id(node) not in module_level
        for node in ast.walk(tree)
    )


def imported_modules(path: Path) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(ast.parse(text(path))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno, node.module))
    return found


def production_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*.py")
        if not p.name.startswith(("test_", "conftest")) and "testing" not in p.parts
    )


def test_files() -> list[Path]:
    return sorted(ROOT.rglob("test_*.py"))


def text(path: Path) -> str:
    return path.read_text()


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit("Usage: python3 test_conventions.py --update")
    BASELINE.write_text(json.dumps(debt(), indent=2) + "\n")
    print(f"Wrote {BASELINE.relative_to(ROOT)}")
