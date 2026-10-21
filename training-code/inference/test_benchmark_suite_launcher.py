"""Suite orchestration preserves failures and continues to unvisited projects."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from testing.host import write_host_settings


def test_suite_records_child_failure_then_finishes_partial_report(tmp_path):
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    source = Path(__file__).parents[1] / "launch-benchmark-suite.sh"
    shutil.copyfile(source, code / source.name)
    shutil.copyfile(source.with_name("launch-lib.sh"), code / "launch-lib.sh")
    cache = write_host_settings(root)
    out = root / "results/suite"
    out.mkdir(parents=True)
    (out / "config.json").write_text("{}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-benchmark-suite.sh\n")
    child = root / "results/child"
    child.mkdir()
    counter = tmp_path / "counter"
    (code / "launch-evaluation.sh").write_text(
        '#!/bin/bash\nprintf "57\\n" > "$1/launcher-exit.txt"\nexit 57\n'
    )
    executables = tmp_path / "bin"
    executables.mkdir()
    docker = executables / "docker"
    docker.write_text("""#!/bin/bash
echo "$*" >> "$TEST_CALLS"
if [[ "$*" == *' next '* ]]; then
  if [[ -e "$TEST_COUNTER" ]]; then echo complete; else touch "$TEST_COUNTER"; echo "$TEST_CHILD"; fi
fi
if [[ "$*" == *' report '* ]]; then touch "$TEST_REPORT"; fi
exit 0
""")
    docker.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(code / source.name), str(out), str(out / "config.json")],
        env={
            **os.environ,
            "PATH": f"{executables}:{os.environ['PATH']}",
            "TEST_COUNTER": str(counter),
            "TEST_CHILD": str(child),
            "TEST_REPORT": str(out / "report.json"),
            "TEST_CALLS": str(tmp_path / "docker-calls"),
        },
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert (child / "launcher-exit.txt").read_text().strip() == "57"
    assert (out / "project-exits.tsv").read_text() == "child\t57\n"
    assert (out / "report.json").exists()
    for call in (tmp_path / "docker-calls").read_text().splitlines():
        assert f"-v {cache}:{cache}" in call and "audit-grpo-deps:cu129-v1" in call
