"""Start a schedule through launch-segment.sh with a stub Docker boundary."""

import os
import shutil
import subprocess
from pathlib import Path

from testing.host import write_host_settings

SCHEDULE = "test-schedule"


def test_a_started_run_records_the_commit_and_uncommitted_changes_it_froze(tmp_path):
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    write_host_settings(root)
    with (root / "host.env").open("a") as stream:
        stream.write("AUDIT_GRPO_HF_REPO=owner/backups\n")
    (root / "runs").mkdir()
    for name in ("launch-segment.sh", "launch-lib.sh", "provenance.py"):
        shutil.copyfile(Path(__file__).with_name(name), code / name)
    (code / "launch-training.sh").write_text("exit 0\n")
    (code / "schedules.json").write_text("{}\n")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "training-code"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    (code / "schedules.json").write_text('{"registered": "since the commit"}\n')
    executables = tmp_path / "bin"
    executables.mkdir()
    docker = executables / "docker"
    docker.write_text("#!/bin/bash\nexit 0\n")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(code / "launch-segment.sh"), SCHEDULE],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run = root / "runs" / SCHEDULE
    assert (run / "SOURCE_SHA256SUMS").read_bytes() == (code / "SOURCE_SHA256SUMS").read_bytes()
    assert (run / "source-commit.txt").read_text() == f"{head}\n"
    diff = (run / "source-uncommitted.diff").read_text()
    assert '+{"registered": "since the commit"}' in diff
    assert "SOURCE_SHA256SUMS" in diff
