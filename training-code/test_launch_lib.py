"""launch-lib.sh helpers: deadline accounting and the transient deadline timer."""

import os
import subprocess
from pathlib import Path

LIB = Path(__file__).with_name("launch-lib.sh")


def run(script: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    prelude = f'code_root={tmp_path}; runtime_root={tmp_path}; source "{LIB}"\n'
    return subprocess.run(
        ["bash", "-c", prelude + script],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )


def test_remaining_reports_seconds_left_and_fails_with_deadline_status_after(tmp_path):
    ahead = run("end=$(( $(date +%s) + 100 )); remaining", tmp_path)
    assert ahead.returncode == 0 and 98 <= int(ahead.stdout) <= 100
    passed = run("end=$(( $(date +%s) - 1 )); remaining", tmp_path)
    assert passed.returncode == 124 and passed.stdout == ""


def test_deadline_timer_replaces_a_stale_unit_and_stops_the_named_containers(tmp_path):
    calls = tmp_path / "calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("sudo", "systemctl", "systemd-run"):
        path = bin_dir / name
        body = 'exec "$@"' if name == "sudo" else f'echo "{name} $*" >> {calls}'
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)
    result = run(
        "start_deadline_timer unit 90 c-rollout c-update",
        tmp_path,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )
    assert result.returncode == 0, result.stderr
    stop = "/usr/bin/docker stop -t 0 c-rollout c-update"
    assert calls.read_text().splitlines() == [
        "systemctl stop unit.timer",
        "systemctl reset-failed unit.timer unit.service",
        f"systemd-run --quiet --unit unit --on-active=90s {stop}",
        "systemctl is-active --quiet unit.timer",
    ]
