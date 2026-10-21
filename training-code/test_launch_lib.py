"""launch-lib.sh helpers: deadline accounting and the transient deadline timer."""

import os
import subprocess
from pathlib import Path

from testing.host import write_host_settings

LIB = Path(__file__).with_name("launch-lib.sh")


def run(script: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    if not (tmp_path / "host.env").exists():
        write_host_settings(tmp_path)
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


def test_launchers_refuse_to_start_without_this_hosts_settings(tmp_path):
    prelude = f'code_root={tmp_path}; runtime_root={tmp_path}; source "{LIB}"; echo sourced'
    result = subprocess.run(["bash", "-c", prelude], capture_output=True, text=True)
    assert result.returncode == 2 and "sourced" not in result.stdout
    assert f"{tmp_path}/host.env" in result.stderr and "host.env.example" in result.stderr
    (tmp_path / "host.env").write_text("# no cache yet\n")
    result = subprocess.run(["bash", "-c", prelude], capture_output=True, text=True)
    assert result.returncode != 0 and "AUDIT_GRPO_CACHE" in result.stderr


def test_host_settings_set_the_cache_and_the_restored_workspace(tmp_path):
    cache = write_host_settings(tmp_path)
    result = run('printf "%s\\n" "$CACHE" "$FROZEN_WORKSPACE" "$CPU_IMAGE"', tmp_path)
    assert result.stdout.splitlines() == [
        str(cache),
        f"{tmp_path}/workspace",
        "audit-grpo-deps:cu129-v1",
    ]
