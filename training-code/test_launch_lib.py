"""launch-lib.sh helpers: deadline accounting and the detached deadline timer."""

import hashlib
import os
import subprocess
import time
from pathlib import Path

import pytest

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


def fake_docker(tmp_path: Path) -> tuple[dict[str, str], Path]:
    calls = tmp_path / "docker-calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(f'#!/bin/bash\necho "docker $*" >> {calls}\n')
    (bin_dir / "docker").chmod(0o755)
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}"}, calls


def wait_for(path: Path, seconds: float = 10) -> str:
    stop = time.monotonic() + seconds
    while time.monotonic() < stop:
        if path.exists() and path.read_text():
            return path.read_text()
        time.sleep(0.1)
    return ""


def running(pid: int) -> bool:
    # In a container without an init process, a killed orphan stays a zombie (state Z).
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state not in ("Z", "X")


def test_deadline_timer_stops_the_named_containers_after_its_launcher_has_exited(tmp_path):
    env, calls = fake_docker(tmp_path)
    started = run("start_deadline_timer unit 1 c-rollout c-update", tmp_path, **env)
    assert started.returncode == 0, started.stderr
    assert wait_for(calls).splitlines() == ["docker stop -t 0 c-rollout c-update"]


def test_stopped_deadline_timer_never_fires_and_leaves_no_process(tmp_path):
    env, calls = fake_docker(tmp_path)
    run("start_deadline_timer unit 2 c-rollout", tmp_path, **env)
    record = tmp_path / "results/deadline-timers/unit.pid"
    pid = int(record.read_text().split()[0])
    assert run("stop_deadline_timer unit", tmp_path, **env).returncode == 0
    time.sleep(3)
    assert not calls.exists() and not record.exists()
    assert not running(pid)


def test_restarting_a_deadline_timer_replaces_the_previous_one(tmp_path):
    env, calls = fake_docker(tmp_path)
    run("start_deadline_timer unit 1 c-old", tmp_path, **env)
    run("start_deadline_timer unit 1 c-new", tmp_path, **env)
    assert wait_for(calls)
    time.sleep(2)  # the replaced timer would have fired by now
    assert calls.read_text().splitlines() == ["docker stop -t 0 c-new"]


def test_stopping_a_stale_timer_record_never_kills_an_unrelated_process(tmp_path):
    other = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        record = tmp_path / "results/deadline-timers/unit.pid"
        record.parent.mkdir(parents=True)
        record.write_text(f"{other.pid} 0 c-rollout\n")
        assert run("stop_deadline_timer unit", tmp_path).returncode == 0
        assert other.poll() is None and not record.exists()
    finally:
        other.kill()


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
    result = run('printf "%s\\n" "$CACHE" "$FROZEN_WORKSPACE"', tmp_path)
    assert result.stdout.splitlines() == [str(cache), f"{tmp_path}/workspace"]
    # Docker cannot create /workspace/cache inside a read-only /workspace mount.
    assert (tmp_path / "workspace/cache").is_dir()


def compiling_docker(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """A docker stand-in that records calls and writes the compiler's output file."""
    calls = tmp_path / "docker-calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        f"""#!/bin/bash
echo "docker $*" >> {calls}
out=; target=
for arg in "$@"; do
  [[ "$arg" == *:/out ]] && out=${{arg%:/out}}
  [[ "$prev" == -o ]] && target=${{arg#/out/}}
  prev=$arg
done
[[ -n "$out" && -n "$target" ]] && touch "$out/$target"
"""
    )
    (bin_dir / "docker").chmod(0o755)
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}"}, calls


def test_disabled_peer_access_preloads_a_compiled_shim_in_every_container(tmp_path):
    cache = write_host_settings(tmp_path, peer_access="disabled")
    env, calls = compiling_docker(tmp_path)
    source = LIB.with_name("peer_access") / "no_peer_access.c"
    name = f"no-peer-access-{hashlib.sha256(source.read_bytes()).hexdigest()[:12]}.so"
    show = 'printf "%s\\n" "${environment[@]}"'
    first = run(show, tmp_path, **env)
    assert first.returncode == 0, first.stderr
    assert f"LD_PRELOAD=/workspace/cache/peer-access/{name}" in first.stdout.splitlines()
    assert (cache / "peer-access" / name).is_file()
    assert run(show, tmp_path, **env).returncode == 0
    assert sum(" gcc " in line for line in calls.read_text().splitlines()) == 1


def test_enabled_peer_access_changes_nothing(tmp_path):
    write_host_settings(tmp_path, peer_access="enabled")
    env, calls = compiling_docker(tmp_path)
    result = run('printf "%s\\n" "${environment[@]}"', tmp_path, **env)
    assert result.returncode == 0 and "LD_PRELOAD" not in result.stdout
    assert not calls.exists()


def test_compile_caches_persist_in_the_host_cache(tmp_path):
    write_host_settings(tmp_path, peer_access="enabled")
    env, _calls = compiling_docker(tmp_path)
    result = run('printf "%s\\n" "${environment[@]}"', tmp_path, **env)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "VLLM_CACHE_ROOT=/workspace/cache/vllm" in lines
    assert "TRITON_CACHE_DIR=/workspace/cache/triton" in lines


@pytest.mark.parametrize("setting", ["", "AUDIT_GRPO_PEER_ACCESS=off\n"])
def test_peer_access_must_be_declared_with_an_allowed_value(tmp_path, setting):
    cache = write_host_settings(tmp_path)
    (tmp_path / "host.env").write_text(f"AUDIT_GRPO_CACHE={cache}\n{setting}")
    result = run("echo sourced", tmp_path)
    assert result.returncode == 2 and "sourced" not in result.stdout
    assert "AUDIT_GRPO_PEER_ACCESS" in result.stderr and "enabled, disabled" in result.stderr
