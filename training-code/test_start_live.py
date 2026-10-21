"""start-live.sh: one credential-bearing judge container, created once and never duplicated."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from testing.host import write_host_settings

CODE = Path(__file__).parent


def live_runtime(tmp_path: Path, state: str) -> tuple[Path, Path, dict[str, str]]:
    """A runtime tree with the real scripts, a .env and a docker reporting `state`."""
    code = tmp_path / "runtime/training-code"
    code.mkdir(parents=True)
    for name in ("start-live.sh", "launch-lib.sh"):
        shutil.copyfile(CODE / name, code / name)
    write_host_settings(tmp_path / "runtime")
    (tmp_path / "runtime/.env").write_text("LITELLM_GATEWAY_URL=x\n")
    calls = tmp_path / "docker-calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        f"""#!/bin/bash
echo "$*" >> {calls}
if [[ "$1" == inspect ]]; then
  [[ {state} == missing ]] && exit 1
  echo {state}
fi
exit 0
"""
    )
    (bin_dir / "docker").chmod(0o755)
    return code, calls, {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def start(code: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(code / "start-live.sh")], env=env, capture_output=True, text=True
    )


def test_a_missing_live_container_is_created_with_credentials_read_only_and_network(tmp_path):
    code, calls, env = live_runtime(tmp_path, "missing")
    result = start(code, env)
    assert result.returncode == 0, result.stderr
    run = next(line for line in calls.read_text().splitlines() if line.startswith("run "))
    secrets = f"{tmp_path}/runtime/.env:/opt/audit-grpo-secrets/judge.env:ro"
    assert "--name audit-grpo-live" in run and f"-v {secrets}" in run
    assert "--network none" not in run  # The judge calls an external gateway.
    assert run.endswith("audit-grpo-deps:cu129-v1 sleep infinity")


@pytest.mark.parametrize("state,expected", [("running", None), ("exited", "start audit-grpo-live")])
def test_an_existing_live_container_is_reused_not_duplicated(tmp_path, state, expected):
    code, calls, env = live_runtime(tmp_path, state)
    assert start(code, env).returncode == 0
    lines = calls.read_text().splitlines()
    assert not any(line.startswith("run ") for line in lines)
    assert (
        (expected in lines)
        if expected
        else lines == ["inspect --format {{.State.Status}} audit-grpo-live"]
    )


def test_no_live_container_without_the_credentials_file(tmp_path):
    code, calls, env = live_runtime(tmp_path, "missing")
    (tmp_path / "runtime/.env").unlink()
    result = start(code, env)
    assert result.returncode == 2 and ".env" in result.stderr and not calls.exists()
