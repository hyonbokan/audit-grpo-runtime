"""Run the shell launcher with a failing Docker boundary and observe cleanup."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from testing.host import write_host_settings


def test_failed_gpu_process_preserves_exit_and_stops_the_cycle(tmp_path):
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    write_host_settings(root)
    run = root / "runs/test-cycle"
    run.mkdir(parents=True)
    source = Path(__file__).parent / "launch-cycle.sh"
    shutil.copyfile(source, code / "launch-cycle.sh")
    shutil.copyfile(Path(__file__).with_name("launch-lib.sh"), code / "launch-lib.sh")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-cycle.sh\n")
    config = run / "config.json"
    config.write_text("{}")
    executables = tmp_path / "bin"
    executables.mkdir()
    scripts = {
        "docker": """#!/bin/bash
case "$1" in
run)
  if [[ "$*" == *'cycle_contract settings'* ]]; then
    [[ "$*" == *'/run-config.json'* ]] || exit 93
    printf '%s\\n' native rollout 20 10 false '"device=0"' '"device=0,1"'; exit 0
  fi
  for arg in "$@"; do
    if [[ "$arg" == --name ]]; then echo "GPU boundary failed $*"; exit 57; fi
  done
  exit 0;;
inspect) echo '{"Status":"exited","ExitCode":57}';;
logs) echo 'GPU boundary failed';;
stop) exit 0;;
*) echo 'unexpected next phase' >&2; exit 99;;
esac
""",
    }
    for name, text in scripts.items():
        path = executables / name
        path.write_text(text)
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(code / "launch-cycle.sh"), str(run), str(config)],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 57, result.stderr
    assert (run / "launcher-exit.txt").read_text().strip() == "57"
    assert (run / "last-phase.txt").read_text().strip() == "rollout"
    assert "GPU boundary failed" in (run / "rollout.log").read_text()
    assert '--gpus "device=0,1"' in (run / "rollout.log").read_text()
    assert (run / "audit-grpo-test-cycle-rollout-state.json").exists()
    assert not (run / "grading.log").exists()
    assert not (run / "update.log").exists()


def test_resume_skips_completed_phases_and_archives_the_failed_attempt(tmp_path):
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    write_host_settings(root)
    run = root / "runs/test-cycle"
    run.mkdir(parents=True)
    for name in ("launch-cycle.sh", "launch-lib.sh", "cycle_recovery.py"):
        shutil.copyfile(Path(__file__).with_name(name), code / name)
    digest = hashlib.sha256((code / "launch-cycle.sh").read_bytes()).hexdigest()
    (run / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-cycle.sh\n")
    config = run / "config.json"
    config.write_text("{}")
    (run / "prepared.json").write_text("{}")
    (run / "rollout").mkdir()
    (run / "rollout/receipt.json").write_text('{"status": "versioned_rollout_saved"}')
    (run / "rollout.log").write_text("completed rollout")
    (run / "grading").mkdir()
    (run / "grading/rewards.json").write_text("[1, 0, 1, 0]")
    (run / "update").mkdir()
    (run / "update/receipt.json").write_text('{"status": "failed"}')
    (run / "launcher-exit.txt").write_text("1\n")
    executables = tmp_path / "bin"
    executables.mkdir()
    scripts = {
        # Only configuration reads and the GPU update may reach Docker; grading
        # (docker exec/cp) and a repeated preparation would fail this boundary.
        "docker": """#!/bin/bash
case "$1" in
run)
  [[ "$*" != *'cycle_contract prepare'* ]] || exit 94
  if [[ "$*" == *'cycle_contract settings'* ]]; then
    printf '%s\\n' native rollout 20 10 false '"device=0"' '"device=0,1"'; exit 0
  fi
  for arg in "$@"; do
    if [[ "$arg" == --name ]]; then echo "GPU phase $*" >&2; exit 57; fi
  done
  exit 0;;
rm|stop|logs) exit 0;;
inspect) echo '{}';;
*) echo "unexpected docker $1" >&2; exit 99;;
esac
""",
    }
    for name, text in scripts.items():
        path = executables / name
        path.write_text(text)
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(code / "launch-cycle.sh"), str(run), str(config), "--resume"],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 57, result.stderr
    assert (run / "last-phase.txt").read_text().strip() == "update"
    assert '--gpus "device=0"' in (run / "update.log").read_text()
    assert "update" in (run / "update.log").read_text()
    assert (run / "rollout.log").read_text() == "completed rollout"
    assert (run / "grading/rewards.json").exists()
    [archive] = (run / "attempts").iterdir()
    assert (archive / "update/receipt.json").exists()
    assert (archive / "launcher-exit.txt").read_text() == "1\n"
