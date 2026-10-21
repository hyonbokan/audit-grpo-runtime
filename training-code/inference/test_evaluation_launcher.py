"""A failed evaluation arm stops the comparison and preserves the real exit status."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from testing.host import write_host_settings


@pytest.mark.parametrize("native_benchmark", [False, True])
def test_failed_generation_prevents_grading_and_second_arm(tmp_path, native_benchmark):
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    run = root / "results/test-evaluation"
    run.mkdir(parents=True)
    source = Path(__file__).parents[1] / "launch-evaluation.sh"
    shutil.copyfile(source, code / "launch-evaluation.sh")
    shutil.copyfile(Path(__file__).parents[1] / "launch-lib.sh", code / "launch-lib.sh")
    write_host_settings(root)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-evaluation.sh\n")
    (run / "config.json").write_text("{}")
    executables = tmp_path / "bin"
    executables.mkdir()
    scripts = {
        "docker": """#!/bin/bash
case "$1" in
run)
  if [[ "$*" == *'evaluation.py settings'* ]]; then
    printf '%s\\n' native rollout 20 10 evaluation 'checkpoint-1 checkpoint-2' strict evaluation serial
    exit 0
  fi
  for arg in "$@"; do
    if [[ "$arg" == --name ]]; then echo 'generation failed'; exit 57; fi
  done
  exit 0;;
inspect) echo '{"Status":"exited","ExitCode":57}';;
logs) echo 'generation failed';;
stop) exit 0;;
*) echo 'unexpected phase' >&2; exit 99;;
esac
""",
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
    }
    if native_benchmark:
        scripts["docker"] = scripts["docker"].replace(
            "evaluation 'checkpoint-1 checkpoint-2'",
            "benchmark_evaluation 'base checkpoint-2'",
        )
    for name, contents in scripts.items():
        path = executables / name
        path.write_text(contents)
        path.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(code / "launch-evaluation.sh"), str(run), str(run / "config.json")],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 57, completed.stderr
    assert (run / "launcher-exit.txt").read_text().strip() == "57"
    label = "base" if native_benchmark else "checkpoint-1"
    assert (run / "last-phase.txt").read_text().strip() == label
    assert "generation failed" in (run / f"{label}.log").read_text()
    assert (run / f"audit-grpo-test-evaluation-{label}-state.json").exists()
    assert not (run / f"{label}-grading.log").exists()
    assert not (run / "checkpoint-2.log").exists()


def test_continuation_grades_only_the_conditional_copy_of_the_new_arm(tmp_path):
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    run = root / "results/test-evaluation"
    run.mkdir(parents=True)
    source = Path(__file__).parents[1] / "launch-evaluation.sh"
    shutil.copyfile(source, code / source.name)
    shutil.copyfile(Path(__file__).parents[1] / "launch-lib.sh", code / "launch-lib.sh")
    write_host_settings(root)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-evaluation.sh\n")
    (run / "config.json").write_text("{}")
    commands = tmp_path / "commands.txt"
    executables = tmp_path / "bin"
    executables.mkdir()
    scripts = {
        "docker": """#!/bin/bash
printf '%s\\n' "$*" >> "$TEST_COMMANDS"
if [[ "$1" == run && "$*" == *'evaluation.py settings'* ]]; then
  printf '%s\\n' native rollout 20 10 benchmark_evaluation checkpoint-8 conditional evaluation serial
fi
exit 0
""",
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
    }
    for name, contents in scripts.items():
        path = executables / name
        path.write_text(contents)
        path.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(code / source.name), str(run), str(run / "config.json")],
        env={
            **os.environ,
            "PATH": f"{executables}:{os.environ['PATH']}",
            "TEST_COMMANDS": str(commands),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    trace = commands.read_text().splitlines()
    generation = [line for line in trace if "--name" in line]
    assert len(generation) == 1
    assert "audit-grpo-test-evaluation-checkpoint-8" in generation[0]
    assert any("benchmark_evaluation.py normalize" in line for line in trace)
    grading = next(line for line in trace if "benchmark_judge.py" in line)
    assert "--samples /results/test-evaluation/conditional/checkpoint-8/samples.json" in grading
    assert not (run / "base.log").exists()


def test_paired_native_grading_can_overlap_second_generation_and_is_joined(tmp_path):
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    run = root / "results/test-evaluation"
    run.mkdir(parents=True)
    source = Path(__file__).parents[1] / "launch-evaluation.sh"
    shutil.copyfile(source, code / source.name)
    shutil.copyfile(Path(__file__).parents[1] / "launch-lib.sh", code / "launch-lib.sh")
    write_host_settings(root)
    (run / "SOURCE_SHA256SUMS").write_text(
        f"{hashlib.sha256(source.read_bytes()).hexdigest()}  launch-evaluation.sh\n"
    )
    (run / "config.json").write_text("{}")
    executables = tmp_path / "bin"
    executables.mkdir()
    marker = tmp_path / "second-generation"
    scripts = {
        "docker": """#!/bin/bash
if [[ "$1" == run && "$*" == *'evaluation.py settings'* ]]; then
  printf '%s\\n' native rollout 20 10 benchmark_evaluation 'base checkpoint-2' conditional evaluation overlap
fi
if [[ "$1" == run && "$*" == *'--name'* && "$*" == *'test-evaluation-checkpoint-2'* ]]; then
  touch "$TEST_MARKER"
fi
if [[ "$1" == exec && "$*" == *'benchmark_judge.py'* && "$*" == *'base/samples.json'* ]]; then
  for i in $(seq 1 20); do [[ -e "$TEST_MARKER" ]] && exit 61; sleep 0.05; done
  exit 62
fi
exit 0
""",
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
    }
    for name, contents in scripts.items():
        path = executables / name
        path.write_text(contents)
        path.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(code / source.name), str(run), str(run / "config.json")],
        env={
            **os.environ,
            "PATH": f"{executables}:{os.environ['PATH']}",
            "TEST_MARKER": str(marker),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert marker.exists(), completed.stderr
    assert completed.returncode == 61
    assert (run / "launcher-exit.txt").read_text().strip() == "61"
    assert not (run / "comparison.log").exists()
