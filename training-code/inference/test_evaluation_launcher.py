"""launch-evaluation.sh against a fake Docker: arm order, grading mode and failures."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from testing.host import write_host_settings

CODE = Path(__file__).parents[1]


def launch(tmp_path, docker, **environment):
    """Run the real launcher in a runtime tree whose docker is the given script."""
    root = tmp_path / "runtime"
    code = root / "training-code"
    code.mkdir(parents=True)
    run = root / "results/test-evaluation"
    run.mkdir(parents=True)
    for name in ("launch-evaluation.sh", "launch-lib.sh"):
        shutil.copyfile(CODE / name, code / name)
    write_host_settings(root)
    digest = hashlib.sha256((code / "launch-evaluation.sh").read_bytes()).hexdigest()
    (run / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-evaluation.sh\n")
    (run / "config.json").write_text("{}")
    executables = tmp_path / "bin"
    executables.mkdir()
    for name, contents in {"docker": docker, "nvidia-smi": "#!/bin/bash\nexit 0\n"}.items():
        path = executables / name
        path.write_text(contents)
        path.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(code / "launch-evaluation.sh"), str(run), str(run / "config.json")],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}", **environment},
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed, run


def settings(grading):
    return f"""if [[ "$1" == run && "$*" == *'benchmark_evaluation.py settings'* ]]; then
  printf '%s\\n' native rollout 20 10 'base checkpoint-2' {grading}
  exit 0
fi
"""


def test_failed_generation_prevents_grading_and_second_arm(tmp_path):
    docker = f"""#!/bin/bash
{settings("serial")}
case "$1" in
run)
  for arg in "$@"; do
    if [[ "$arg" == --name ]]; then echo 'generation failed'; exit 57; fi
  done
  exit 0;;
inspect) echo '{{"Status":"exited","ExitCode":57}}';;
logs) echo 'generation failed';;
stop) exit 0;;
*) echo 'unexpected phase' >&2; exit 99;;
esac
"""
    completed, run = launch(tmp_path, docker)
    assert completed.returncode == 57, completed.stderr
    assert (run / "launcher-exit.txt").read_text().strip() == "57"
    assert (run / "last-phase.txt").read_text().strip() == "base"
    assert "generation failed" in (run / "base.log").read_text()
    assert not (run / "base-grading.log").exists()
    assert not (run / "checkpoint-2.log").exists()


def test_each_arm_is_generated_then_graded_on_its_own_samples_before_the_comparison(tmp_path):
    commands = tmp_path / "commands.txt"
    docker = f"""#!/bin/bash
printf '%s\\n' "$*" >> "$TEST_COMMANDS"
{settings("serial")}
exit 0
"""
    completed, run = launch(tmp_path, docker, TEST_COMMANDS=str(commands))
    assert completed.returncode == 0, completed.stderr
    trace = commands.read_text().splitlines()
    steps = [
        next(f"generate {label}" for label in ("base", "checkpoint-2") if f"-{label} " in line)
        if "--name" in line
        else f"grade {line.split('--samples ')[1].split()[0]}"
        if "benchmark_judge.py" in line
        else "finish"
        if "benchmark_evaluation.py finish" in line
        else None
        for line in trace
    ]
    assert [step for step in steps if step] == [
        "generate base",
        "grade /results/test-evaluation/base/samples.json",
        "generate checkpoint-2",
        "grade /results/test-evaluation/checkpoint-2/samples.json",
        "finish",
    ]
    assert (run / "last-phase.txt").read_text().strip() == "complete"


def test_paired_grading_can_overlap_second_generation_and_is_joined(tmp_path):
    marker = tmp_path / "second-generation"
    docker = f"""#!/bin/bash
{settings("overlap")}
if [[ "$1" == run && "$*" == *'--name'* && "$*" == *'test-evaluation-checkpoint-2'* ]]; then
  touch "$TEST_MARKER"
fi
if [[ "$1" == exec && "$*" == *'benchmark_judge.py'* && "$*" == *'base/samples.json'* ]]; then
  for i in $(seq 1 20); do [[ -e "$TEST_MARKER" ]] && exit 61; sleep 0.05; done
  exit 62
fi
exit 0
"""
    completed, run = launch(tmp_path, docker, TEST_MARKER=str(marker))
    assert marker.exists(), completed.stderr
    assert completed.returncode == 61
    assert (run / "launcher-exit.txt").read_text().strip() == "61"
    assert not (run / "comparison.log").exists()
