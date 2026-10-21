"""launch-training.sh against a fake Docker: visit failures, backups, resume, evaluation."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from testing.host import write_host_settings

CODE = Path(__file__).parent


class Runtime:
    """A runtime tree with the real launcher, a stubbed cycle launcher and a scripted docker."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.code = tmp_path / "runtime/training-code"
        self.code.mkdir(parents=True)
        for name in ("launch-training.sh", "launch-lib.sh"):
            shutil.copyfile(CODE / name, self.code / name)
        write_host_settings(tmp_path / "runtime")
        self.run = tmp_path / "runtime/results/run"
        self.run.mkdir(parents=True)
        self.config = self.run / "config.json"
        self.config.write_text("{}")
        digest = hashlib.sha256((self.code / "launch-training.sh").read_bytes()).hexdigest()
        (self.run / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-training.sh\n")
        self.visit = tmp_path / "runtime/results/run-visit-001"
        self.visit.mkdir()
        (self.visit / "config.json").write_text("{}")
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.calls = tmp_path / "docker-calls"
        self.cycle_calls = tmp_path / "cycle-calls"

    def cycle(self, status: int, *, prepare: bool = False) -> None:
        prepared = 'touch "$1/prepared.json"\n' if prepare else ""
        self.script(
            self.code / "launch-cycle.sh",
            f'echo "$*" >> {self.cycle_calls}\n{prepared}exit {status}\n',
        )

    def docker(self, cases: dict[str, str]) -> None:
        """Each case maps a substring of the docker arguments to the shell it runs."""
        arms = "".join(f'  *"{pattern}"*) {action} ;;\n' for pattern, action in cases.items())
        self.script(
            self.bin / "docker",
            f'echo "$*" >> {self.calls}\ncase "$*" in\n{arms}  *) exit 0 ;;\nesac\n',
        )

    def launch(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "bash",
                str(self.code / "launch-training.sh"),
                str(self.run),
                str(self.config),
                *extra,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}"},
        )

    def read(self, name: str) -> str:
        return (self.run / name).read_text().strip()

    def docker_calls(self) -> str:
        return self.calls.read_text() if self.calls.exists() else ""

    def one_visit_then_complete(self) -> str:
        """`controller next` returns the visit once, then reports the schedule complete."""
        seen = self.tmp / "next-called"
        return f"if [ -e {seen} ]; then echo complete; else touch {seen}; echo {self.visit}; fi"

    @staticmethod
    def script(path: Path, body: str) -> None:
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)


@pytest.fixture
def runtime(tmp_path):
    return Runtime(tmp_path)


NOT_TRAINED = 'printf "cp\\nmerge\\n3\\nfalse\\n"'


def test_a_visit_that_hits_its_time_limit_is_recorded_and_training_continues(runtime):
    runtime.cycle(124)
    recorded = runtime.tmp / "recorded"
    runtime.docker(
        {
            "-m controller next": runtime.one_visit_then_complete(),
            "-m controller failure-policy": 'printf "1\\n1\\n"',
            "-m controller record-failure": f"touch {recorded}",
            "-m controller identity": NOT_TRAINED,
        }
    )
    result = runtime.launch()
    assert recorded.exists(), result.stdout + result.stderr
    assert "training-failed-124" in result.stdout
    assert runtime.read("training-exit.txt") == "0"


@pytest.mark.parametrize("status", [130, 143])
def test_an_operator_stop_ends_training_without_recording_a_failed_visit(runtime, status):
    runtime.cycle(status)
    runtime.docker(
        {
            "-m controller next": runtime.one_visit_then_complete(),
            "-m controller failure-policy": 'printf "1\\n1\\n"',
            "-m controller identity": NOT_TRAINED,
        }
    )
    runtime.launch()
    assert "record-failure" not in runtime.docker_calls()
    assert runtime.read("training-exit.txt") == "1"


def test_a_failed_visit_backup_is_reported_but_does_not_end_training(runtime):
    runtime.cycle(0)
    runtime.docker(
        {
            "-m controller next": runtime.one_visit_then_complete(),
            "-m controller failure-policy": 'printf "1\\n1\\n"',
            "-m controller identity": NOT_TRAINED,
            '-m hf_backup"*"--label visit-': "exit 1",
        }
    )
    result = runtime.launch()
    assert "backup-failed-visit-run-visit-001" in result.stdout, result.stdout + result.stderr
    assert "visit-run-visit-001 1" in runtime.read("backup-failures.txt")
    assert runtime.read("training-exit.txt") == "0"
    calls = runtime.docker_calls().splitlines()
    discard = next(i for i, c in enumerate(calls) if "--phase discard" in c)
    assert "--label visit-run-visit-001" in calls[discard]
    assert discard < next(i for i, c in enumerate(calls) if "-m controller prune" in c)


def test_every_recorded_visit_is_pruned_after_its_backup(runtime):
    runtime.cycle(0)
    runtime.docker(
        {
            "-m controller next": runtime.one_visit_then_complete(),
            "-m controller failure-policy": 'printf "1\\n1\\n"',
            "-m controller identity": NOT_TRAINED,
            "-m controller prune": 'echo \'{"files": 3, "bytes": 900}\'',
        }
    )
    result = runtime.launch()
    assert result.returncode == 1, result.stdout + result.stderr
    calls = runtime.docker_calls().splitlines()
    upload = next(i for i, c in enumerate(calls) if "--phase upload" in c and "visit-" in c)
    assert upload < next(i for i, c in enumerate(calls) if "-m controller prune" in c)
    assert "--phase discard" not in runtime.docker_calls()
    assert runtime.read("retention.log") == '{"files": 3, "bytes": 900}'


def test_a_failed_final_backup_stops_the_run_before_evaluation(runtime):
    runtime.cycle(0)
    runtime.docker(
        {
            "-m controller next": runtime.one_visit_then_complete(),
            "-m controller failure-policy": 'printf "1\\n1\\n"',
            '-m hf_backup"*"--label training-final': "exit 1",
        }
    )
    result = runtime.launch()
    assert result.returncode != 0
    assert runtime.read("training-exit.txt") == "0"
    assert runtime.read("last-phase.txt") == "backup-training-final"
    assert "-m controller identity" not in runtime.docker_calls()


def test_failed_training_with_stop_on_failure_never_evaluates(runtime):
    runtime.cycle(1)
    runtime.docker(
        {
            "-m controller next": f"echo {runtime.visit}",
            "-m controller failure-policy": 'printf "1\\n0\\n"',
            "-m controller identity": NOT_TRAINED,
        }
    )
    result = runtime.launch()
    assert result.returncode == 1, result.stdout + result.stderr
    assert "-m controller evaluation" not in runtime.docker_calls()
    assert runtime.read("last-phase.txt") == "training-unaccepted"


def test_failed_visit_is_resumed_once_then_skipped_without_evaluating(runtime):
    runtime.cycle(1, prepare=True)
    skipped = runtime.tmp / "skipped"
    runtime.docker(
        {
            "-m controller next": f"[[ -e {skipped} ]] && echo complete || echo {runtime.visit}",
            "-m controller failure-policy": 'printf "2\\n1\\n"',
            "-m controller record-failure --attempts 2": f"touch {skipped}",
            "-m controller identity": NOT_TRAINED,
        }
    )
    result = runtime.launch()
    assert result.returncode == 1, result.stdout + result.stderr
    visit = runtime.visit
    assert runtime.cycle_calls.read_text().splitlines() == [
        f"{visit} {visit}/config.json",
        f"{visit} {visit}/config.json --resume",
    ]
    assert skipped.exists(), "Failed visit was not recorded before moving on"
    assert "-m controller evaluation" not in runtime.docker_calls()
    assert runtime.read("last-phase.txt") == "training-unaccepted"


def test_resume_continues_from_saved_state_and_refuses_a_finished_run(runtime):
    (runtime.run / "state.json").write_text("{}")
    (runtime.run / "launcher-exit.txt").write_text("143\n")
    runtime.docker(
        {
            "-m controller next": "echo complete",
            "-m controller failure-policy": 'printf "2\\n1\\n"',
            "-m controller identity": NOT_TRAINED,
        }
    )
    result = runtime.launch("--resume")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "-m controller prepare" not in runtime.docker_calls()
    assert "--label initial" not in runtime.docker_calls()
    assert (runtime.run / "controller-attempt-1/launcher-exit.txt").read_text() == "143\n"
    again = runtime.launch("--resume")
    assert again.returncode == 2 and "already finished" in again.stderr


def test_accepted_training_merges_the_final_checkpoint_and_runs_the_suite(runtime):
    runtime.config.write_text(
        json.dumps({"cycle": {"model_path": "/models/qwen", "model_revision": "abc123"}})
    )
    suite = runtime.tmp / "runtime/results/run-evaluation"
    suite.mkdir()
    suite_calls = runtime.tmp / "suite-calls"
    runtime.script(
        runtime.code / "launch-benchmark-suite.sh", f'echo "$*" >> {suite_calls}\nexit 0\n'
    )
    runtime.docker(
        {
            "-m controller next": "echo complete",
            "-m controller failure-policy": 'printf "1\\n1\\n"',
            "-m controller identity": 'printf "/results/cp4\\n/workspace/cache/m4\\n4\\ntrue\\n"',
            "-m controller evaluation": f"echo {suite}",
        }
    )
    result = runtime.launch()
    assert result.returncode == 0, result.stdout + result.stderr
    merge = next(line for line in runtime.docker_calls().splitlines() if "merge_checkpoint" in line)
    assert "--base /models/qwen --adapter /results/cp4 --output /workspace/cache/m4" in merge
    assert "--revision abc123" in merge
    assert suite_calls.read_text().split() == [str(suite), f"{suite}/config.json"]
    assert runtime.read("last-phase.txt") == "complete"
    calls = runtime.docker_calls().splitlines()
    release = next(i for i, c in enumerate(calls) if "-m controller release-final-merge" in c)
    assert release < next(i for i, c in enumerate(calls) if "--label evaluated-final" in c)
