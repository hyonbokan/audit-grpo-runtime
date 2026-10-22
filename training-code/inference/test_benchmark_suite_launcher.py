"""Suite orchestration: one shared engine, per-project grading, failures kept per project."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from testing.host import write_host_settings

SETTINGS = (
    'printf "native-image\\nrollout-image\\n100\\n50\\ncheckpoint-7\\n%s\\n" "$TEST_PROJECTS"'
)


class Suite:
    """A runtime tree with the real suite launcher and a scripted docker."""

    def __init__(self, tmp_path: Path, projects: str):
        self.tmp = tmp_path
        root = tmp_path / "runtime"
        self.code = root / "training-code"
        self.code.mkdir(parents=True)
        source = Path(__file__).parents[1] / "launch-benchmark-suite.sh"
        shutil.copyfile(source, self.code / source.name)
        shutil.copyfile(source.with_name("launch-lib.sh"), self.code / "launch-lib.sh")
        self.cache = write_host_settings(root)
        self.results = root / "results"
        self.out = self.results / "suite"
        self.out.mkdir(parents=True)
        (self.out / "config.json").write_text("{}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        (self.out / "SOURCE_SHA256SUMS").write_text(f"{digest}  launch-benchmark-suite.sh\n")
        for name in projects.split():
            (self.results / name).mkdir()
        self.projects = projects
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.calls = tmp_path / "docker-calls"
        script(self.bin / "nvidia-smi", "exit 0\n")

    def docker(self, cases: dict[str, str]) -> None:
        arms = "".join(f'  *"{pattern}"*) {action} ;;\n' for pattern, action in cases.items())
        script(
            self.bin / "docker",
            f'echo "$*" >> {self.calls}\ncase "$*" in\n'
            f'  *"benchmark_suite.py settings"*) {SETTINGS} ;;\n'
            f'  *"benchmark_suite.py manifest"*) echo /results/suite/rollout-manifest.json ;;\n'
            f"{arms}  *) exit 0 ;;\nesac\n",
        )

    def launch(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "bash",
                str(self.code / "launch-benchmark-suite.sh"),
                str(self.out),
                str(self.out / "config.json"),
            ],
            env={
                **os.environ,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "TEST_PROJECTS": self.projects,
            },
            text=True,
            capture_output=True,
            timeout=20,
        )

    def call_lines(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []


def script(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def test_every_project_samples_in_one_engine_then_grades_and_finishes(tmp_path):
    suite = Suite(tmp_path, "run-a run-b")
    suite.docker({})
    result = suite.launch()
    assert result.returncode == 0, result.stderr
    calls = suite.call_lines()
    gpu = [c for c in calls if "--gpus" in c]
    assert len(gpu) == 1 and "rollout-image" in gpu[0]
    assert "rollout.py evaluate-suite --manifest /results/suite/rollout-manifest.json" in gpu[0]
    prepares = [i for i, c in enumerate(calls) if "benchmark_evaluation.py prepare" in c]
    gradings = [i for i, c in enumerate(calls) if "benchmark_judge.py" in c]
    finishes = [i for i, c in enumerate(calls) if "benchmark_evaluation.py finish" in c]
    rollout = calls.index(gpu[0])
    assert len(prepares) == len(gradings) == len(finishes) == 2
    assert max(prepares) < rollout < min(gradings)
    for name, grading, finish in zip(("run-a", "run-b"), gradings, finishes, strict=True):
        assert f"--samples /results/{name}/checkpoint-7/samples.json" in calls[grading]
        assert grading < finish and f"--out /results/{name}" in calls[finish]
        assert (suite.results / name / "launcher-exit.txt").read_text() == "0\n"
    assert (suite.out / "project-exits.tsv").read_text() == "run-a\t0\nrun-b\t0\n"
    assert (suite.out / "suite-exit.txt").read_text() == "0\n"
    assert "benchmark_suite.py report" in calls[-1]
    for call in calls:
        if call.startswith("run "):
            assert f"-v {suite.cache}:{suite.cache}" in call


def test_a_project_whose_grading_fails_is_recorded_while_the_others_finish(tmp_path):
    suite = Suite(tmp_path, "run-a run-b")
    suite.docker({'benchmark_judge.py"*"/results/run-a/': "exit 3"})
    result = suite.launch()
    assert result.returncode == 0, result.stderr
    assert (suite.out / "project-exits.tsv").read_text() == "run-a\t3\nrun-b\t0\n"
    finishes = [c for c in suite.call_lines() if "benchmark_evaluation.py finish" in c]
    assert len(finishes) == 1 and "/results/run-b" in finishes[0]


@pytest.mark.parametrize("failing", ["evaluate-suite", "benchmark_evaluation.py prepare"])
def test_a_failed_shared_engine_or_preparation_stops_before_grading(tmp_path, failing):
    suite = Suite(tmp_path, "run-a run-b")
    suite.docker({failing: "exit 9"})
    result = suite.launch()
    assert result.returncode == 9
    assert not [c for c in suite.call_lines() if "benchmark_judge.py" in c]
    assert (suite.out / "suite-exit.txt").read_text() == "9\n"
    assert "benchmark_suite.py report" in suite.call_lines()[-1]


def test_a_suite_with_nothing_left_to_run_only_reports(tmp_path):
    suite = Suite(tmp_path, "")
    suite.docker({})
    result = suite.launch()
    assert result.returncode == 0, result.stderr
    calls = suite.call_lines()
    assert not [c for c in calls if "--gpus" in c or "benchmark_evaluation.py" in c]
    assert "benchmark_suite.py report" in calls[-1]
