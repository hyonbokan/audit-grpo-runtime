"""An interrupted cycle resumes in place, reusing only complete phase receipts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cycle_recovery import archive_failed_attempt, completed_phases

CONFIG = {"run_id": "visit", "initial_step": 3, "target_step": 4}


def cycle(tmp_path, **receipts):
    root = tmp_path / "visit"
    root.mkdir()
    (root / "prepared.json").write_text(json.dumps(CONFIG, indent=2))
    for phase, status in receipts.items():
        (root / phase).mkdir()
        if phase == "grading":
            (root / "grading/rewards.json").write_text("[1, 0, 1, 0]")
        else:
            (root / phase / "receipt.json").write_text(json.dumps({"status": status}))
    for name in ("launcher-exit.txt", "last-phase.txt", "cycle-result.json"):
        (root / name).write_text("1\n")
    for name in ("rollout.log", "grading.log", "update.log", "launcher.log"):
        (root / name).write_text(name)
    return root


def test_failed_update_reuses_rollout_and_grades_and_archives_the_rest(tmp_path):
    root = cycle(
        tmp_path,
        rollout="versioned_rollout_saved",
        grading=None,
        update="failed",
    )
    assert completed_phases(root) == {
        "rollout": True,
        "grading": True,
        "update": False,
        "refresh": False,
    }
    archive = archive_failed_attempt(root, CONFIG)
    assert {p.name for p in archive.iterdir()} == {
        "update",
        "update.log",
        "launcher-exit.txt",
        "last-phase.txt",
        "cycle-result.json",
        "resume.json",
    }
    # Reused phases keep their outputs and logs; the controller log keeps appending.
    for name in (
        "rollout",
        "grading",
        "rollout.log",
        "grading.log",
        "launcher.log",
        "prepared.json",
    ):
        assert (root / name).exists()


def test_a_later_receipt_is_not_reused_after_an_incomplete_earlier_phase(tmp_path):
    root = cycle(tmp_path, rollout="failed", update="corrected_update_saved")
    assert completed_phases(root)["update"] is False
    archive = archive_failed_attempt(root, CONFIG)
    assert (archive / "rollout").is_dir() and (archive / "update").is_dir()


def test_resume_rejects_changed_or_unprepared_configuration(tmp_path):
    root = cycle(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        archive_failed_attempt(root, {**CONFIG, "target_step": 5})
    (root / "prepared.json").unlink()
    with pytest.raises(ValueError, match="prepared"):
        archive_failed_attempt(root, CONFIG)


def test_cli_runs_with_only_the_standard_library(tmp_path):
    root = cycle(tmp_path, rollout="versioned_rollout_saved")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG))
    script = Path(__file__).with_name("cycle_recovery.py")
    output = subprocess.run(
        [sys.executable, "-I", str(script), "archive", str(root), str(config)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "reusing ['rollout']" in output
    assert len(list((root / "attempts").iterdir())) == 1
