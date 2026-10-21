"""Disk retention: resume-only state, rollout adapter copies, merges and free space."""

import json
from pathlib import Path

import pytest

import retention

RESUME_FILES = ("optimizer.pt", "scheduler.pt", "rng_state.pth")
KEPT_FILES = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")


def write_visit(results: Path, index: int, step: int | None) -> dict:
    """One recorded visit; `step` None is a skipped visit with no update."""
    root = results / f"run-visit-{index:03d}"
    for phase in ("rollout", "refresh"):
        mapped = root / phase / "mapped-adapter"
        mapped.mkdir(parents=True)
        (mapped / "adapter_model.safetensors").write_bytes(b"m" * 10)
        (root / phase / "receipt.json").write_text("{}")
    visit = {"root": str(root), "decision": "skip_no_reward_contrast"}
    if step is not None:
        checkpoint = root / "update" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in RESUME_FILES + KEPT_FILES:
            (checkpoint / name).write_bytes(b"x" * 100)
        visit.update(decision="update", update={"global_step": step})
    return visit


@pytest.fixture
def run(tmp_path):
    results = tmp_path / "results"
    visits = [
        write_visit(results, 1, 3),
        write_visit(results, 2, None),
        write_visit(results, 3, 4),
        write_visit(results, 4, 5),
    ]
    state = {
        "visits": visits,
        "current_checkpoint": str(results / "run-visit-004/update/checkpoint-5"),
    }
    return results, state


def test_prune_keeps_resume_state_for_the_latest_accepted_checkpoints_only(run):
    results, state = run
    removed = retention.prune_visits(state, resume_states_kept=2)
    oldest = results / "run-visit-001/update/checkpoint-3"
    assert sorted(p.name for p in oldest.iterdir()) == sorted(KEPT_FILES)
    for kept in ("run-visit-003/update/checkpoint-4", "run-visit-004/update/checkpoint-5"):
        assert {p.name for p in (results / kept).iterdir()} == set(RESUME_FILES + KEPT_FILES)
    assert {entry["path"] for entry in removed} >= {str(oldest / name) for name in RESUME_FILES}


def test_prune_removes_every_rollout_and_refresh_adapter_copy_but_keeps_receipts(run):
    results, state = run
    removed = retention.prune_visits(state, resume_states_kept=2)
    for index in range(1, 5):
        for phase in ("rollout", "refresh"):
            assert not (results / f"run-visit-{index:03d}/{phase}/mapped-adapter").exists()
            assert (results / f"run-visit-{index:03d}/{phase}/receipt.json").is_file()
    assert sum(entry["bytes"] for entry in removed) == 3 * 100 + 8 * 10


def test_prune_never_removes_the_current_checkpoints_resume_state(run):
    results, state = run
    retention.prune_visits(state, resume_states_kept=1)
    current = Path(state["current_checkpoint"])
    assert {p.name for p in current.iterdir()} == set(RESUME_FILES + KEPT_FILES)
    previous = results / "run-visit-003/update/checkpoint-4"
    assert sorted(p.name for p in previous.iterdir()) == sorted(KEPT_FILES)


def test_prune_twice_removes_nothing_more(run):
    _results, state = run
    retention.prune_visits(state, resume_states_kept=2)
    assert retention.prune_visits(state, resume_states_kept=2) == []


@pytest.mark.parametrize("kept", [0, -1, True, 1.5])
def test_prune_rejects_a_policy_that_could_drop_the_current_resume_state(run, kept):
    _results, state = run
    with pytest.raises(ValueError, match="resume_states_kept"):
        retention.prune_visits(state, resume_states_kept=kept)


def test_release_merge_keeps_its_manifest_as_the_receipt_and_deletes_the_weights(tmp_path):
    merged = tmp_path / "cache/merges/run-step-5-abc"
    merged.mkdir(parents=True)
    manifest = {"status": "merged_and_verified", "adapter_sha256": "abc", "files": {}}
    (merged / "MERGE_MANIFEST.json").write_text(json.dumps(manifest))
    (merged / "model-00001-of-00002.safetensors").write_bytes(b"w" * 1000)
    receipt = tmp_path / "run/merge-receipts/run-step-5-abc.json"
    released = retention.release_merge(merged, receipt)
    assert not merged.exists()
    assert json.loads(receipt.read_text())["manifest"] == manifest
    assert released["bytes"] == 1000 + len(json.dumps(manifest))


def test_release_merge_of_a_missing_directory_records_nothing(tmp_path):
    receipt = tmp_path / "receipt.json"
    assert retention.release_merge(tmp_path / "absent", receipt) is None
    assert not receipt.exists()


def test_release_merge_refuses_a_directory_without_a_verified_manifest(tmp_path):
    merged = tmp_path / "partial"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"w")
    with pytest.raises(ValueError, match="verified merge"):
        retention.release_merge(merged, tmp_path / "receipt.json")
    assert (merged / "model.safetensors").is_file()


def test_free_space_check_names_the_path_and_both_sizes(tmp_path, monkeypatch):
    free = 30 * 2**30
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda _path: (0, 0, free))
    retention.require_free_space([tmp_path], minimum_gib=30)
    with pytest.raises(OSError, match=rf"{tmp_path}.*30\.0 GiB free.*31 GiB"):
        retention.require_free_space([tmp_path], minimum_gib=31)


def test_free_space_check_measures_the_nearest_existing_parent(tmp_path, monkeypatch):
    measured = []
    monkeypatch.setattr(
        retention.shutil, "disk_usage", lambda path: measured.append(path) or (0, 0, 2**40)
    )
    retention.require_free_space([tmp_path / "not/yet/created"], minimum_gib=1)
    assert measured == [tmp_path]
