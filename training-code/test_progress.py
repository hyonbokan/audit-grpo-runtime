import hashlib
import json

import pytest

import progress


def test_shuffled_passes_cover_every_training_row_before_repetition():
    rows = [f"project-{i}" for i in range(20)]
    order = progress.training_order(rows, epochs=2, seed=3407)
    assert set(order[:20]) == set(rows) and set(order[20:]) == set(rows)
    assert order == progress.training_order(rows, epochs=2, seed=3407)
    assert order[:20] != rows


def test_no_contrast_is_a_recorded_skip_and_unknown_is_a_failure():
    assert progress.group_decision([-2, -2, -2, -2]) == "skip_no_reward_contrast"
    assert progress.group_decision([-2, -0.1, -2, -2]) == "update"
    for rewards in [[None, 0, 0, 0], [float("nan"), 0, 0, 0], [0, 0, 0]]:
        with pytest.raises(ValueError):
            progress.group_decision(rewards)


def test_changed_corpus_file_fails_before_any_training(tmp_path):
    (tmp_path / "train.jsonl").write_text("original")
    manifest = {"files": {"train.jsonl": hashlib.sha256(b"original").hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    c = {
        "dataset_host": str(tmp_path),
        "dataset_manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    progress.verify_dataset(c)
    (tmp_path / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        progress.verify_dataset(c)
