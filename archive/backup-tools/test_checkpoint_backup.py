import hashlib
import json

import pytest

import checkpoint_backup


def fixture(tmp_path):
    runtime = tmp_path / "runtime"
    checkpoint = runtime / "runs/visit/update/checkpoint-3"
    checkpoint.mkdir(parents=True)
    contents = {
        "adapter_model.safetensors": b"new adapter",
        "adapter_config.json": b"{}",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng_state.pth": b"rng",
        "trainer_state.json": b'{"global_step":3}',
    }
    for name, content in contents.items():
        (checkpoint / name).write_bytes(content)
    digest = hashlib.sha256(contents["adapter_model.safetensors"]).hexdigest()
    state = {
        "current_step": 3,
        "current_checkpoint": "/results/visit/update/checkpoint-3",
        "current_adapter_sha256": digest,
        "visits": [
            {
                "decision": "update",
                "update": {"global_step": 3, "adapter_sha256": digest},
            }
        ],
    }
    return runtime, checkpoint, state


def test_resume_snapshot_copies_complete_accepted_state_and_preserves_optimizer(
    tmp_path,
):
    runtime, checkpoint, state = fixture(tmp_path)
    out = tmp_path / "snapshot"
    assert hasattr(checkpoint_backup, "prepare_snapshot"), (
        "Checkpoint snapshot support is missing"
    )
    result = checkpoint_backup.prepare_snapshot(runtime, state, out)
    assert (out / "checkpoint/optimizer.pt").read_bytes() == b"optimizer"
    assert json.loads((out / "state.json").read_text()) == state
    assert (
        result["step"] == 3
        and result["adapter_sha256"] == state["current_adapter_sha256"]
    )
    for name, digest in result["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_partial_or_unaccepted_checkpoint_is_not_backed_up_as_completed(tmp_path):
    runtime, checkpoint, state = fixture(tmp_path)
    state["visits"] = []
    with pytest.raises(ValueError, match="accepted"):
        checkpoint_backup.prepare_snapshot(runtime, state, tmp_path / "unaccepted")
    _, _, state = fixture(tmp_path / "second")
    (checkpoint / "rng_state.pth").unlink()
    state["current_checkpoint"] = "/results/visit/update/checkpoint-3"
    with pytest.raises(FileNotFoundError):
        checkpoint_backup.prepare_snapshot(runtime, state, tmp_path / "partial")


def test_changed_adapter_is_rejected_before_snapshot_creation(tmp_path):
    runtime, checkpoint, state = fixture(tmp_path)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"different")
    out = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="hash"):
        checkpoint_backup.prepare_snapshot(runtime, state, out)
    assert not out.exists()
