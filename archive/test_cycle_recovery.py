"""Saved rollouts retain their actual sampling and weight identity during recovery."""

import hashlib
import json

import pytest


def saved_cycle(tmp_path):
    old = tmp_path / "old"
    rollout = old / "rollout"
    rollout.mkdir(parents=True)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"prompt_ids": [10], "row_id": "project/all-scope"}))
    config = {
        "initial_step": 2,
        "initial_adapter_sha256": "a" * 64,
        "sampling": {"n": 4, "max_tokens": 10},
        "engine": {"dtype": "bfloat16"},
        "row_id": "project/all-scope",
        "request": str(request),
    }
    version = "step-2:" + "a" * 64
    (old / "prepared.json").write_text(json.dumps(config))
    samples = [
        {
            "row_id": "project/all-scope",
            "prompt_ids": [10],
            "completion_ids": [20],
            "logprobs": [-0.5],
            "sampling_support_ids": [[20]],
            "weight_version": version,
            "logprob_semantics": "sampling_processed",
        }
        for _ in range(4)
    ]
    (rollout / "samples.json").write_text(json.dumps(samples))
    (rollout / "batch.json").write_text(json.dumps({"completion_tokens": 4}))
    (rollout / "receipt.json").write_text(
        json.dumps(
            {
                "status": "versioned_rollout_saved",
                "weight_version": version,
                "transfer": {"cast_transfer_verified": True},
                "completion_tokens": 4,
            }
        )
    )
    files = ["prepared.json", "rollout/samples.json", "rollout/batch.json", "rollout/receipt.json"]
    config["rollout_recovery"] = {
        "source": str(old),
        "sha256": {name: hashlib.sha256((old / name).read_bytes()).hexdigest() for name in files},
    }
    return config, old


def test_recovery_copies_exact_saved_rollout_and_binds_its_provenance(tmp_path):
    from cycle_recovery import recover_rollout

    config, old = saved_cycle(tmp_path)
    out = tmp_path / "new"
    out.mkdir()
    recover_rollout(config, out)
    assert (out / "rollout/samples.json").read_bytes() == (
        old / "rollout/samples.json"
    ).read_bytes()
    assert json.loads((out / "rollout-recovery.json").read_text())["source"] == str(old)


@pytest.mark.parametrize("change", ["sampling", "weights", "samples"])
def test_recovery_rejects_changed_contract_or_saved_bytes_before_copy(tmp_path, change):
    from cycle_recovery import recover_rollout

    config, old = saved_cycle(tmp_path)
    if change == "sampling":
        config["sampling"]["max_tokens"] = 9
    elif change == "weights":
        config["initial_adapter_sha256"] = "b" * 64
    else:
        (old / "rollout/samples.json").write_text("[]")
    out = tmp_path / "new"
    out.mkdir()
    with pytest.raises(ValueError):
        recover_rollout(config, out)
    assert not (out / "rollout").exists()
