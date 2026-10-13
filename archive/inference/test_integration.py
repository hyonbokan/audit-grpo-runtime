import json

import pytest

import integration


def test_checkpoint_gate_rejects_unverified_update(tmp_path):
    (tmp_path / "step_metrics.json").write_text(json.dumps({
        "global_step": 0, "adapter_weights_changed": False,
    }))
    with pytest.raises(ValueError, match="update"):
        integration.check_checkpoint(tmp_path)


def test_checkpoint_gate_requires_full_trainer_state(tmp_path):
    (tmp_path / "step_metrics.json").write_text(json.dumps({
        "global_step": 1, "adapter_weights_changed": True,
    }))
    with pytest.raises(FileNotFoundError):
        integration.check_checkpoint(tmp_path)


def test_adapter_probe_detects_probability_change_with_same_tokens():
    first = {"draws": [{"completion_ids": [4, 5], "logprobs": [-0.8, -0.9]}]}
    second = {"draws": [{"completion_ids": [4, 5], "logprobs": [-0.81, -0.9]}]}
    assert integration.outputs_changed(first, second)
    assert not integration.outputs_changed(first, first)


def test_repeat_tokens_separate_from_probability_stability():
    from integration import repeated_token_ids

    first = {"draws": [{"completion_ids": [1, 2], "logprobs": [-0.2, -0.1]}]}
    noisy = {"draws": [{"completion_ids": [1, 2], "logprobs": [-0.25, -0.1]}]}
    changed = {"draws": [{"completion_ids": [1, 3], "logprobs": [-0.2, -0.1]}]}
    assert repeated_token_ids(first, noisy)
    assert not repeated_token_ids(first, changed)
    assert integration.outputs_changed(first, noisy)
