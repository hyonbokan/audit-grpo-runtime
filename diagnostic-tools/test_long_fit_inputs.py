import json

import long_fit_inputs as fit
import pytest

TEMPLATE = {
    "row_id": "tiny/all-scope",
    "initial_step": 13,
    "initial_adapter_sha256": "a" * 64,
    "continuation_schedule": {"training_cutoff": "2026-01-01T00:00:00+00:00", "cursor": 0},
    "offload_layers": 0,
}
SETTINGS = {
    "schedule_id": "fit-x",
    "training_cutoff": "2026-02-01T00:00:00+00:00",
    "seconds_limit": 3600,
    "offload_layers": 64,
    "tiled_mlp": True,
    "update_placement": {"gpus": [0, 1], "embedding": "gpu", "max_memory_gib": 30},
}


def draw(prompt: int, completion: int) -> dict:
    return {
        "prompt_ids": list(range(prompt)),
        "completion_ids": [7] * completion,
        "finish_reason": "stop",
    }


def test_draws_are_cut_to_the_target_sequence_length_and_marked_cut():
    built = fit.build(TEMPLATE, SETTINGS, [draw(10, 30), draw(10, 5)], length=25, vocab=100)
    samples = built["samples"]
    assert [len(s["prompt_ids"]) + len(s["completion_ids"]) for s in samples] == [25, 15]
    assert [s["finish_reason"] for s in samples] == ["length", "stop"]
    assert samples[0]["sampling_support_ids"][0] == [7, 8]
    assert samples[0]["row_id"] == "tiny/all-scope"
    assert samples[0]["weight_version"] == "step-13:" + "a" * 64
    assert built["rewards"] == [1.0, 0.0, 0.0, 0.0]


def test_config_takes_the_fit_schedule_settings():
    config = fit.build(TEMPLATE, SETTINGS, [draw(10, 30)], length=25, vocab=100)["config"]
    assert config["schedule_id"] == "fit-x"
    assert config["absolute_deadline"] == SETTINGS["training_cutoff"]
    assert config["continuation_schedule"]["training_cutoff"] == SETTINGS["training_cutoff"]
    assert (config["offload_layers"], config["tiled_mlp"]) == (64, True)
    assert TEMPLATE["offload_layers"] == 0  # the template is not modified


def test_a_length_shorter_than_the_prompt_is_rejected():
    with pytest.raises(ValueError, match="shorter than the prompt"):
        fit.build(TEMPLATE, SETTINGS, [draw(10, 30)], length=10, vocab=100)


def test_seed_moves_staged_phases_into_a_prepared_run(tmp_path):
    run = tmp_path / "fit-run"
    run.mkdir()
    built = fit.build(TEMPLATE, SETTINGS, [draw(10, 30)], length=25, vocab=100)
    fit.write_staged(built, run)
    assert not (run / "rollout").exists()
    fit.seed(run)
    assert json.loads((run / "grading/rewards.json").read_text()) == [1.0, 0.0, 0.0, 0.0]
    assert json.loads((run / "rollout/receipt.json").read_text())["synthetic_fit_input"] is True
    assert not fit.staging(run).exists()
