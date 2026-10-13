"""Exercise preparation through its real CLI; no GPU or engine simulation."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import generation


def prepare(tmp_path, *, context=36864, precision="bf16", prompt=(11, 12, 13)):
    # Rejecting oversize inputs must never silently shorten the code prompt.
    plan = {
        "model": {"id": "Qwen/Qwen3.8-27B", "revision": "a" * 40, "precision": precision},
        "training": {"max_completion_tokens": 16384},
        "sampling_profile": {"id": "schema-low-16k-v1", "policy_reasoning_effort": "low"},
    }
    sample = {"row_id": "2022-12-caviar/all-scope", "prompt_ids": list(prompt)}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "sample.json").write_text(json.dumps(sample))
    out = tmp_path / "request.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("benchmark.py")),
            "prepare",
            "--plan",
            str(tmp_path / "plan.json"),
            "--sample",
            str(tmp_path / "sample.json"),
            "--out",
            str(out),
            "--context",
            str(context),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, out


def test_prepare_preserves_every_native_prompt_token(tmp_path):
    result, out = prepare(tmp_path)
    assert result.returncode == 0, result.stderr
    request = json.loads(out.read_text())
    assert request["prompt_ids"] == [11, 12, 13]
    assert request["max_model_len"] == 36864
    assert request["sampling"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 512,
        "logprobs": 1,
        "seed": 3407,
    }
    assert request["full_completion_budget_tested"] is False
    assert request["judge_calls"] == request["optimizer_steps"] == 0


@pytest.mark.parametrize(
    "options, message",
    [
        ({"context": 16386}, "whole prompt"),
        ({"precision": "fp8"}, "BF16"),
        ({"prompt": ()}, "prompt tokens"),
    ],
)
def test_invalid_request_is_rejected_without_writing_output(tmp_path, options, message):
    result, out = prepare(tmp_path, **options)
    assert result.returncode != 0
    assert message in result.stderr
    assert not out.exists()


def test_preparation_never_overwrites_an_existing_request(tmp_path):
    assert prepare(tmp_path)[0].returncode == 0
    result, out = prepare(tmp_path, prompt=(99,))
    assert result.returncode != 0
    assert json.loads(out.read_text())["prompt_ids"] == [11, 12, 13]


def response(*, prompt=(11, 12, 13), missing_logprob=False):
    # Data at the engine boundary, not a simulated generation implementation.
    chosen = SimpleNamespace(logprob=-0.75, rank=1, decoded_token="a")
    return SimpleNamespace(
        prompt_token_ids=list(prompt),
        outputs=[
            SimpleNamespace(
                index=0,
                text="ab",
                token_ids=[21, 22],
                finish_reason="length",
                stop_reason=None,
                cumulative_logprob=-1.5,
                logprobs=[{} if missing_logprob else {21: chosen}, {22: chosen}],
            )
        ],
    )


def test_capture_retains_chosen_tokens_and_measures_aggregate_throughput():
    result = benchmark.capture_batch(response(), [11, 12, 13], count=1, seconds=0.5)
    assert result["completion_tokens"] == 2
    assert result["aggregate_tokens_per_second"] == 4
    assert result["draws"][0]["completion_ids"] == [21, 22]
    assert result["draws"][0]["logprobs"] == [-0.75, -0.75]
    assert result["draws"][0]["raw_text"] == "ab"
    assert result["logprob_distribution_verified"] is False


@pytest.mark.parametrize(
    "output, count, message",
    [
        (response(prompt=(11, 12)), 1, "prompt"),
        (response(), 2, "count"),
        (response(missing_logprob=True), 1, "logprob"),
    ],
)
def test_capture_rejects_engine_outputs_that_break_the_sample_join(output, count, message):
    with pytest.raises(ValueError, match=message):
        benchmark.capture_batch(output, [11, 12, 13], count=count, seconds=0.5)


def test_request_validation_detects_changed_prompt_before_engine_loading(tmp_path):
    result, out = prepare(tmp_path)
    assert result.returncode == 0
    request = json.loads(out.read_text())
    request["prompt_ids"][0] = 99
    with pytest.raises(ValueError, match="hash"):
        benchmark.validate_request(request)


def test_request_validation_checks_full_completion_ceiling(tmp_path):
    result, out = prepare(tmp_path)
    assert result.returncode == 0
    request = json.loads(out.read_text())
    request["max_model_len"] = 1000
    with pytest.raises(ValueError, match="whole prompt"):
        benchmark.validate_request(request)


def test_capture_labels_probability_semantics_and_the_policy_weight_version():
    result = benchmark.capture_batch(
        response(),
        [11, 12, 13],
        count=1,
        seconds=0.5,
        logprob_semantics="sampling_processed",
        weight_version="adapter-v1",
    )
    assert result["draws"][0]["logprob_semantics"] == "sampling_processed"
    assert result["draws"][0]["weight_version"] == "adapter-v1"
    assert result["draws"][0]["raw_logprobs"] is None
    assert result["draws"][0]["sampling_support_ids"] is None


def test_capture_rejects_an_unknown_probability_contract():
    with pytest.raises(ValueError, match="semantics"):
        benchmark.capture_batch(
            response(),
            [11, 12, 13],
            count=1,
            seconds=0.5,
            logprob_semantics="logits",
            weight_version="adapter-v1",
        )
    with pytest.raises(ValueError, match="weight version"):
        benchmark.capture_batch(
            response(), [11, 12, 13], count=1, seconds=0.5, logprob_semantics="model_raw"
        )
