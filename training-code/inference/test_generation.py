"""Engine-boundary capture and request validation; no GPU or engine simulation."""

import hashlib
import json
from types import SimpleNamespace

import pytest

import generation


def request(prompt=(11, 12, 13), *, context=36864):
    ids = list(prompt)
    return {
        "prompt_ids": ids,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "model": {"precision": "bf16"},
        "full_completion_ceiling": 16384,
        "max_model_len": context,
    }


def test_request_validation_accepts_complete_native_request():
    generation.validate_request(request())


@pytest.mark.parametrize(
    "change, message",
    [
        ({"prompt_ids": [99, 12, 13]}, "hash"),
        ({"max_model_len": 1000}, "whole prompt"),
        ({"model": {"precision": "fp8"}}, "BF16"),
    ],
)
def test_request_validation_rejects_changed_or_oversize_requests(change, message):
    with pytest.raises(ValueError, match=message):
        generation.validate_request({**request(), **change})


def test_request_validation_rejects_empty_prompt():
    with pytest.raises(ValueError, match="prompt tokens"):
        generation.validate_request(request(prompt=()))


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
    result = generation.capture_batch(response(), [11, 12, 13], count=1, seconds=0.5)
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
        generation.capture_batch(output, [11, 12, 13], count=count, seconds=0.5)


def test_capture_labels_probability_semantics_and_the_policy_weight_version():
    result = generation.capture_batch(
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
        generation.capture_batch(
            response(),
            [11, 12, 13],
            count=1,
            seconds=0.5,
            logprob_semantics="logits",
            weight_version="adapter-v1",
        )
    with pytest.raises(ValueError, match="weight version"):
        generation.capture_batch(
            response(), [11, 12, 13], count=1, seconds=0.5, logprob_semantics="model_raw"
        )
