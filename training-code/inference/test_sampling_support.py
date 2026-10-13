"""Upstream support survives capture; unusable distributions cannot enter replay."""

import pytest
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput, SamplingMask

from generation import capture_batch


def output(support):
    completion = CompletionOutput(
        index=0,
        text="answer",
        token_ids=[21, 22],
        cumulative_logprob=-1.5,
        logprobs=[{21: Logprob(-0.75)}, {22: Logprob(-0.75)}],
        sampling_mask=None if support is None else SamplingMask(token_ids=support),
        finish_reason="length",
    )
    return RequestOutput(
        request_id="group-1",
        prompt=None,
        prompt_token_ids=[11, 12],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )


def capture(value, **kwargs):
    return capture_batch(
        value,
        [11, 12],
        count=1,
        seconds=0.5,
        logprob_semantics="sampling_processed",
        weight_version="step-1-sha256",
        require_sampling_support=True,
        **kwargs,
    )


def test_capture_preserves_real_upstream_mask_without_aliasing():
    original = output([[21, 4], [22]])
    result = capture(original)
    draw = result["draws"][0]
    assert draw["completion_ids"] == [21, 22]
    assert draw["logprobs"] == [-0.75, -0.75]
    assert draw["sampling_support_ids"] == [[21, 4], [22]]
    original.outputs[0].sampling_mask.token_ids[0].clear()
    assert draw["sampling_support_ids"] == [[21, 4], [22]]


@pytest.mark.parametrize(
    "support",
    [None, [[21]], [[21], [7]], [[21, -1], [22]], [[21, True], [22]], [[21, 21], [22]]],
)
def test_required_mask_rejects_unusable_distributions(support):
    with pytest.raises(ValueError, match="sampling support"):
        capture(output(support))


def test_support_cannot_validate_raw_model_probabilities_for_replay():
    with pytest.raises(ValueError, match="processed"):
        capture_batch(
            output([[21], [22]]),
            [11, 12],
            count=1,
            seconds=0.5,
            logprob_semantics="model_raw",
            weight_version="step-1-sha256",
            require_sampling_support=True,
        )
