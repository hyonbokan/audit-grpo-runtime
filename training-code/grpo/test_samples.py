"""Reject mislabeled probability sidecars without changing legacy artifacts."""

import pytest

from grpo.samples import capture_sample


class Tokenizer:
    def decode(self, _ids, **_options):
        return "answer"


def test_capture_rejects_swapped_raw_and_sampling_probabilities():
    with pytest.raises(ValueError, match=r"raw.*sampling"):
        capture_sample(
            "row",
            [1],
            [2],
            [-0.7],
            Tokenizer(),
            finish_reason="stop",
            raw_logprobs=[-0.1],
            sampling_support_ids=[[2]],
        )


@pytest.mark.parametrize("raw, support", [([], [[2]]), ([-0.9], []), ([-0.9], [[3]])])
def test_capture_rejects_incomplete_or_unjoined_probability_sidecars(raw, support):
    with pytest.raises(ValueError):
        capture_sample(
            "row",
            [1],
            [2],
            [-0.7],
            Tokenizer(),
            finish_reason="stop",
            raw_logprobs=raw,
            sampling_support_ids=support,
        )


def test_capture_keeps_probability_sidecar_and_weight_version():
    sample = capture_sample(
        "row",
        [1],
        [2, 5],
        [-0.7, -0.2],
        Tokenizer(),
        finish_reason="stop",
        raw_logprobs=[-0.9, -0.4],
        sampling_support_ids=[(2, 4), (7, 5)],
        weight_version="step-3:abc",
    )
    assert sample["weight_version"] == "step-3:abc"
    assert sample["logprob_semantics"] == "sampling_processed"
    assert sample["sampling_support_ids"] == [[2, 4], [7, 5]]
    assert sample["raw_logprobs"] == [-0.9, -0.4]


class ThinkingTokenizer:
    def decode(self, ids, **_kwargs):
        return (
            {
                7: "<think>Guard absent; caller drains balance.</think>",
                8: '{"findings":[]}',
                9: "<|im_end|>",
            }[ids[0]]
            if len(ids) == 1
            else ('<think>Guard absent; caller drains balance.</think>{"findings":[]}<|im_end|>')
        )


def test_sample_keeps_original_tokens_probabilities_and_reasoning_final_join():
    sample = capture_sample(
        "row", [1, 2], [7, 8, 9], [-0.1, -0.2, -0.3], ThinkingTokenizer(), finish_reason="stop"
    )
    assert sample["completion_ids"] == [7, 8, 9]
    assert sample["logprobs"] == [-0.1, -0.2, -0.3]
    assert sample["reasoning"] == "Guard absent; caller drains balance."
    assert sample["final_text"] == '{"findings":[]}'
    assert sample["raw_text"].endswith("<|im_end|>")
    with pytest.raises(ValueError, match="token"):
        capture_sample("row", [1], [7, 8], [-0.1], ThinkingTokenizer(), finish_reason="stop")
