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
