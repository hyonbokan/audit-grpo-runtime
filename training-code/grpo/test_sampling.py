"""Catch raw/sampling confusion, changed native masks, and retained score history."""

import math

import pytest
import torch

from grpo import sampling


def test_filter_keeps_native_top_k_boundary_ties():
    scores = torch.tensor([[3.0, 2.0, 2.0, 0.0]])
    filtered = sampling.filter_logits(
        scores,
        torch.tensor([2]),
        torch.tensor([1.0]),
    )
    assert filtered[0].tolist() == [3.0, 2.0, 2.0, -math.inf]
    assert scores[0].tolist() == [3.0, 2.0, 2.0, 0.0]


def test_filter_excludes_low_probability_tokens_after_top_k():
    scores = torch.tensor([[math.log(0.6), math.log(0.3), math.log(0.1)]])
    filtered = sampling.filter_logits(scores, None, torch.tensor([0.8]))
    assert torch.isfinite(filtered).tolist() == [[True, True, False]]


def test_recorder_preserves_raw_and_sampling_logprobs_for_chosen_tokens():
    recorder = sampling.SamplingRecorder(top_k=2, top_p=1.0)
    scores = torch.tensor([[math.log(0.6), math.log(0.3), math.log(0.1)]])
    recorder(torch.tensor([[0]]), scores)
    recorder.finish(1)
    assert recorder.logprobs == pytest.approx([math.log(1 / 3)], abs=1e-6)
    assert recorder.raw_logprobs == pytest.approx([math.log(0.3)], abs=1e-6)
    assert recorder.sampling_support_ids == [[0, 1]]
    assert recorder.previous is None
    assert recorder.previous_raw is None


def test_recorder_advances_once_per_token_without_vocabulary_history():
    recorder = sampling.SamplingRecorder(top_k=2, top_p=1.0)
    scores = torch.tensor([[3.0, 2.0, 0.0]])
    recorder(torch.tensor([[0]]), scores)
    recorder(torch.tensor([[0, 1]]), scores)
    recorder.finish(0)
    assert len(recorder.logprobs) == len(recorder.raw_logprobs) == 2
    assert recorder.sampling_support_ids == [[0, 1], [0, 1]]


def test_recorder_rejects_a_token_outside_the_sampled_support():
    recorder = sampling.SamplingRecorder(top_k=1, top_p=0.95)
    recorder(torch.tensor([[0]]), torch.tensor([[3.0, 2.0, 0.0]]))
    with pytest.raises(ValueError, match="support"):
        recorder.finish(2)
