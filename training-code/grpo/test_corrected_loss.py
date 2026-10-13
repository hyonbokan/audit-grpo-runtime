"""Independent probability/gradient checks for the selected corrected objective."""

import math

import pytest
import torch

from grpo import corrected_loss


def test_sparse_support_projection_matches_full_projection_and_gradients():
    hidden = torch.tensor([[1.0, 2.0], [2.0, -1.0]], requires_grad=True)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
    values = corrected_loss.support_logprobs(hidden, weight, [0, 2], [[0, 1], [2, 0]], chunk=1)
    assert values.tolist() == pytest.approx([-1.3132616875, -5.0067153485], abs=1e-6)
    values.sum().backward()
    expected = torch.tensor([[0.7310586, -0.7310586], [-1.9866143, 0.9933071]])
    assert torch.allclose(hidden.grad, expected, atol=1e-6)


def test_ragged_support_chunks_match_one_token_chunks():
    weight = torch.tensor([[1.0, 0.5], [-0.5, 1.0], [2.0, -1.0], [0.0, 3.0]])
    chosen, support = [1, 2, 3], [[1], [2, 0, 3], [3, 1]]
    results = []
    for chunk in (1, 2):
        hidden = torch.tensor([[1.0, 2.0], [2.0, -1.0], [0.5, 0.5]], requires_grad=True)
        values = corrected_loss.support_logprobs(hidden, weight, chosen, support, chunk=chunk)
        values.sum().backward()
        results.append((values.detach(), hidden.grad))
    (one, one_grad), (two, two_grad) = results
    assert one[0].item() == 0.0
    assert torch.allclose(one, two) and torch.allclose(one_grad, two_grad)


def test_single_update_correction_changes_real_gradient_and_is_detached():
    current = torch.tensor([math.log(0.4)], requires_grad=True)
    old = current.detach().clone().requires_grad_()
    sampled = torch.tensor([math.log(0.2)], requires_grad=True)
    loss, weights = corrected_loss.corrected_loss(
        current, old, sampled, advantage=2.0, total_tokens=4, epsilon=0.2, correction_max=3.0
    )
    assert loss.item() == pytest.approx(-1.0)
    assert weights.item() == pytest.approx(2.0)
    loss.backward()
    assert current.grad.item() == pytest.approx(-1.0)
    assert old.grad is None and sampled.grad is None


def test_correction_cap_and_ppo_clip_are_both_applied():
    current = torch.tensor([math.log(0.8)], requires_grad=True)
    old = torch.tensor([math.log(0.4)])
    sampled = torch.tensor([math.log(0.01)])
    loss, weights = corrected_loss.corrected_loss(
        current, old, sampled, advantage=1.0, total_tokens=1, epsilon=0.2, correction_max=3.0
    )
    assert loss.item() == pytest.approx(-3.6)
    assert weights.item() == 3.0
    loss.backward()
    assert current.grad.item() == 0.0


@pytest.mark.parametrize("support", [[[1]], [[0, 0]], [[0, -1]], [[0, 3]], []])
def test_invalid_support_cannot_produce_a_training_loss(support):
    with pytest.raises(ValueError, match="support"):
        corrected_loss.support_logprobs(torch.ones(1, 2), torch.ones(3, 2), [0], support)


def test_nonfinite_probability_aborts_before_backward():
    with pytest.raises(ValueError, match="finite"):
        corrected_loss.corrected_loss(
            torch.tensor([float("nan")]),
            torch.zeros(1),
            torch.zeros(1),
            advantage=1.0,
            total_tokens=1,
            epsilon=0.2,
            correction_max=3.0,
        )
