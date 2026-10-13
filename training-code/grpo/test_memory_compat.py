"""Native activation hooks preserve gradients and clean up on failures."""

import copy

import pytest
import torch

from grpo import memory_compat


def test_forward_hooks_preserve_gradients_and_remove_their_scope():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
    control = copy.deepcopy(model)
    values = torch.randn(3, 4)
    control(values).square().sum().backward()
    memory = memory_compat.ActivationOffload(model, minimum_bytes=1)
    with memory.model_forward_context(model):
        model(values).square().sum().backward()
    for actual, expected in zip(model.parameters(), control.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert memory.events
    assert memory.parameter_bytes_offloaded == 0
    assert not model._forward_pre_hooks
    assert not model._forward_hooks


def test_forward_exception_removes_hooks_and_closes_saved_tensor_context():
    model = torch.nn.Linear(4, 1)
    memory = memory_compat.ActivationOffload(model)
    with pytest.raises(RuntimeError), memory.model_forward_context(model):
        model(torch.ones(1, 3))
    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    events = len(memory.events)
    model(torch.ones(1, 4)).sum().backward()
    assert len(memory.events) == events
