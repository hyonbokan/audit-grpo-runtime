"""Native activation hooks preserve gradients and clean up; frozen-layer offload is as asked."""

import copy
import sys
import types

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


class FakeSwapper:
    def __init__(self, layers, host_bytes):
        self.indices = list(range(layers))
        self.bytes = host_bytes

    def host_bytes(self):
        return self.bytes


def fake_block_swap(monkeypatch, swapper=None):
    """Stand in for Unsloth's block swap, which needs a GPU to import."""
    calls = []

    def install_block_swap(model, layers, **options):
        calls.append((layers, options))
        return swapper if swapper is not None else FakeSwapper(layers, host_bytes=1)

    utils = types.ModuleType("unsloth.models._utils")
    utils.install_block_swap = install_block_swap
    for name, module in (
        ("unsloth", types.ModuleType("unsloth")),
        ("unsloth.models", types.ModuleType("unsloth.models")),
        ("unsloth.models._utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_zero_offload_layers_keeps_every_frozen_layer_on_gpu(monkeypatch):
    calls = fake_block_swap(monkeypatch)
    assert memory_compat.configure_frozen_layer_offload(object(), layers=0) is None
    assert calls == []


def test_requested_offload_installs_that_many_swapped_layers(monkeypatch):
    calls = fake_block_swap(monkeypatch)
    swapper = memory_compat.configure_frozen_layer_offload(object(), layers=16, prefetch_depth=1)
    assert swapper.indices == list(range(16))
    assert calls == [(16, {"prefetch_depth": 1, "use_gradient_checkpointing": "unsloth"})]


@pytest.mark.parametrize("layers", [-1, 1.5, True])
def test_invalid_offload_layer_counts_are_rejected(monkeypatch, layers):
    calls = fake_block_swap(monkeypatch)
    with pytest.raises(ValueError, match="offload_layers"):
        memory_compat.configure_frozen_layer_offload(object(), layers=layers)
    assert calls == []


def test_a_partial_install_is_rejected(monkeypatch):
    fake_block_swap(monkeypatch, swapper=FakeSwapper(8, host_bytes=1))
    with pytest.raises(ValueError, match="not installed"):
        memory_compat.configure_frozen_layer_offload(object(), layers=16)
