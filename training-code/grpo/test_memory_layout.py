import pytest
import torch

from .memory_compat import ActivationOffload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA saved layouts")
@pytest.mark.parametrize("expanded", [False, True])
def test_offload_restores_strided_and_expanded_tensor_layouts(expanded):
    source = torch.randn(128, 512, device="cuda")[:, 1:257].transpose(0, 1)
    if expanded:
        source = source[:1].expand(256, 128)
    memory = ActivationOffload(torch.nn.Linear(1, 1).cuda(), minimum_bytes=1)
    restored = memory.unpack(memory.pack(source))
    assert restored.stride() == source.stride()
    assert restored.storage_offset() == source.storage_offset()
    torch.testing.assert_close(restored, source, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual Inductor CUDA backward")
def test_compiled_backward_with_strided_saved_activations_matches_control():
    torch.manual_seed(17)

    def function(values):
        view = values[:, :, :256].reshape(1, 128, 2, 128)
        return torch.nn.functional.silu(view).square().mean()

    values = torch.randn(1, 128, 512, device="cuda", requires_grad=True)
    control = values.detach().clone().requires_grad_()
    compiled = torch.compile(function, fullgraph=True)
    compiled(control).backward()
    memory = ActivationOffload(torch.nn.Linear(1, 1).cuda(), minimum_bytes=1)
    with memory.context():
        compiled(values).backward()
    torch.testing.assert_close(values.grad, control.grad, rtol=0, atol=0)
