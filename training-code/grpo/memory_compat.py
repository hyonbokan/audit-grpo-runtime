"""Native memory compatibility, preserving the validated replay offload behavior.

GPU SDK imports stay lazy so CPU analysis can import the maintained trainer.
Private checkpoint and Unsloth hooks are confined to this module.
"""

from contextlib import contextmanager, nullcontext
from typing import Any


class ActivationOffload:
    """Move saved CUDA activations to CPU; retain parameter storage on CUDA."""

    def __init__(self, model: Any, *, minimum_bytes: int = 1024 * 1024) -> None:
        self.protected = {
            parameter.untyped_storage().data_ptr() for parameter in model.parameters()
        }
        self.minimum_bytes = minimum_bytes
        self.events = []
        self.checkpoint_calls = 0
        self.checkpoint_input_samples = []

    @property
    def offloaded_bytes(self):
        return sum(event["bytes"] for event in self.events if event["offloaded"])

    @property
    def parameter_bytes_offloaded(self):
        return sum(
            event["bytes"] for event in self.events if event["parameter"] and event["offloaded"]
        )

    def pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        parameter = tensor.untyped_storage().data_ptr() in self.protected
        offload = tensor.device.type == "cuda" and size >= self.minimum_bytes and not parameter
        self.events.append({"bytes": size, "parameter": parameter, "offloaded": offload})
        saved = tensor.detach().to("cpu").contiguous() if offload else tensor.detach()
        layout = (tensor.shape, tensor.stride(), tensor.storage_offset()) if offload else None
        return tensor.device, saved, layout

    def unpack(self, saved):
        import torch

        device, tensor, layout = saved
        if layout is None:
            return tensor.to(device)
        shape, strides, offset = layout
        extent = (
            offset
            + 1
            + sum((size - 1) * stride for size, stride in zip(shape, strides, strict=True))
        )
        storage = torch.empty(extent, device=device, dtype=tensor.dtype)
        restored = storage.as_strided(shape, strides, offset)
        destination, source = restored, tensor
        for axis, (size, stride) in enumerate(zip(shape, strides, strict=True)):
            if size > 1 and stride == 0:
                destination = destination.narrow(axis, 0, 1)
                source = source.narrow(axis, 0, 1)
        destination.copy_(source)
        return restored

    def context(self):
        import torch

        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    @contextmanager
    def model_forward_context(self, model, *, extra_modules=()):
        contexts = []

        def begin(_model, _inputs):
            context = self.context()
            context.__enter__()
            contexts.append(context)

        def end(_model, _inputs, _output):
            if contexts:
                contexts.pop().__exit__(None, None, None)

        modules = {id(module): module for module in (model, *extra_modules)}
        handles = []
        for module in modules.values():
            handles.append(module.register_forward_pre_hook(begin))
            handles.append(module.register_forward_hook(end, always_call=True))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            while contexts:
                contexts.pop().__exit__(None, None, None)


def install_cpu_checkpoints(layers: list[Any], memory: ActivationOffload) -> int:
    import torch
    import torch.utils.checkpoint as checkpoint_module
    from unsloth_zoo.gradient_checkpointing import _gradient_checkpoint_recompute_marker

    checkpoint_class = getattr(checkpoint_module, "_old_CheckpointFunction", None)
    if checkpoint_class is None:
        checkpoint_class = checkpoint_module.CheckpointFunction
    if checkpoint_class.__name__ != "CheckpointFunction":
        raise ValueError("Original Torch checkpoint class unavailable")

    def checkpoint(function, *inputs, **kwargs):
        preserve = kwargs.pop("preserve_rng_state", True)
        kwargs.pop("use_reentrant", None)
        if kwargs:
            raise ValueError("Unexpected checkpoint call keywords")
        memory.checkpoint_calls += 1
        if len(memory.checkpoint_input_samples) < 2:
            memory.checkpoint_input_samples.append(
                {
                    "grad_enabled": torch.is_grad_enabled(),
                    "function_type": type(function).__name__,
                    "inputs": [
                        {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "device": str(value.device),
                            "requires_grad": value.requires_grad,
                            "bytes": value.numel() * value.element_size(),
                        }
                        if isinstance(value, torch.Tensor)
                        else {"type": type(value).__name__}
                        for value in inputs
                    ],
                }
            )

        def run(*values):
            marker = (
                _gradient_checkpoint_recompute_marker()
                if torch.is_grad_enabled()
                else nullcontext()
            )
            with marker:
                return function(*values)

        with memory.context():
            return checkpoint_class.apply(run, preserve, *inputs)

    for layer in layers:
        layer._gradient_checkpointing_func = checkpoint
    return len(layers)


def restore_existing_embedding_offload(model: Any) -> int:
    from unsloth.models.vision import offload_input_embedding

    embeddings = model.get_input_embeddings()
    if not getattr(embeddings, "_unsloth_offload_hooks_installed", False):
        raise ValueError("An existing embedding offload configuration is required")
    if embeddings.weight.requires_grad:
        raise ValueError("Only the already frozen embedding may be offloaded")
    output = model.get_output_embeddings()
    if output.weight is embeddings.weight:
        raise ValueError("Tied output weights must stay on the decoder device")
    return offload_input_embedding(model, embeddings=[embeddings])


def configure_frozen_layer_offload(model: Any, *, layers: int = 16, prefetch_depth: int = 2) -> Any:
    from unsloth.models._utils import install_block_swap

    swapper = install_block_swap(
        model,
        layers,
        prefetch_depth=prefetch_depth,
        use_gradient_checkpointing="unsloth",
    )
    if swapper is None or len(swapper.indices) != layers or swapper.host_bytes() <= 0:
        raise ValueError("Requested frozen-layer offload was not installed")
    return swapper
