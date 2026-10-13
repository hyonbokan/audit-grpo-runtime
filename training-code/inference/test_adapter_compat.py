"""Typed LoRA requests survive the actual untyped RPC serialization boundary."""

from types import SimpleNamespace

import msgspec
import pytest
from vllm.lora.request import LoRARequest

from adapter_compat import VerifiedAdapterWorker


def test_rpc_decoded_adapter_is_reconstructed_before_worker_loading():
    source = LoRARequest("step-2-new", 2, "/new/adapter", load_inplace=True)
    payload = msgspec.msgpack.decode(msgspec.msgpack.encode(source))
    assert isinstance(payload, list)  # Characterizes the failure seen in the real engine.

    class ModelRunnerBoundary:
        def add_lora(self, request):
            # The actual worker also requires typed attribute access here.
            return (
                request.lora_int_id == 2
                and request.lora_name == "step-2-new"
                and request.lora_path == "/new/adapter"
                and request.load_inplace
            )

    worker = SimpleNamespace(model_runner=ModelRunnerBoundary())
    assert VerifiedAdapterWorker.load_adapter(worker, payload) is True


def test_invalid_rpc_identity_is_rejected_before_loading():
    class ModelRunnerBoundary:
        def add_lora(self, request):
            pytest.fail("An invalid adapter reached model loading")

    worker = SimpleNamespace(model_runner=ModelRunnerBoundary())
    with pytest.raises((ValueError, msgspec.ValidationError)):
        VerifiedAdapterWorker.load_adapter(worker, ["invalid", 0, "/new/adapter"])
