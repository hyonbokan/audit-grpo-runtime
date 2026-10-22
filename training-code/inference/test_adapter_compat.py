"""Typed LoRA requests survive the actual untyped RPC serialization boundary."""

import re
from types import SimpleNamespace

import msgspec
import pytest
import torch
from vllm.lora.request import LoRARequest

import adapter_compat
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


# --- Tensor-parallel layout


def layer(name, **attributes):
    """A LoRA layer stand-in; the layout rule is chosen by vLLM's class name."""
    return type(name, (), attributes)()


@pytest.mark.parametrize(
    "name,dims",
    [
        ("MergedQKVParallelLinearWithLoRA", {"lora_b_stacked": -2}),
        ("MergedColumnParallelLinearWithLoRA", {"lora_b_stacked": -2}),
        ("MergedColumnParallelLinearVariableSliceWithLoRA", {"lora_b_stacked": -2}),
        ("RowParallelLinearWithLoRA", {"lora_a_stacked": -1}),
        ("ReplicatedLinearWithLoRA", {}),
    ],
)
def test_each_reviewed_layer_names_the_dimension_vllm_splits(name, dims):
    base = SimpleNamespace(num_kv_head_replicas=1)
    assert adapter_compat.sharded_dims(layer(name, base_layer=base)) == dims


def test_plain_column_layer_splits_lora_b_unless_it_packs_gate_and_up():
    assert adapter_compat.sharded_dims(
        layer("ColumnParallelLinearWithLoRA", is_merged_col_linear=False)
    ) == {"lora_b_stacked": -2}
    with pytest.raises(ValueError, match="interleaves"):
        adapter_compat.sharded_dims(
            layer("ColumnParallelLinearWithLoRA", is_merged_col_linear=True)
        )


@pytest.mark.parametrize(
    "module",
    [
        layer("QKVParallelLinearWithLoRA"),
        layer("ColumnParallelLinearWithShardedLoRA"),
        layer(
            "MergedQKVParallelLinearWithLoRA",
            base_layer=SimpleNamespace(num_kv_head_replicas=2),
        ),
    ],
)
def test_unreviewed_or_overlapping_layouts_are_rejected(module):
    with pytest.raises(ValueError):
        adapter_compat.sharded_dims(module)


def test_shards_report_the_rank_every_lora_layer_agrees_on(tmp_path):
    manager = SimpleNamespace(
        lora_config=SimpleNamespace(fully_sharded_loras=False),
        get_adapter=lambda _id: SimpleNamespace(id=7, loras={"a": None, "b": None}),
        lora_index_to_id=[7],
        packed_modules={},
        modules={
            "a": layer("ReplicatedLinearWithLoRA", tp_rank=1),
            "b": layer("RowParallelLinearWithLoRA", tp_rank=1),
        },
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=manager))
    )
    shard = adapter_compat.load_shard(VerifiedAdapterWorker.adapter_shards(worker, 7, tmp_path))
    assert (shard["rank"], shard["slot"], shard["cpu"]) == (1, 0, {})
    assert shard["sharded_dims"] == {"b.lora_a_stacked": -1}
    manager.modules["a"] = layer("ReplicatedLinearWithLoRA", tp_rank=0)
    with pytest.raises(ValueError, match="disagree"):
        VerifiedAdapterWorker.adapter_shards(worker, 7, tmp_path)


def shard_worker(modules, targeted):
    stacked = {"lora_a_stacked": [torch.ones(2, 3)], "lora_b_stacked": [torch.ones(2, 3)]}
    manager = SimpleNamespace(
        lora_config=SimpleNamespace(fully_sharded_loras=False),
        get_adapter=lambda _id: SimpleNamespace(
            id=7, loras={name: SimpleNamespace(lora_a=None, lora_b=None) for name in targeted}
        ),
        lora_index_to_id=[7],
        packed_modules={},
        modules={name: layer(kind, tp_rank=0, **stacked) for name, kind in modules.items()},
    )
    return SimpleNamespace(
        model_runner=SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=manager))
    )


def test_layers_the_adapter_does_not_target_are_left_out_of_the_shards(tmp_path):
    # vLLM wraps the embedding for LoRA even when the adapter has no embedding weights.
    worker = shard_worker(
        {"embed": "VocabParallelEmbeddingWithLoRA", "o_proj": "RowParallelLinearWithLoRA"},
        targeted=["o_proj"],
    )
    shard = adapter_compat.load_shard(VerifiedAdapterWorker.adapter_shards(worker, 7, tmp_path))
    assert shard["modules"] == ["o_proj"]
    assert set(shard["gpu"]) == {"o_proj.lora_a_stacked", "o_proj.lora_b_stacked"}
    assert shard["sharded_dims"] == {"o_proj.lora_a_stacked": -1}


def test_a_targeted_layer_without_a_reviewed_layout_is_still_rejected(tmp_path):
    worker = shard_worker({"embed": "VocabParallelEmbeddingWithLoRA"}, targeted=["embed"])
    with pytest.raises(ValueError, match="Unreviewed tensor-parallel LoRA layout"):
        VerifiedAdapterWorker.adapter_shards(worker, 7, tmp_path)


def test_a_shard_crosses_the_rpc_as_plain_data_and_its_buffers_round_trip(tmp_path):
    # vLLM's collective RPC turns tensors into nested lists; buffers travel by file instead.
    shard = {
        "rank": 0,
        "adapter_id": 7,
        "slot": 0,
        "packed": {"qkv": ["q", "k"]},
        "modules": ["o"],
        "cpu": {"o.lora_a": [torch.ones(2, 3), None]},
        "gpu": {"o.lora_a_stacked": [torch.arange(6.0).reshape(1, 2, 3)]},
        "sharded_dims": {"o.lora_a_stacked": -1},
    }
    reply = adapter_compat.save_shard(shard, tmp_path)
    assert msgspec.json.decode(msgspec.json.encode(reply)) == reply
    loaded = adapter_compat.load_shard(reply)
    assert loaded["cpu"]["o.lora_a"][1] is None
    assert torch.equal(loaded["cpu"]["o.lora_a"][0], torch.ones(2, 3))
    assert torch.equal(loaded["gpu"]["o.lora_a_stacked"][0], torch.arange(6.0).reshape(1, 2, 3))
    assert {key: loaded[key] for key in ("rank", "packed", "sharded_dims")} == {
        "rank": 0,
        "packed": {"qkv": ["q", "k"]},
        "sharded_dims": {"o.lora_a_stacked": -1},
    }


@pytest.mark.parametrize("installed", ["0.31.1", "0.31.0+cpu", "0.31.0rc1"])
def test_other_vllm_builds_are_rejected_before_reading_buffers(monkeypatch, installed):
    monkeypatch.setattr(adapter_compat.importlib.metadata, "version", lambda _name: installed)
    with pytest.raises(ValueError, match=rf"installed {re.escape(installed)}, allowed"):
        adapter_compat.require_pinned_vllm()
