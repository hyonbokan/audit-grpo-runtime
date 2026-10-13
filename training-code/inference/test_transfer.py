"""Catch missing modules, stale buffers, and unnoticed adapter precision loss."""

import torch

import transfer


def verify(*args):
    return transfer.verify_transfer(*args, scaling=2.0)


def fixture():
    source = {
        "base_model.model.model.layers.0.x.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "base_model.model.model.layers.0.x.lora_B.weight": torch.tensor([[3.0]]),
    }
    payload = {
        "packed": {},
        "cpu": {
            "language_model.model.layers.0.x.lora_a": [torch.tensor([[1.0, 2.0]])],
            "language_model.model.layers.0.x.lora_b": [torch.tensor([[6.0]])],
        },
        "gpu": {
            "language_model.model.layers.0.x.lora_a_stacked": [torch.tensor([[[1.0, 2.0]]])],
            "language_model.model.layers.0.x.lora_b_stacked": [torch.tensor([[[6.0]]])],
        },
        "modules": ["language_model.model.layers.0.x"],
    }
    return source, payload


def test_all_source_tensors_must_reach_active_buffers():
    source, payload = fixture()
    report = verify(source, payload)
    assert report["exact_transfer_verified"]
    assert report["verified_tensor_count"] == 2
    del payload["gpu"]["language_model.model.layers.0.x.lora_b_stacked"]
    assert not verify(source, payload)["exact_transfer_verified"]


def test_precision_loss_cannot_pass_exact_transfer():
    source, payload = fixture()
    source["base_model.model.model.layers.0.x.lora_A.weight"][0, 0] = 1.001
    report = verify(source, payload)
    assert not report["exact_transfer_verified"]
    assert report["max_abs_source_difference"] > 0


def test_packed_slice_is_verified_by_source_name():
    source, payload = fixture()
    payload["packed"] = {
        "language_model.model.layers.0.fused": [
            "language_model.model.layers.0.x",
            "language_model.model.layers.0.y",
        ]
    }
    payload["cpu"] = {k.replace(".x.", ".fused."): [*v, None] for k, v in payload["cpu"].items()}
    payload["gpu"] = {
        k.replace(".x.", ".fused."): [*v, torch.zeros_like(v[0])] for k, v in payload["gpu"].items()
    }
    payload["modules"] = ["language_model.model.layers.0.fused"]
    assert verify(source, payload)["exact_transfer_verified"]


def test_declared_cast_is_reported_without_claiming_exact_transfer():
    source, payload = fixture()
    for storage in ("cpu", "gpu"):
        payload[storage] = {
            k: [v.to(torch.bfloat16) for v in values] for k, values in payload[storage].items()
        }
    source["base_model.model.model.layers.0.x.lora_A.weight"][0, 0] = 1.001
    report = verify(source, payload)
    assert report.get("cast_transfer_verified") is True
    assert report["exact_transfer_verified"] is False


def test_gdn_qkv_group_expands_to_three_gpu_slices_and_z_follows():
    source = {
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight": torch.tensor(
            [[1.0, 2.0]]
        ),
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight": torch.tensor(
            [[3.0], [4.0], [5.0]]
        ),
        "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_A.weight": torch.tensor(
            [[6.0, 7.0]]
        ),
        "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_B.weight": torch.tensor(
            [[8.0]]
        ),
    }
    name = "language_model.model.layers.0.linear_attn.in_proj_qkvz"
    payload = {
        "packed": {name: [name.replace("qkvz", "qkv"), name.replace("qkvz", "z")]},
        "modules": [name],
        "cpu": {
            name + ".lora_a": [torch.tensor([[1.0, 2.0]]), torch.tensor([[6.0, 7.0]])],
            name + ".lora_b": [torch.tensor([[6.0], [8.0], [10.0]]), torch.tensor([[16.0]])],
        },
        "gpu": {
            name + ".lora_a_stacked": [torch.tensor([[[1.0, 2.0]]])] * 3
            + [torch.tensor([[[6.0, 7.0]]])],
            name + ".lora_b_stacked": [torch.tensor([[[v]]]) for v in (6.0, 8.0, 10.0, 16.0)],
        },
    }
    assert verify(source, payload)["exact_transfer_verified"]
    payload["gpu"][name + ".lora_a_stacked"][1] = torch.tensor([[[9.0, 2.0]]])
    assert not verify(source, payload)["exact_transfer_verified"]
