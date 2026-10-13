import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from adapter_mapping import prepare_adapter


def test_real_vllm_mapping_and_exact_tensors(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        "base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight": torch.randn(16, 32),
        "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight": torch.randn(32, 16),
    }
    save_file(tensors, str(source / "adapter_model.safetensors"))
    (source / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    destination = tmp_path / "mapped"
    receipt = prepare_adapter(source, destination)
    mapped = load_file(str(destination / "adapter_model.safetensors"))
    assert receipt["tensor_count"] == len(tensors)
    for key, value in tensors.items():
        new = key.replace(
            "base_model.model.model.layers.", "base_model.model.model.language_model.layers.", 1
        )
        assert torch.equal(value, mapped[new])
        name, _ = parse_fine_tuned_lora_name(
            new,
            Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper,
        )
        assert name.startswith("language_model.model.layers.")
    assert load_file(str(source / "adapter_model.safetensors")).keys() == tensors.keys()
    assert (source / "adapter_config.json").read_bytes() == (
        destination / "adapter_config.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_adapter(source, destination)


def test_reject_unreviewed_prefix(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {"unknown.layers.0.lora_A.weight": torch.ones(2)}, str(source / "adapter_model.safetensors")
    )
    (source / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="prefix"):
        prepare_adapter(source, tmp_path / "mapped")
