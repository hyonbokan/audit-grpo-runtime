import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from merge_checkpoint import merge_checkpoint


def make_fixture(root: Path):
    base, adapter = root / "base", root / "adapter"
    base.mkdir()
    adapter.mkdir()
    key = "model.language_model.layers.0.self_attn.q_proj.weight"
    save_file(
        {key: torch.eye(2, dtype=torch.bfloat16), "untouched": torch.ones(1)},
        str(base / "model.safetensors"),
    )
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    key: "model.safetensors",
                    "untouched": "model.safetensors",
                },
            }
        )
    )
    (base / "config.json").write_text("{}")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 1,
                "lora_alpha": 3,
                "bias": "none",
                "fan_in_fan_out": False,
                "use_rslora": False,
                "use_dora": False,
                "rank_pattern": {},
                "alpha_pattern": {},
                "modules_to_save": None,
            }
        )
    )
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    save_file(
        {
            prefix + ".lora_A.weight": torch.tensor([[1.0, 2.0]]),
            prefix + ".lora_B.weight": torch.tensor([[1.0], [2.0]]),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    return base, adapter, key


def test_merge_uses_config_scaling_preserves_sources_and_non_adapter_weights(tmp_path):
    base, adapter, key = make_fixture(tmp_path)
    before = {str(p): p.read_bytes() for r in (base, adapter) for p in r.iterdir()}
    output = tmp_path / "merged"
    receipt = merge_checkpoint(base, adapter, output, "pinned-test-revision")
    tensors = load_file(str(output / "model.safetensors"))
    assert tensors[key].tolist() == [[4.0, 6.0], [6.0, 13.0]]
    assert tensors[key].dtype == torch.bfloat16
    assert tensors["untouched"].tolist() == [1.0]
    assert receipt["merged_modules"] == 1
    assert all(Path(p).read_bytes() == value for p, value in before.items())
    assert not (output / "adapter_config.json").exists()
    assert (output / "config.json").read_bytes() == (base / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        merge_checkpoint(base, adapter, output, "pinned-test-revision")


def test_unsupported_adapter_features_fail_before_creating_artifact(tmp_path):
    base, adapter, _ = make_fixture(tmp_path)
    path = adapter / "adapter_config.json"
    config = json.loads(path.read_text())
    config["use_dora"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsupported"):
        merge_checkpoint(base, adapter, tmp_path / "merged", "pinned-test-revision")
    assert not (tmp_path / "merged").exists()


def test_missing_base_target_fails_before_creating_artifact(tmp_path):
    base, adapter, _ = make_fixture(tmp_path)
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"untouched": "model.safetensors"}})
    )
    with pytest.raises(ValueError, match="target"):
        merge_checkpoint(base, adapter, tmp_path / "merged", "pinned-test-revision")
    assert not (tmp_path / "merged").exists()
