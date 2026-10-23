"""The step-0 checkpoint: base-equivalent adapter, empty optimizer, resumable state files."""

import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

import hf_backup
import initial_checkpoint
from grpo import native_policy
from runtime_io import sha256_file

LEARNING_RATE = 2e-5
A, B = (
    "base_model.model.layers.0.q_proj.lora_A.weight",
    "base_model.model.layers.0.q_proj.lora_B.weight",
)
CONFIG = {"r": 2, "lora_alpha": 4, "peft_type": "LORA"}


@pytest.fixture
def template(tmp_path):
    root = tmp_path / "template"
    root.mkdir()
    tensors = {A: torch.full((2, 16), 0.5), B: torch.full((6, 2), 0.25)}
    save_file(tensors, root / "adapter_model.safetensors")
    (root / "adapter_config.json").write_text(json.dumps(CONFIG))
    model = torch.nn.ParameterList(torch.nn.Parameter(t.clone()) for t in tensors.values())
    optimizer = torch.optim.AdamW([{"params": list(model)}], lr=5e-6)
    for parameter in model:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.save(optimizer.state_dict(), root / "optimizer.pt")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scheduler.step()
    torch.save(scheduler.state_dict(), root / "scheduler.pt")
    (root / "trainer_state.json").write_text('{"global_step": 14}')
    return root


def create(template, tmp_path, seed=7, name="initial"):
    return initial_checkpoint.create(template, tmp_path / name, seed, LEARNING_RATE)


def test_the_adapter_equals_the_base_model_and_starts_lora_a_from_the_standard_init(
    template, tmp_path
):
    create(template, tmp_path)
    fresh = load_file(tmp_path / "initial/adapter_model.safetensors")
    assert set(fresh) == {A, B}
    assert torch.count_nonzero(fresh[B]) == 0
    bound = 1 / 16**0.5
    assert fresh[A].abs().max() <= bound and fresh[A].abs().max() > bound / 2
    assert fresh[A].dtype == fresh[B].dtype == torch.float32


def test_the_same_seed_gives_the_same_adapter_and_another_seed_a_different_one(template, tmp_path):
    first = create(template, tmp_path)
    again = create(template, tmp_path, name="again")
    other = create(template, tmp_path, seed=8, name="other")
    assert first["adapter_sha256"] == again["adapter_sha256"]
    assert first["adapter_sha256"] != other["adapter_sha256"]
    assert first["adapter_sha256"] == sha256_file(tmp_path / "initial/adapter_model.safetensors")


def test_the_optimizer_is_empty_with_the_templates_groups_and_the_scheduler_is_at_step_zero(
    template, tmp_path
):
    create(template, tmp_path)
    saved = torch.load(tmp_path / "initial/optimizer.pt", weights_only=True)
    original = torch.load(template / "optimizer.pt", weights_only=True)
    assert saved["state"] == {} and saved["param_groups"] == original["param_groups"]
    scheduler = torch.load(tmp_path / "initial/scheduler.pt", weights_only=True)
    assert scheduler["last_epoch"] == 0 and scheduler["_step_count"] == 1
    assert scheduler["base_lrs"] == scheduler["_last_lr"] == [LEARNING_RATE]
    assert json.loads((tmp_path / "initial/trainer_state.json").read_text()) == {"global_step": 0}
    assert (tmp_path / "initial/adapter_config.json").read_bytes() == (
        template / "adapter_config.json"
    ).read_bytes()


def test_the_checkpoint_restores_and_takes_a_first_update_at_step_one(template, tmp_path):
    create(template, tmp_path)
    checkpoint = tmp_path / "initial"
    model = torch.nn.ParameterList(
        torch.nn.Parameter(t) for t in load_file(checkpoint / "adapter_model.safetensors").values()
    )
    holder = torch.nn.Module()
    holder.parameters = lambda: iter(model)
    optimizer = native_policy.restore_optimizer(holder, checkpoint, learning_rate=LEARNING_RATE)
    scheduler = native_policy.restore_scheduler(optimizer, checkpoint, learning_rate=LEARNING_RATE)
    for parameter in model:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert {int(state["step"]) for state in optimizer.state.values()} == {1}
    assert scheduler.last_epoch == 1


def test_a_backup_accepts_the_empty_optimizer_only_at_step_zero(template, tmp_path):
    create(template, tmp_path)
    checkpoint = tmp_path / "initial"
    assert hf_backup.verify_resumable(checkpoint, 0, LEARNING_RATE)["optimizer_step"] == 0
    with pytest.raises(ValueError, match="step differs"):
        hf_backup.verify_resumable(checkpoint, 1, LEARNING_RATE)
    shutil.copy(template / "optimizer.pt", checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="counters"):
        hf_backup.verify_resumable(checkpoint, 0, LEARNING_RATE)


def test_an_empty_optimizer_is_refused_after_step_zero(template, tmp_path):
    create(template, tmp_path)
    checkpoint = tmp_path / "initial"
    (checkpoint / "trainer_state.json").write_text('{"global_step": 3}')
    model = torch.nn.ParameterList(
        torch.nn.Parameter(t) for t in load_file(checkpoint / "adapter_model.safetensors").values()
    )
    holder = torch.nn.Module()
    holder.parameters = lambda: iter(model)
    with pytest.raises(ValueError, match="fresh optimizer"):
        native_policy.restore_optimizer(holder, checkpoint, learning_rate=LEARNING_RATE)


def test_an_unknown_tensor_kind_or_a_mismatched_optimizer_is_refused(template, tmp_path):
    save_file(
        {A: torch.zeros(2, 16), "other.weight": torch.zeros(1)},
        template / "adapter_model.safetensors",
    )
    with pytest.raises(ValueError, match="lora_A nor lora_B"):
        create(template, tmp_path)
    save_file({A: torch.zeros(2, 16)}, template / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="optimizer"):
        create(template, tmp_path, name="second")


def test_an_existing_output_is_refused(template, tmp_path):
    (tmp_path / "initial").mkdir()
    with pytest.raises(FileExistsError):
        create(template, tmp_path)
