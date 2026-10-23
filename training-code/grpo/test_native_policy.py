"""Restore real optimizer moments and advance them under the reviewed schedule."""

import pytest
import torch

from grpo import native_policy


def test_optimizer_moments_continue_from_step_one_with_nonzero_learning_rate(tmp_path):
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW([{"params": list(model.parameters())}, {"params": []}], lr=5e-6)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.param_groups[0]["lr"] = 0.0
    optimizer.param_groups[1]["lr"] = 0.0
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    restored = native_policy.restore_optimizer(model, tmp_path, learning_rate=5e-6)
    assert [group["lr"] for group in restored.param_groups] == [5e-6, 5e-6]
    assert torch.equal(
        restored.state[model.weight]["exp_avg"], optimizer.state[model.weight]["exp_avg"]
    )
    model.weight.grad = torch.ones_like(model.weight)
    restored.step()
    assert int(restored.state[model.weight]["step"].item()) == 2


def test_optimizer_shape_mismatch_is_rejected_before_any_update(tmp_path):
    old = torch.nn.Linear(3, 1, bias=False)
    optimizer = torch.optim.AdamW(old.parameters())
    old(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    with pytest.raises(ValueError, match="shape"):
        native_policy.restore_optimizer(
            torch.nn.Linear(2, 1, bias=False), tmp_path, learning_rate=5e-6
        )


def test_constant_scheduler_restores_saved_progress_and_advances_once(tmp_path):
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    torch.save(scheduler.state_dict(), tmp_path / "scheduler.pt")
    restored = native_policy.restore_scheduler(optimizer, tmp_path, learning_rate=5e-6)
    assert restored.last_epoch == 2
    optimizer.step()
    restored.step()
    assert restored.last_epoch == 3
    assert restored.get_last_lr() == [5e-6]


def test_wrapped_eight_bit_moments_restore_through_original_optimizer(tmp_path):
    from bitsandbytes.optim import AdamW8bit

    model = torch.nn.Linear(128, 64, bias=False)
    optimizer = AdamW8bit([{"params": [model.weight]}, {"params": []}], lr=5e-6)
    optimizer.init_state(optimizer.param_groups[0], model.weight, 0, 0)
    state = optimizer.state[model.weight]
    state["step"] = 1
    state["state1"].fill_(191)
    state["state2"].fill_(23)
    state["absmax1"].fill_(0.5)
    state["absmax2"].fill_(0.25)
    optimizer.param_groups[0]["lr"] = optimizer.param_groups[1]["lr"] = 0.0
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    restored = native_policy.restore_optimizer(
        model, tmp_path, learning_rate=5e-6, optimizer_kind="adamw_8bit"
    )
    assert isinstance(restored, AdamW8bit)
    assert [group["lr"] for group in restored.param_groups] == [5e-6, 5e-6]
    assert restored.state[model.weight]["step"] == 1
    for key in ("state1", "state2", "qmap1", "qmap2", "absmax1", "absmax2"):
        assert torch.equal(restored.state[model.weight][key], state[key])
    assert restored.state[model.weight]["state1"].dtype == torch.uint8


def test_checkpoint_accepts_original_optimizer_integer_step(tmp_path):
    from bitsandbytes.optim import AdamW8bit
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import GPT2Config, GPT2LMHeadModel

    model = get_peft_model(
        GPT2LMHeadModel(GPT2Config(vocab_size=8, n_layer=1, n_head=1, n_embd=4)),
        LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"]),
    )
    before = {key: value.clone() for key, value in get_peft_model_state_dict(model).items()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW8bit(parameters, lr=5e-6)
    for index, parameter in enumerate(parameters):
        optimizer.init_state(optimizer.param_groups[0], parameter, 0, index)
        optimizer.state[parameter]["step"] = 2
        with torch.no_grad():
            parameter.add_(0.01)
    receipt = native_policy.save_checkpoint(
        model,
        optimizer,
        before,
        tmp_path / "checkpoint-2",
        {"initial_step": 1, "target_step": 2, "learning_rate": 5e-6},
    )
    assert receipt["optimizer_steps"] == [2]
    saved = torch.load(tmp_path / "checkpoint-2/optimizer.pt", weights_only=True)
    assert {state["step"] for state in saved["state"].values()} == {2}


def test_saved_checkpoint_contains_changed_adapter_and_resumable_step_two(tmp_path):
    import json

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import GPT2Config, GPT2LMHeadModel

    model = get_peft_model(
        GPT2LMHeadModel(GPT2Config(vocab_size=8, n_layer=1, n_head=1, n_embd=4)),
        LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"]),
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-6)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = {
        key: value.detach().clone() for key, value in get_peft_model_state_dict(model).items()
    }
    optimizer.step()
    destination = tmp_path / "checkpoint-2"
    receipt = native_policy.save_checkpoint(
        model,
        optimizer,
        before,
        destination,
        {"initial_step": 1, "target_step": 2, "learning_rate": 5e-6},
    )
    after = load_file(str(destination / "adapter_model.safetensors"))
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert receipt["optimizer_steps"] == [2]
    assert json.loads((destination / "trainer_state.json").read_text()) == {"global_step": 2}
    saved_schedule = json.loads((destination / "continuation_schedule.json").read_text())
    assert saved_schedule == receipt["continuation_schedule"]
    saved = torch.load(destination / "optimizer.pt", weights_only=True)
    assert {int(value["step"].item()) for value in saved["state"].values()} == {2}
    assert (destination / "rng_state.pth").is_file()
    assert (destination / "scheduler.pt").is_file()


def test_a_fresh_eight_bit_optimizer_restores_empty_and_takes_its_first_step(tmp_path):
    from bitsandbytes.optim import AdamW8bit

    model = torch.nn.Linear(128, 64, bias=False)
    optimizer = AdamW8bit([{"params": [model.weight]}], lr=5e-6)
    saved = optimizer.state_dict()
    assert saved["state"] == {}
    torch.save(saved, tmp_path / "optimizer.pt")
    (tmp_path / "trainer_state.json").write_text('{"global_step": 0}')
    restored = native_policy.restore_optimizer(
        model, tmp_path, learning_rate=2e-5, optimizer_kind="adamw_8bit"
    )
    assert restored.param_groups[0]["lr"] == 2e-5
    model.weight.grad = torch.ones_like(model.weight)
    restored.step()
    assert int(restored.state[model.weight]["step"]) == 1
    assert restored.state[model.weight]["state1"].dtype == torch.uint8
