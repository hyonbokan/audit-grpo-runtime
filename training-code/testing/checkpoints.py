"""A tiny trained LoRA checkpoint: the template the step-0 checkpoint is shaped from."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

A, B = (
    "base_model.model.layers.0.q_proj.lora_A.weight",
    "base_model.model.layers.0.q_proj.lora_B.weight",
)
ADAPTER_CONFIG = {"r": 2, "lora_alpha": 4, "peft_type": "LORA"}


def trained_checkpoint(root: Path) -> Path:
    """Adapter, one-step optimizer and scheduler, at global step 14.

    Like a Trainer checkpoint, the optimizer has a second, empty parameter group.
    """
    root.mkdir(parents=True)
    tensors = {A: torch.full((2, 16), 0.5), B: torch.full((6, 2), 0.25)}
    save_file(tensors, root / "adapter_model.safetensors")
    (root / "adapter_config.json").write_text(json.dumps(ADAPTER_CONFIG))
    model = torch.nn.ParameterList(torch.nn.Parameter(t.clone()) for t in tensors.values())
    optimizer = torch.optim.AdamW(
        [{"params": list(model)}, {"params": [], "weight_decay": 0.0}], lr=5e-6
    )
    for parameter in model:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.save(optimizer.state_dict(), root / "optimizer.pt")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scheduler.step()
    torch.save(scheduler.state_dict(), root / "scheduler.pt")
    (root / "trainer_state.json").write_text('{"global_step": 14}')
    return root
