"""A real CPU model/optimizer checks that correction reaches the actual update."""

import math

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from grpo import corrected_update
from grpo.corrected_loss import support_logprobs


def model_and_group():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    model.get_output_embeddings().weight.requires_grad_(False)
    group = [
        {
            "row_id": "scope",
            "prompt_ids": [1, 2],
            "completion_ids": [token, 6],
            "logprobs": [math.log(0.5)] * 2,
            "sampling_support_ids": [[token, 12], [6, 13]],
            "logprob_semantics": "sampling_processed",
            "weight_version": "step-1:abc",
            "reward": reward,
        }
        for token, reward in [(3, 1.0), (4, 0.0), (5, 1.0), (7, -1.0)]
    ]
    settings = {
        "learning_rate": 5e-6,
        "epsilon": 0.2,
        "correction_max": 3.0,
        "projection_chunk": 2,
        "max_grad_norm": 1.0,
        "max_clip_fraction": 0.05,
        "min_ess_fraction": 0.5,
    }
    return model, group, settings


def test_four_draw_correction_changes_model_weights_with_one_optimizer_update():
    model, group, settings = model_and_group()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"])
    before = [p.detach().clone() for p in parameters]
    draws_before = draw_logprobs(model, group)
    result = corrected_update.apply_update(
        model, optimizer, group, settings, expected_version="step-1:abc"
    )
    deltas = [b - a for a, b in zip(draws_before, draw_logprobs(model, group), strict=True)]
    assert result["advantages"] == [0.75, -0.25, 0.75, -1.25]
    # One step moves log-probability toward above-mean draws, weighted by advantage.
    assert sum(a * d for a, d in zip(result["advantages"], deltas, strict=True)) > 0
    assert result["optimizer_updates"] == 1
    assert result["gradient_norm"] > 0
    assert result["correction_tokens"] == 8
    assert result["final_hidden_transfers"] == []  # one device: nothing moved
    assert math.isfinite(result["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, parameters, strict=True))
    assert {int(state["step"].item()) for state in optimizer.state.values()} == {1}


def draw_logprobs(model, group):
    """Summed support log-probability of each draw's completion under the current weights."""
    head = model.get_output_embeddings().weight
    totals = []
    with torch.no_grad():
        for sample in group:
            ids = torch.tensor([sample["prompt_ids"] + sample["completion_ids"][:-1]])
            hidden = model.get_decoder()(input_ids=ids).last_hidden_state[0, 1:]
            values = support_logprobs(
                hidden, head, sample["completion_ids"], sample["sampling_support_ids"]
            )
            totals.append(values.sum().item())
    return totals


@pytest.mark.parametrize(
    "bad, guard",
    [
        ("stale", "stale"),
        ("missing_support", "stale"),
        ("no_contrast", "reward contrast"),
        ("large_mismatch", "Correction health"),
    ],
)
def test_invalid_group_cannot_change_weights_or_optimizer_state(bad, guard):
    model, group, settings = model_and_group()
    if bad == "stale":
        group[0]["weight_version"] = "step-0:old"
    elif bad == "missing_support":
        group[0]["sampling_support_ids"] = None
    elif bad == "no_contrast":
        for sample in group:
            sample["reward"] = 0.0
    else:
        group[0]["logprobs"] = [-100.0, -100.0]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-6)
    before = [p.detach().clone() for p in parameters]
    with pytest.raises(ValueError, match=guard):
        corrected_update.apply_update(
            model, optimizer, group, settings, expected_version="step-1:abc"
        )
    assert not optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(before, parameters, strict=True))
