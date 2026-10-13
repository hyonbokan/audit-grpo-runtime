"""Completed policy draws survive a later native-generation failure."""

import json

import pytest


@pytest.fixture
def sampler(tmp_path, monkeypatch):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from grpo import trainer

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = None

        def apply_chat_template(self, _messages, **_options):
            return [3, 4, 5]

        def decode(self, ids, **_options):
            return " ".join(map(str, ids))

    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=64,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
        )
    )
    monkeypatch.setattr(trainer, "generation_device", lambda _model: torch.device("cpu"))
    plan = trainer.default_plan()
    plan["training"]["max_completion_tokens"] = 3
    return trainer.NativeSampler(model, Tokenizer(), plan, timing_path=tmp_path / "events.jsonl")


def test_completed_draws_are_saved_with_lossless_token_artifacts(sampler, tmp_path):
    samples = sampler("scope", [{"role": "user", "content": "contract Toy {}"}], 4)
    saved = sorted((tmp_path / "draws/group-0001").glob("draw-*.json"))
    assert len(saved) == 4
    assert [json.loads(path.read_text()) for path in saved] == samples
    assert all(sample["prompt_ids"] == [3, 4, 5] for sample in samples)
    assert all(len(sample["completion_ids"]) == len(sample["logprobs"]) == 3 for sample in samples)


def test_first_draw_survives_failure_during_the_second_draw(sampler, tmp_path, monkeypatch):
    generate = sampler.model.generate
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("native generation failed")
        return generate(*args, **kwargs)

    monkeypatch.setattr(sampler.model, "generate", fail_after_first)
    with pytest.raises(RuntimeError, match="native generation failed"):
        sampler("scope", [{"role": "user", "content": "contract Toy {}"}], 4)
    saved = list((tmp_path / "draws/group-0001").glob("draw-*.json"))
    assert len(saved) == 1
    sample = json.loads(saved[0].read_text())
    assert sample["row_id"] == "scope"
    assert sample["prompt_ids"] == [3, 4, 5]
    assert len(sample["completion_ids"]) == len(sample["logprobs"]) == 3
    assert sampler.model.training is True


def test_new_draws_label_raw_and_sampling_probabilities(sampler):
    samples = sampler("scope", [{"role": "user", "content": "contract Toy {}"}], 4)
    for sample in samples:
        assert sample.get("logprob_semantics") == "sampling_processed"
        assert sample.get("raw_logprob_semantics") == "model_raw"
        assert len(sample.get("raw_logprobs", [])) == len(sample["completion_ids"])
        assert len(sample.get("sampling_support_ids", [])) == len(sample["completion_ids"])
        assert all(
            token in support
            for token, support in zip(
                sample["completion_ids"], sample["sampling_support_ids"], strict=True
            )
        )


def test_native_draws_bind_the_explicit_policy_version(sampler):
    from grpo.trainer import NativeSampler

    versioned = NativeSampler(
        sampler.model, sampler.tokenizer, sampler.plan, weight_version="native-step-1"
    )
    samples = versioned("scope", [{"role": "user", "content": "contract Toy {}"}], 4)
    assert all(sample["weight_version"] == "native-step-1" for sample in samples)
