"""Frozen training requests keep the original prompt identity and use the full context."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

import frozen_requests

PINNED_TOKENIZER = Path(
    "/workspace/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)


def test_extended_request_preserves_exact_prompt_and_records_old_identity():
    old = {
        "row_id": "tiny",
        "prompt_ids": [10, 20],
        "full_completion_ceiling": 32768,
        "max_model_len": 36864,
        "template_options": {"reasoning_effort": "xhigh"},
    }
    before = copy.deepcopy(old)
    new = frozen_requests.extend_request(old, "a" * 64, maximum_context=262144)
    assert old == before
    assert new["prompt_ids"] == [10, 20]
    assert new["max_model_len"] == 262144
    assert new["full_completion_ceiling"] == 262142
    assert new["parent_request_sha256"] == "a" * 64


def test_medium_request_retokens_real_template_and_preserves_policy_messages():
    from transformers import AutoTokenizer

    if not PINNED_TOKENIZER.is_dir():
        pytest.skip(f"Pinned tokenizer not on this host: {PINNED_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(PINNED_TOKENIZER, local_files_only=True)
    messages = [{"role": "user", "content": "Audit the complete contract: contract A {}"}]
    original_messages = copy.deepcopy(messages)
    options = {"enable_thinking": True, "reasoning_effort": "xhigh", "add_generation_prompt": True}
    old_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)
    original = {
        "row_id": "tiny/train",
        "model": {"id": "Qwen/Qwen3.8-27B"},
        "prompt_ids": old_ids,
        "template_options": options,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }
    before = copy.deepcopy(original)
    new = frozen_requests.rerender_request(
        original, messages, tokenizer, "medium", "a" * 64, maximum_context=262144
    )
    decoded = tokenizer.decode(new["prompt_ids"], skip_special_tokens=False)
    assert "Reasoning effort is set to xhigh." not in decoded
    assert "Reasoning effort is set to low." not in decoded
    assert decoded.startswith("<|im_start|>user\nAudit the complete contract: contract A {}")
    assert decoded.endswith("<|im_start|>assistant\n<think>\n")
    assert new["template_options"]["reasoning_effort"] == "medium"
    assert messages == original_messages and original == before
    assert new["prompt_ids"] != old_ids
    assert (
        new["prompt_ids_sha256"]
        == hashlib.sha256(json.dumps(new["prompt_ids"]).encode()).hexdigest()
    )
    assert new["full_completion_ceiling"] == 262144 - len(new["prompt_ids"])
    assert new["parent_request_sha256"] == "a" * 64
    assert new["model"] == original["model"]
    with pytest.raises(ValueError):
        frozen_requests.rerender_request(
            original, messages, tokenizer, "low", "a" * 64, maximum_context=262144
        )
