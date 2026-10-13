"""Freeze a schedule's training requests from the dataset's original prompt tokens."""

from __future__ import annotations

import copy
import json
from typing import Any

from runtime_io import sha256_bytes


def extend_request(request: dict, parent_sha256: str, *, maximum_context: int) -> dict:
    """Keep the original prompt and give the completion the full remaining context."""
    result = copy.deepcopy(request)
    result.update(
        max_model_len=maximum_context,
        full_completion_ceiling=maximum_context - len(request["prompt_ids"]),
        parent_request_sha256=parent_sha256,
    )
    return result


def rerender_request(
    request: dict,
    messages: list[dict],
    tokenizer: Any,
    effort: str,
    parent_sha256: str,
    *,
    maximum_context: int,
) -> dict:
    """Re-tokenize the identical policy messages at medium effort."""
    if effort != "medium":
        raise ValueError("Only reviewed medium training retokenization is supported")
    if sha256_bytes(tokenizer.chat_template.encode()) != request["chat_template_sha256"]:
        raise ValueError("Pinned chat template differs from original request")
    options = dict(request["template_options"])
    original_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, **options
    )
    if list(original_ids) != request["prompt_ids"]:
        raise ValueError("Complete policy messages differ from original prompt tokens")
    options.update(enable_thinking=True, reasoning_effort=effort)
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options))
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    # This pinned template implements medium by omitting the xhigh/low preamble.
    if any(f"Reasoning effort is set to {value}." in decoded for value in ("xhigh", "low")):
        raise ValueError("Actual prompt retains a different effort instruction")
    result = extend_request(request, parent_sha256, maximum_context=maximum_context)
    result.update(
        prompt_ids=ids,
        prompt_ids_sha256=sha256_bytes(json.dumps(ids).encode()),
        template_options=options,
        full_completion_ceiling=result["max_model_len"] - len(ids),
        policy_messages_sha256=sha256_bytes(json.dumps(messages, ensure_ascii=False).encode()),
    )
    return result
