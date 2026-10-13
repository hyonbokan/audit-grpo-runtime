"""Lossless sampled-token artifacts shared across the GPU and judge environments."""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def capture_sample(
    row_id: str,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    logprobs: Sequence[float],
    tokenizer: Any,
    *,
    finish_reason: str,
    raw_logprobs: Sequence[float] | None = None,
    sampling_support_ids: Sequence[Sequence[int]] | None = None,
    weight_version: str | None = None,
) -> dict[str, Any]:
    if not completion_ids or len(completion_ids) != len(logprobs):
        raise ValueError("Completion token/logprob lengths differ")
    if any(type(token) is not int or token < 0 for token in [*prompt_ids, *completion_ids]):
        raise ValueError("Invalid token ID")
    if any(not math.isfinite(value) or value > 1e-6 for value in logprobs):
        raise ValueError("Invalid sampled token logprob")
    raw = tokenizer.decode(
        completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # The native template may put the opening think marker in the prompt.
    reasoning, separator, final = raw.partition("</think>")
    if separator:
        reasoning = reasoning.removeprefix("<think>").strip()
    else:
        reasoning, final = "", raw
    final = final.strip().removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()
    sample = {
        "row_id": row_id,
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(completion_ids),
        "logprobs": list(logprobs),
        "raw_text": raw,
        "reasoning": reasoning,
        "final_text": final,
        "finish_reason": finish_reason,
    }
    if raw_logprobs is not None or sampling_support_ids is not None:
        if raw_logprobs is None or sampling_support_ids is None:
            raise ValueError("Raw logprobs and sampling support must be supplied together")
        if len(raw_logprobs) != len(completion_ids) or len(sampling_support_ids) != len(
            completion_ids
        ):
            raise ValueError("Raw probability/support lengths differ from completion")
        if any(not math.isfinite(value) or value > 1e-6 for value in raw_logprobs):
            raise ValueError("Invalid raw model logprob")
        if any(raw > sampled + 1e-6 for raw, sampled in zip(raw_logprobs, logprobs, strict=True)):
            raise ValueError("raw model logprob exceeds sampling logprob")
        if weight_version is not None and (
            not isinstance(weight_version, str) or not weight_version
        ):
            raise ValueError("Invalid policy weight version")
        if any(
            token not in support
            for token, support in zip(
                completion_ids,
                sampling_support_ids,
                strict=True,
            )
        ):
            raise ValueError("Chosen token is outside sampling support")
        sample.update(
            raw_logprobs=list(raw_logprobs),
            sampling_support_ids=[list(support) for support in sampling_support_ids],
            logprob_semantics="sampling_processed",
            raw_logprob_semantics="model_raw",
            weight_version=weight_version,
        )
    return sample


def write_new(path: Path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
