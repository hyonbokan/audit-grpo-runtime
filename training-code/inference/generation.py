"""Native request validation and sampled-batch capture shared by rollout and evaluation."""

import hashlib
import json
import math


def validate_request(request):
    ids = request["prompt_ids"]
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != request["prompt_ids_sha256"]:
        raise ValueError("Native prompt hash differs")
    if request["model"]["precision"] != "bf16":
        raise ValueError("BF16 weights required")
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Nonempty native prompt tokens required")
    if len(ids) + request["full_completion_ceiling"] > request["max_model_len"]:
        raise ValueError("Context must accommodate the whole prompt and completion ceiling")


def capture_batch(
    output,
    prompt_ids,
    *,
    count,
    seconds,
    logprob_semantics: str | None = None,
    weight_version: str | None = None,
    require_sampling_support: bool = False,
):
    if require_sampling_support and logprob_semantics != "sampling_processed":
        raise ValueError("Sampling support replay requires processed probabilities")
    if logprob_semantics is not None:
        if logprob_semantics not in ("model_raw", "sampling_processed"):
            raise ValueError("Unknown probability semantics")
        if not isinstance(weight_version, str) or not weight_version:
            raise ValueError("Explicit policy weight version required")
    if list(output.prompt_token_ids) != prompt_ids:
        raise ValueError("Engine prompt differs from complete native prompt")
    if len(output.outputs) != count:
        raise ValueError("Engine completion count differs")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Positive measured duration required")
    draws = []
    for draw in output.outputs:
        ids = list(draw.token_ids)
        probabilities = draw.logprobs
        if probabilities is None or len(probabilities) != len(ids):
            raise ValueError("Missing token logprob records")
        try:
            values = [entry[token].logprob for token, entry in zip(ids, probabilities, strict=True)]
        except KeyError as exc:
            raise ValueError("Missing chosen-token logprob") from exc
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Nonfinite chosen-token logprob")
        draws.append(
            {
                "index": draw.index,
                "raw_text": draw.text,
                "completion_ids": ids,
                "logprobs": values,
                "finish_reason": draw.finish_reason,
                "stop_reason": draw.stop_reason,
            }
        )
        if logprob_semantics is not None:
            mask = getattr(draw, "sampling_mask", None)
            support = None if mask is None else [list(tokens) for tokens in mask.token_ids]
            if support is None:
                if require_sampling_support:
                    raise ValueError("Required sampling support was not returned by the engine")
            else:
                if len(support) != len(ids):
                    raise ValueError("Per-token sampling support lengths differ from completion")
                if any(
                    not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or len(set(tokens)) != len(tokens)
                    or chosen not in tokens
                    for chosen, tokens in zip(ids, support, strict=True)
                ):
                    raise ValueError("Invalid per-token sampling support")
            draws[-1].update(
                logprob_semantics=logprob_semantics,
                weight_version=weight_version,
                raw_logprobs=values if logprob_semantics == "model_raw" else None,
                sampling_support_ids=support,
            )
    tokens = sum(len(draw["completion_ids"]) for draw in draws)
    return {
        "draw_count": count,
        "prompt_tokens": len(prompt_ids),
        "draws": draws,
        "generation_seconds_including_prefill": seconds,
        "completion_tokens": tokens,
        "aggregate_tokens_per_second": tokens / seconds,
        "logprob_distribution_verified": False,
    }
