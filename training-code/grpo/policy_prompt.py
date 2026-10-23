"""Policy prompt tokenization and the chat-template options a rollout uses."""

from __future__ import annotations

from typing import Any

TEMPLATE_OPTIONS = {
    "add_generation_prompt": True,
    "enable_thinking": True,
    "reasoning_effort": "xhigh",
    "preserve_thinking": True,
}


def prompt_token_ids(tokenizer: Any, messages: list[dict], *, template_options=None) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        **(TEMPLATE_OPTIONS if template_options is None else template_options),
    )


def policy_template_options(plan: dict) -> dict[str, Any]:
    options = dict(TEMPLATE_OPTIONS)
    profile = plan.get("sampling_profile")
    if profile is not None:
        effort = profile.get("policy_reasoning_effort")
        if not profile.get("id") or effort not in ("low", "medium", "xhigh"):
            raise ValueError("Versioned profile needs a supported policy reasoning effort")
        options["reasoning_effort"] = effort
    return options
