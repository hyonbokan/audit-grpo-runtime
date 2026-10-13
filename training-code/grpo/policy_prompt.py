"""Policy prompt rendering, policy rows and the pinned judge contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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


def load_policy_rows(workspace: Path) -> dict[str, list[dict]]:
    """Policy-visible rows per split; private fields or duplicate rows are rejected."""
    splits = {}
    identities = set()
    for split in ("train", "calibration", "evaluation"):
        rows = [
            json.loads(line)
            for line in (workspace / f"data/policy/{split}.jsonl").read_text().splitlines()
        ]
        for row in rows:
            if set(row) != {"row_id", "messages"} or row["row_id"] in identities:
                raise ValueError("Private fields or duplicate row in policy records")
            identities.add(row["row_id"])
        splits[split] = [{"row_id": row["row_id"], "prompt": row["messages"]} for row in rows]
    return splits


def judging_contract(workspace: Path) -> dict[str, str]:
    data = (workspace / "configs/reward_contract.json").read_bytes()
    judge = json.loads(data)["judge"]
    return {
        "model": judge["model"],
        "reasoning_effort": judge["reasoning_effort"],
        "reward_contract_sha256": hashlib.sha256(data).hexdigest(),
    }
