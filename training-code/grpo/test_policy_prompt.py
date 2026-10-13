import hashlib
import json

import pytest

from grpo.policy_prompt import (
    judging_contract,
    load_policy_rows,
    policy_template_options,
    prompt_token_ids,
)


class EffortTokenizer:
    def apply_chat_template(self, _messages, **options):
        return [1, 2, 3] if options["reasoning_effort"] == "medium" else [1, 2, 3, 4, 5]


def test_versioned_profile_selects_policy_effort():
    plan = {"sampling_profile": {"id": "schema-medium-8k-v1", "policy_reasoning_effort": "medium"}}
    options = policy_template_options(plan)
    assert options["reasoning_effort"] == "medium"
    assert prompt_token_ids(EffortTokenizer(), [], template_options=options) == [1, 2, 3]
    assert prompt_token_ids(EffortTokenizer(), []) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("profile", [{"id": "x", "policy_reasoning_effort": "high"}, {"id": ""}])
def test_profile_rejects_unsupported_or_unnamed_effort(profile):
    with pytest.raises(ValueError, match="reasoning effort"):
        policy_template_options({"sampling_profile": profile})


def test_policy_rows_reject_private_fields(tmp_path):
    policy = tmp_path / "data/policy"
    policy.mkdir(parents=True)
    for split in ("train", "calibration", "evaluation"):
        row = {"row_id": split, "messages": [{"role": "user", "content": "code"}]}
        (policy / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    assert load_policy_rows(tmp_path)["train"] == [
        {"row_id": "train", "prompt": [{"role": "user", "content": "code"}]}
    ]
    (policy / "train.jsonl").write_text(json.dumps({"row_id": "t", "messages": [], "gt": 1}))
    with pytest.raises(ValueError, match="Private fields"):
        load_policy_rows(tmp_path)


def test_judging_contract_binds_contract_bytes(tmp_path):
    (tmp_path / "configs").mkdir()
    data = json.dumps({"judge": {"model": "m", "reasoning_effort": "medium"}}).encode()
    (tmp_path / "configs/reward_contract.json").write_bytes(data)
    contract = judging_contract(tmp_path)
    assert contract["model"] == "m" and contract["reasoning_effort"] == "medium"
    assert contract["reward_contract_sha256"] == hashlib.sha256(data).hexdigest()
