import json
from pathlib import Path

import pytest

import cycle_contract


def test_xhigh_schedule_requires_explicit_xhigh_and_a_larger_answer_budget():
    c = json.loads((Path(__file__).parent / "fixtures/cycle.json").read_text())
    c.update(
        schedule_id="audited-xhigh-20261009",
        initial_step=2,
        target_step=3,
        row_id="audit/group-1",
        dataset_workspace="/dataset",
        judge_operation_cap_usd=150,
        policy_reasoning_effort="xhigh",
    )
    c["sampling"]["max_tokens"] = 32768
    c["engine"]["max_model_len"] = 65536
    cycle_contract.validate_config(c)
    c["policy_reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="reasoning"):
        cycle_contract.validate_config(c)


def test_low_prompt_cannot_be_relabelled_as_an_xhigh_training_request():
    c = {"policy_reasoning_effort": "xhigh", "sampling": {"max_tokens": 32768}}
    plan = {"sampling_profile": {"policy_reasoning_effort": "xhigh"}}
    request = {
        "template_options": {"enable_thinking": True, "reasoning_effort": "low"},
        "full_completion_ceiling": 32768,
    }
    with pytest.raises(ValueError, match="reasoning"):
        cycle_contract.validate_policy_profile(c, request, plan)
    request["template_options"]["reasoning_effort"] = "xhigh"
    cycle_contract.validate_policy_profile(c, request, plan)
    request["full_completion_ceiling"] = 16384
    with pytest.raises(ValueError, match="ceiling"):
        cycle_contract.validate_policy_profile(c, request, plan)
