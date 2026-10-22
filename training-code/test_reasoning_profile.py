import pytest

import cycle_contract


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
