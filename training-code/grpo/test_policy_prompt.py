import pytest

from grpo.policy_prompt import (
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
