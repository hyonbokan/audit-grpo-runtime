"""Pilot boundaries: private joins, sampled-token provenance and whole-group grading."""

import copy
import json

import pytest

from testing.judge import Judge, contract, finding, sample


def test_policy_dataset_excludes_private_targets_and_rejects_changed_prompt(workspace):
    from grpo import reward

    records, private = reward.load_split(workspace, "train")
    assert [record["row_id"] for record in records] == list(private) == ["audit/group-123"]
    assert all(set(record) == {"row_id", "prompt"} for record in records)
    row = records[0]
    assert "scorer_metadata" not in row
    assert private[row["row_id"]]["scorer_metadata"]["eligible_targets"]
    changed = copy.deepcopy(row["prompt"])
    changed[-1]["content"] += "\nPrivate target inserted."
    with pytest.raises(ValueError, match="policy prompt"):
        reward.validate_policy(changed, private[row["row_id"]])


class Tokenizer:
    def decode(self, ids, **_kwargs):
        return (
            {
                7: "<think>Guard absent; caller drains balance.</think>",
                8: '{"findings":[]}',
                9: "<|im_end|>",
            }[ids[0]]
            if len(ids) == 1
            else ('<think>Guard absent; caller drains balance.</think>{"findings":[]}<|im_end|>')
        )


def test_sample_keeps_original_tokens_probabilities_and_reasoning_final_join():
    from grpo import reward

    sample = reward.capture_sample(
        "row", [1, 2], [7, 8, 9], [-0.1, -0.2, -0.3], Tokenizer(), finish_reason="stop"
    )
    assert sample["completion_ids"] == [7, 8, 9]
    assert sample["logprobs"] == [-0.1, -0.2, -0.3]
    assert sample["reasoning"] == "Guard absent; caller drains balance."
    assert sample["final_text"] == '{"findings":[]}'
    assert sample["raw_text"].endswith("<|im_end|>")
    with pytest.raises(ValueError, match="token"):
        reward.capture_sample("row", [1], [7, 8], [-0.1], Tokenizer(), finish_reason="stop")


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ("medium", "high"))
async def test_scalar_uses_structured_description_and_duplicate_raw_count(
    private, tmp_path, effort
):
    from grpo import reward

    selected = contract()
    selected["judge"]["reasoning_effort"] = effort
    result = await reward.grade_group(
        [sample(), sample([finding(), finding()])],
        private,
        selected,
        Judge(effort=effort),
        tmp_path,
        num_generations=2,
    )
    assert result == [1.2, 1.15]
    saved = json.loads((tmp_path / "attempt-1/grades.json").read_text())
    assert saved[1]["components"]["hits"] == 1
    assert saved[1]["components"]["raw_findings"] == 2
    assert "No caller check" in saved[0]["case"]["payload"]["candidate_findings"][0]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("judge", "expected"), [(Judge(identity_only=True), -0.05), (Judge(contradiction=True), 0.95)]
)
async def test_identity_harm_and_contradiction_gates(private, tmp_path, judge, expected):
    from grpo import reward

    assert await reward.grade_group(
        [sample(), sample()], private, contract(), judge, tmp_path, num_generations=2
    ) == [expected, expected]


@pytest.mark.asyncio
async def test_format_floor_empty_and_overflow_do_not_call_judge(private, tmp_path):
    from grpo import reward

    async def no_judge(_request):
        pytest.fail("Format gates should not call the paid judge")

    result = await reward.grade_group(
        [
            sample(final_text="broken"),
            sample([]),
            sample([finding()] * 33),
            sample(finish_reason="length"),
        ],
        private,
        contract(),
        no_judge,
        tmp_path,
        num_generations=4,
    )
    assert result == [-2, 0, -2, -2]


@pytest.mark.asyncio
async def test_unknown_retries_whole_group_and_aborts_without_partial_rewards(private, tmp_path):
    from grpo import reward

    result = await reward.grade_group(
        [sample(), sample()],
        private,
        contract(),
        Judge(fail=1),
        tmp_path / "retry",
        num_generations=2,
        max_retries=1,
    )
    assert result == [1.2, 1.2]
    assert (tmp_path / "retry/attempt-2/grades.json").exists()
    with pytest.raises(reward.GroupWithheld):
        await reward.grade_group(
            [sample(), sample()],
            private,
            contract(),
            Judge(review=True),
            tmp_path / "withheld",
            num_generations=2,
            max_retries=1,
        )
    saved = json.loads((tmp_path / "withheld/attempt-2/grades.json").read_text())
    assert all(row["reward"] is None for row in saved)


@pytest.mark.asyncio
async def test_mixed_rows_or_short_group_are_rejected_before_grading(private, tmp_path):
    from grpo import reward

    with pytest.raises(ValueError, match="group"):
        await reward.grade_group(
            [sample()], private, contract(), Judge(), tmp_path, num_generations=2
        )
    with pytest.raises(ValueError, match="group"):
        await reward.grade_group(
            [sample(), sample(row_id="other")],
            private,
            contract(),
            Judge(),
            tmp_path,
            num_generations=2,
        )


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_grading_instead_of_withholding(private, tmp_path):
    from grpo import reward
    from grpo.budget import BudgetExhausted

    async def exhausted(_request):
        raise BudgetExhausted("Judge operation budget exhausted; next call withheld")

    with pytest.raises(BudgetExhausted):
        await reward.grade_group(
            [sample(), sample()],
            private,
            contract(),
            exhausted,
            tmp_path / "budget",
            num_generations=2,
            max_retries=1,
        )
    assert not (tmp_path / "budget/attempt-2").exists()


@pytest.mark.asyncio
async def test_group_retry_with_response_cache_pays_only_for_missing_calls(private, tmp_path):
    from grpo import reward
    from grpo.judge_cache import CachedJudge

    class Counting(Judge):
        calls = 0

        async def __call__(self, request):
            Counting.calls += 1
            # The last call of the first attempt fails; everything else succeeds.
            if Counting.calls == 4:
                raise TimeoutError("judge unavailable")
            return await super().__call__(request)

    judge = CachedJudge(Counting(), tmp_path / "cache")
    result = await reward.grade_group(
        [sample(), sample()],
        private,
        contract(),
        judge,
        tmp_path / "group",
        num_generations=2,
        max_retries=1,
    )
    assert result == [1.2, 1.2]
    assert (judge.paid_calls, judge.replayed) == (4, 3)
    assert Counting.calls == 5
