"""Free-text reward: recall of the known findings, with no gain from listing more."""

import json

import pytest

from testing.judge import TEMPLATE, TRUTH, Judges, matching, report_finding, sample

REWARD = {"version": "free-text-recall-volume-v1", "volume_allowance": 3}


async def grade(judges, samples, row, out, *, reward=REWARD, **options):
    from grpo import reward as free_text_reward

    return await free_text_reward.grade_group(
        samples,
        row,
        reward,
        TEMPLATE,
        judges.extract,
        judges.match,
        judges.new_attempt,
        out,
        num_generations=len(samples),
        **options,
    )


def padded(count):
    return [report_finding(f"extra {i}") for i in range(count)]


@pytest.mark.asyncio
async def test_reward_is_recall_until_the_volume_allowance_then_pays_less_for_padding(
    tmp_path, row
):
    both = [matching("Missing guard"), matching("Stale price")]
    reports = {
        "none": [],
        "half": [matching("Missing guard")],
        "both": both,
        "both padded to 6": both + padded(4),
        "half padded to 6": [matching("Missing guard"), *padded(5)],
    }
    judges = Judges(reports)
    samples = [sample(text) for text in reports]
    rewards = await grade(judges, samples, row, tmp_path)
    assert rewards == pytest.approx([0.0, 0.5, 1.0, 1.0 * 3 / 6, 0.5 * 3 / 6])


@pytest.mark.asyncio
async def test_listing_up_to_the_allowance_costs_nothing(tmp_path, row):
    judges = Judges({"a": [matching("Missing guard"), *padded(2)]})
    assert await grade(judges, [sample("a")] * 2, row, tmp_path) == [0.5, 0.5]


@pytest.mark.asyncio
async def test_a_truncated_draw_scores_zero_is_recorded_and_never_judged(tmp_path, row):
    judges = Judges({"ok": [matching("Missing guard")]})
    samples = [sample("ok"), sample("cut off", finish_reason="length")]
    rewards = await grade(judges, samples, row, tmp_path)
    assert rewards == [0.5, 0.0]
    saved = json.loads((tmp_path / "attempt-1/grades.json").read_text())
    assert [grade_["status"] for grade_ in saved] == ["graded", "truncated"]
    assert saved[1]["reward"] == 0.0
    assert not any("cut off" in prompt for prompt in judges.extract_prompts)


@pytest.mark.asyncio
async def test_each_attempt_records_the_exact_counts_behind_every_reward(tmp_path, row):
    judges = Judges({"a": [matching("Stale price"), *padded(3)]})
    await grade(judges, [sample("a")] * 2, row, tmp_path)
    saved = json.loads((tmp_path / "attempt-1/grades.json").read_text())[0]
    assert saved["sample_index"] == 0
    assert (saved["exact_matches"], saved["known_gt_count"], saved["findings_count"]) == (1, 2, 4)
    assert saved["volume_factor"] == 3 / 4
    assert saved["reward"] == 0.5 * 3 / 4
    assert json.loads((tmp_path / "samples.json").read_text()) == [sample("a")] * 2


@pytest.mark.asyncio
async def test_an_unavailable_extraction_regrades_the_group_once_then_succeeds(tmp_path, row):
    judges = Judges({"a": [matching("Missing guard")]}, failures=1)
    rewards = await grade(judges, [sample("a")] * 2, row, tmp_path)
    assert rewards == [0.5, 0.5]
    assert judges.attempts == 2
    first = json.loads((tmp_path / "attempt-1/grades.json").read_text())
    assert [g["status"] for g in first] == ["unknown", "graded"]
    assert first[0]["reward"] is None and "extraction unavailable" in first[0]["error"]


@pytest.mark.asyncio
async def test_a_group_that_stays_unknown_is_withheld_never_scored_zero(tmp_path, row):
    from grpo.reward import GroupWithheld

    judges = Judges({"a": []}, failures=4)
    with pytest.raises(GroupWithheld, match="withheld after 2 attempts"):
        await grade(judges, [sample("a")] * 2, row, tmp_path)


@pytest.mark.asyncio
async def test_an_empty_answer_that_stopped_is_unknown_not_zero(tmp_path, row):
    from grpo.reward import GroupWithheld

    judges = Judges({})
    with pytest.raises(GroupWithheld):
        await grade(judges, [sample("", finish_reason="stop")] * 2, row, tmp_path, max_retries=0)


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_grading_instead_of_becoming_unknown(tmp_path, row):
    from grpo.budget import BudgetExhausted

    judges = Judges({"a": []})

    async def exhausted(_prompt):
        raise BudgetExhausted("cap")

    judges.extract = exhausted
    with pytest.raises(BudgetExhausted):
        await grade(judges, [sample("a")] * 2, row, tmp_path)


@pytest.mark.asyncio
async def test_a_group_must_be_complete_and_one_row(tmp_path, row):
    from grpo import reward as free_text_reward

    judges = Judges({"a": []})
    with pytest.raises(ValueError, match="Incomplete"):
        await free_text_reward.grade_group(
            [sample("a")], row, REWARD, TEMPLATE, judges.extract, judges.match,
            judges.new_attempt, tmp_path, num_generations=4,
        )  # fmt: skip
    mixed = [sample("a"), sample("a", row_id="other/group")]
    with pytest.raises(ValueError, match="Mixed row"):
        await grade(judges, mixed, row, tmp_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"volume_allowance": None}, "volume_allowance.*O1"),
        ({"volume_allowance": 0}, "volume_allowance"),
        ({"volume_allowance": 2.5}, "volume_allowance"),
        ({"volume_allowance": True}, "volume_allowance"),
        ({"version": ""}, "version"),
        ({"format_floor": -2}, "Unknown"),
    ],
)
def test_the_reward_contract_refuses_an_unset_or_unknown_setting(change, message):
    from grpo import reward as free_text_reward

    with pytest.raises(ValueError, match=message):
        free_text_reward.validate_reward({**REWARD, **change})


def test_the_contract_identity_changes_with_the_allowance_and_the_judge_template():
    from grpo import reward as free_text_reward

    base = free_text_reward.contract_sha256(REWARD, TEMPLATE)
    assert base == free_text_reward.contract_sha256(dict(REWARD), TEMPLATE)
    assert base != free_text_reward.contract_sha256({**REWARD, "volume_allowance": 4}, TEMPLATE)
    assert base != free_text_reward.contract_sha256(REWARD, TEMPLATE + " ")


def test_policy_rows_carry_no_private_fields_and_must_match_the_code_rendering(corpus):
    from grpo import reward as free_text_reward

    records, private = free_text_reward.load_split(corpus, "train")
    assert [r["row_id"] for r in records] == list(private) == ["audit/group-123"]
    assert all(set(r) == {"row_id", "prompt"} for r in records)
    assert private["audit/group-123"]["truth"] == TRUTH
    path = corpus / "data/policy/train.jsonl"
    policy = json.loads(path.read_text())
    policy["messages"][0]["content"] += "\nPrivate target inserted."
    path.write_text(json.dumps(policy) + "\n")
    with pytest.raises(ValueError, match="policy prompt"):
        free_text_reward.load_split(corpus, "train")


def test_a_policy_row_with_extra_fields_is_refused(corpus):
    from grpo import reward as free_text_reward

    path = corpus / "data/policy/train.jsonl"
    policy = json.loads(path.read_text())
    policy["truth"] = []
    path.write_text(json.dumps(policy) + "\n")
    with pytest.raises(ValueError, match="Private fields"):
        free_text_reward.load_split(corpus, "train")
