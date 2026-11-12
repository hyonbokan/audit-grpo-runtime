"""Benchmark votes, indices, and one-to-one matching use the delivered semantics."""

import asyncio

import pytest


def verdict(exact=False, partial=False, index=0):
    return {
        "is_match": exact,
        "is_partial_match": partial,
        "explanation": "reason",
        "severity_from_junior_auditor": "High",
        "severity_from_truth": "High",
        "index_of_finding_from_junior_auditor": index,
    }


def test_vote_prefers_exact_in_three_way_tie():
    from benchmark_matching import choose_vote

    exact = verdict(exact=True)
    assert choose_vote([verdict(), verdict(partial=True), exact]) == exact
    with pytest.raises(ValueError, match="unknown"):
        choose_vote([])


@pytest.mark.parametrize("agree,expected_calls", [(True, 2), (False, 3)])
def test_two_agreeing_iterations_skip_third_and_disagreement_uses_it(agree, expected_calls):
    from benchmark_matching import judge_vote

    calls = []

    async def call(prompt):
        calls.append(prompt)
        return verdict(exact=agree or len(calls) != 2)

    assert asyncio.run(judge_vote("prompt", call))["is_match"] is True
    assert len(calls) == expected_calls


def test_native_evaluator_never_reuses_exact_finding_or_counts_unknown_as_zero():
    from benchmark_matching import evaluate_findings

    findings = [{"issue": "one", "description": "body", "severity": "High", "contracts": ["A.sol"]}]
    truths = [dict(findings[0], issue="first"), dict(findings[0], issue="second")]

    async def judge(prompt):
        return verdict(exact=True)

    result = asyncio.run(
        evaluate_findings(findings, truths, "{ junior_findings } { truth_finding }", judge)
    )
    assert result["exact_matches"] == 1
    assert result["known_gt_count"] == 2
    assert result["exact_coverage"] == 0.5
    assert result["native_metrics"]["precision"] == 1
    assert result["native_metrics"]["f1"] == pytest.approx(2 / 3)
    assert result["unmatched_findings"] == 0


def test_judge_failure_withholds_coverage_instead_of_scoring_missing_finding():
    from benchmark_matching import evaluate_findings

    async def judge(prompt):
        raise RuntimeError("transport failed")

    finding = {"issue": "x", "description": "body", "severity": "High", "contracts": []}
    result = asyncio.run(evaluate_findings([finding], [finding], "prompt", judge))
    assert result["status"] == "unknown"
    assert result["exact_coverage"] is None
    assert result["known_gt_count"] == 1


@pytest.mark.parametrize("index", [-1, 30, 0.5])
def test_out_of_batch_indices_cannot_credit_other_findings(index):
    from benchmark_matching import working_index

    assert working_index(index, 3, 1) == -1
    assert working_index(2, 3, 1) == 32


def test_partial_finding_cannot_also_count_as_later_exact_match():
    from benchmark_matching import evaluate_findings

    finding = {"issue": "x", "description": "body", "severity": "High", "contracts": []}
    responses = iter([verdict(partial=True), verdict(exact=True)])

    async def judge(prompt):
        return next(responses)

    result = asyncio.run(evaluate_findings([finding], [finding, finding], "prompt", judge))
    assert result["partial_matches"] == 0
    assert result["exact_matches"] == 1


def test_positive_verdict_without_valid_finding_index_is_unknown():
    from benchmark_matching import evaluate_findings

    finding = {"issue": "x", "description": "body", "severity": "High", "contracts": []}

    async def judge(prompt):
        return verdict(exact=True, index=-1)

    result = asyncio.run(evaluate_findings([finding], [finding], "prompt", judge))
    assert result["status"] == "unknown"
    assert result["exact_coverage"] is None


@pytest.mark.parametrize("failing_call", [1, 3])
def test_budget_exhaustion_in_a_vote_stops_grading_instead_of_losing_a_vote(failing_call):
    from benchmark_matching import evaluate_findings, judge_vote
    from grpo.budget import BudgetExhausted

    calls = []

    async def call(prompt):
        calls.append(prompt)
        if len(calls) == failing_call:
            raise BudgetExhausted("cap")
        return verdict(exact=len(calls) == 1)

    async def judge(prompt):
        return await judge_vote(prompt, call)

    finding = {"issue": "x", "description": "body", "severity": "High", "contracts": []}
    with pytest.raises(BudgetExhausted, match="cap"):
        asyncio.run(evaluate_findings([finding], [finding], "prompt", judge))
    # The first two iterations run together, so both are sent before the cap stops the vote.
    assert len(calls) == max(failing_call, 2)


def test_budget_exhaustion_lets_the_vote_iteration_in_flight_finish():
    from benchmark_matching import judge_vote
    from grpo.budget import BudgetExhausted

    calls, finished = [], []

    async def call(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise BudgetExhausted("cap")
        await asyncio.sleep(0.01)
        finished.append(len(calls))
        return verdict(exact=True)

    with pytest.raises(BudgetExhausted, match="cap"):
        asyncio.run(judge_vote("prompt", call))
    # The paid call already sent settles and is saved before the cap stops grading.
    assert finished == [2]
