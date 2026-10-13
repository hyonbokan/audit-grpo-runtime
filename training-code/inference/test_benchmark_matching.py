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


def test_empty_findings_keeps_full_gt_denominator_without_judge_calls():
    from benchmark_matching import evaluate_findings

    async def judge(prompt):
        raise AssertionError("Empty scan needs no paid calls")

    truth = {"issue": "x", "description": "body", "severity": "High", "contracts": []}
    result = asyncio.run(evaluate_findings([], [truth], "unused", judge))
    assert result["exact_coverage"] == 0
    assert result["known_gt_count"] == 1


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
