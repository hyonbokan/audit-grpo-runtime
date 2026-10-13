"""Exact saved judge responses are consumed once; only the missing call can be paid."""

import asyncio

import pytest


def request(number):
    return {"model": "gpt-6.1-sol", "reasoning_effort": "medium", "payload": {"number": number}}


def test_recovery_replays_completed_responses_and_allows_only_one_known_missing_call():
    from grpo.judge_recovery import RecoveryJudge

    paid_requests = []

    async def paid(value):
        paid_requests.append(value)
        return {"answer": "new"}

    grades = [
        {"judge_requests": [request(1), request(2)], "judge_responses": [{"answer": "saved"}]}
    ]
    judge = RecoveryJudge(grades, paid)
    assert asyncio.run(judge(request(1))) == {"answer": "saved"}
    assert asyncio.run(judge(request(2))) == {"answer": "new"}
    judge.finish()
    assert paid_requests == [request(2)]
    with pytest.raises(ValueError):
        asyncio.run(judge(request(1)))
    with pytest.raises(ValueError):
        asyncio.run(judge(request(2)))
    assert paid_requests == [request(2)]


def test_unexpected_request_and_unconsumed_responses_fail_closed_without_payment():
    from grpo.judge_recovery import RecoveryJudge

    async def forbidden(_request):
        raise AssertionError("Unexpected paid call")

    judge = RecoveryJudge(
        [{"judge_requests": [request(1), request(2)], "judge_responses": [{"answer": "saved"}]}],
        forbidden,
    )
    with pytest.raises(ValueError):
        asyncio.run(judge(request(3)))
    with pytest.raises(ValueError):
        judge.finish()


def test_multiple_missing_calls_are_not_an_authorized_recovery():
    from grpo.judge_recovery import RecoveryJudge

    with pytest.raises(ValueError):
        RecoveryJudge([{"judge_requests": [request(1), request(2)], "judge_responses": []}], None)
