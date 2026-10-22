"""Router boundary and persisted usage without paid calls."""

import json
from types import SimpleNamespace

import pytest

from testing.judge import contract, sample


@pytest.mark.asyncio
@pytest.mark.parametrize("has_usage", (True, False))
@pytest.mark.parametrize("effort", ("medium", "high"))
@pytest.mark.parametrize("cached", (0, 90))
async def test_router_sends_selected_schema_effort_and_records_response_usage(
    tmp_path, has_usage, effort, cached
):
    from grpo import judge_worker
    from grpo.budget import JudgeBudget

    class Router:
        async def complete(self, model, messages, **options):
            assert model == "gpt-6.1-sol"
            assert options["thinking_effort"].value == effort
            assert options["max_tokens"] is None
            assert options["max_retries"] == 0
            assert options["response_model"].__name__ == "MatchResponse"
            assert json.loads(messages[0]["content"])["candidate_findings"] == []
            return SimpleNamespace(
                content=options["response_model"](assessments=[]),
                input_tokens=100 - cached,
                output_tokens=20,
                cache_read_tokens=cached,
                raw_response=SimpleNamespace(
                    model="gpt-6.1-sol",
                    id="r0",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=20,
                        input_tokens_details=SimpleNamespace(cached_tokens=cached),
                    )
                    if has_usage
                    else None,
                    model_dump_json=lambda **_: (
                        '{"id":"r0","model":"gpt-6.1-sol","status":"completed"}'
                    ),
                ),
            )

    budget = JudgeBudget(tmp_path / "ledger.json", cap=150)
    adapter = judge_worker.RouterJudge(Router(), tmp_path, budget=budget)
    result = await adapter(
        {
            "method": "sgr_match",
            "model": "gpt-6.1-sol",
            "reasoning_effort": effort,
            "system": "Selected system",
            "payload": {"candidate_findings": []},
        }
    )
    assert result == {"assessments": []}
    usage = json.loads((tmp_path / "call-0001/usage.json").read_text())
    assert usage["input_tokens"] == (100 if has_usage else 100 - cached)
    assert usage["output_tokens"] == 20
    assert usage["response_id"] == "r0"
    request = json.loads((tmp_path / "call-0001/request.json").read_text())
    assert request["reasoning_effort"] == effort
    assert request["schema"]["title"] == "MatchResponse"
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["estimated_usd"] == pytest.approx(
        ((100 - cached) * 2 + cached * 0.1 + 20 * 10) / 1e6 if has_usage else 0
    )
    assert ledger["calls"][0]["status"] == "settled"
    reservation = json.loads((tmp_path / "call-0001/budget-reservation.json").read_text())
    assert reservation["reservation_id"] == ledger["calls"][0]["id"]
    if not has_usage:
        assert ledger["calls"][0]["budget_charge_usd"] == ledger["calls"][0]["reserved_usd"]


@pytest.mark.asyncio
async def test_budget_withholds_api_call_after_unknown_usage(tmp_path):
    from grpo import judge_worker
    from grpo.budget import JudgeBudget

    class Router:
        calls = 0

        async def complete(self, *_args, **_options):
            self.calls += 1
            raise TimeoutError("No response received")

    router = Router()
    adapter = judge_worker.RouterJudge(
        router, tmp_path, budget=JudgeBudget(tmp_path / "ledger.json", cap=6)
    )
    request = {
        "method": "sgr_match",
        "model": "gpt-6.1-sol",
        "reasoning_effort": "high",
        "system": "Selected",
        "payload": {"candidate_findings": []},
    }
    with pytest.raises(TimeoutError):
        await adapter(request)
    with pytest.raises(ValueError, match="budget"):
        await adapter(request)
    assert router.calls == 1


@pytest.mark.asyncio
async def test_selected_row_judging_stops_after_one_failed_group_without_retry(tmp_path, private):
    from grpo.judge_worker import grade_selected_group
    from grpo.reward import GroupWithheld

    class FailingJudge:
        calls = 0

        async def __call__(self, _request):
            self.calls += 1
            raise TimeoutError("No paid response")

    judge = FailingJudge()
    with pytest.raises(GroupWithheld):
        await grade_selected_group(
            [sample()] * 4,
            private,
            contract(),
            judge,
            tmp_path,
            {"selected_row_id": "row", "judge_retry_limit": 0},
        )
    assert judge.calls == 4
    assert (tmp_path / "attempt-1/grades.json").exists()
    assert not (tmp_path / "attempt-2").exists()


@pytest.mark.asyncio
async def test_selected_row_mismatch_prevents_any_judge_request(tmp_path, private):
    from grpo.judge_worker import grade_selected_group

    class UnexpectedJudge:
        async def __call__(self, _request):
            pytest.fail("Swapped row must be rejected before paid work")

    with pytest.raises(ValueError, match="row"):
        await grade_selected_group(
            [sample()] * 4,
            private,
            contract(),
            UnexpectedJudge(),
            tmp_path,
            {"selected_row_id": "another-row", "judge_retry_limit": 0},
        )


@pytest.mark.asyncio
async def test_missing_dotenv_credentials_stop_before_training_grading(
    tmp_path, monkeypatch, workspace
):
    from grpo import judge_worker

    async def unexpected_router(*_args, **_kwargs):
        pytest.fail("Missing gateway credentials must stop before router initialization")

    monkeypatch.setattr(judge_worker.RecordedJudgeRouter, "create", unexpected_router)
    plan = {
        "judge": judge_worker.judging_contract(workspace),
        "judge_env_file": str(tmp_path / "missing.env"),
        "access": {
            "local_judge_budget": {"path": str(tmp_path / "ledger.json")},
            "judge_usd_cap": 150,
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([sample()] * 4))
    args = SimpleNamespace(
        plan=plan_path,
        workspace=workspace,
        split="train",
        samples=samples,
        out=tmp_path / "grading",
    )
    with pytest.raises(ValueError, match="gateway credential file"):
        await judge_worker.run(args)
    assert not args.out.exists()


def transient_router(failures, error):
    from grpo.reward import response_schema

    class Router:
        calls = 0

        async def complete(self, *_args, **options):
            Router.calls += 1
            if Router.calls <= failures:
                raise error
            return SimpleNamespace(
                content=response_schema("sgr_match")(assessments=[]),
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                raw_response=SimpleNamespace(
                    model="gpt-6.1-sol",
                    id="r1",
                    status="completed",
                    usage=None,
                    model_dump_json=lambda **_: "{}",
                ),
            )

    return Router


MATCH_REQUEST = {
    "method": "sgr_match",
    "model": "gpt-6.1-sol",
    "reasoning_effort": "high",
    "system": "Selected",
    "payload": {"candidate_findings": []},
}


@pytest.mark.asyncio
async def test_transient_transport_failure_is_retried_with_separate_evidence(tmp_path):
    from llm_router.errors import TransientProviderError

    from grpo import judge_worker
    from grpo.budget import JudgeBudget

    router = transient_router(2, TransientProviderError("peer closed connection"))
    budget = JudgeBudget(tmp_path / "ledger.json", cap=150)
    (tmp_path / "calls").mkdir()
    judge = judge_worker.RouterJudge(
        router(), tmp_path / "calls", budget=budget, retries=2, backoff_seconds=0
    )
    assert await judge(MATCH_REQUEST) == {"assessments": []}
    assert router.calls == 3
    assert json.loads((tmp_path / "calls/call-0001/error.json").read_text())["type"] == (
        "TransientProviderError"
    )
    assert (tmp_path / "calls/call-0003/usage.json").exists()
    # Failed attempts keep their conservative reservations in the shared ledger.
    calls = json.loads((tmp_path / "ledger.json").read_text())["calls"]
    assert [call["status"] for call in calls] == ["settled"] * 3
    assert all(call["budget_charge_usd"] == call["reserved_usd"] > 0 for call in calls[:2])


@pytest.mark.asyncio
async def test_transient_retries_are_bounded_and_other_errors_are_not_retried(tmp_path):
    from llm_router.errors import TransientProviderError

    from grpo import judge_worker

    for name in "abc":
        (tmp_path / name).mkdir()
    router = transient_router(5, TransientProviderError("gateway down"))
    judge = judge_worker.RouterJudge(router(), tmp_path / "a", retries=1, backoff_seconds=0)
    with pytest.raises(TransientProviderError):
        await judge(MATCH_REQUEST)
    assert router.calls == 2

    router = transient_router(5, ValueError("schema mismatch"))
    judge = judge_worker.RouterJudge(router(), tmp_path / "b", retries=3, backoff_seconds=0)
    with pytest.raises(ValueError):
        await judge(MATCH_REQUEST)
    assert router.calls == 1

    with pytest.raises(ValueError, match="retries"):
        judge_worker.RouterJudge(router(), tmp_path / "c", retries=9)
