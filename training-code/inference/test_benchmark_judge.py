"""The paid boundary preserves native effort, raw evidence, and failed-call charges."""

import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 14), reason="Existing judge router is 3.14")


@pytest.mark.parametrize("failure,cached", [(False, 0), (False, 900), (True, 0)])
def test_judge_persists_success_or_failure_and_settles_shared_budget(tmp_path, failure, cached):
    from openai.types.responses import Response

    from benchmark_judge import BenchmarkJudge, MatchingVerdict
    from grpo.budget import JudgeBudget

    class RouterBoundary:
        async def complete(self, model, messages, **settings):
            assert model == "gpt-6-luna"
            assert messages == [{"role": "user", "content": "private judge prompt"}]
            assert settings["thinking_effort"].value == "xhigh"
            assert settings["max_retries"] == 0
            if failure:
                raise RuntimeError("API unavailable")
            content = MatchingVerdict(
                is_match=True,
                is_partial_match=False,
                explanation="matched",
                severity_from_junior_auditor="High",
                severity_from_truth="High",
                index_of_finding_from_junior_auditor=0,
            )
            raw = Response.model_construct(
                id="response-test",
                model=model,
                status="completed",
                usage={
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "input_tokens_details": {"cached_tokens": cached},
                },
            )
            return SimpleNamespace(
                content=content,
                raw_response=raw,
                input_tokens=1000 - cached,
                cache_read_tokens=cached,
                output_tokens=100,
            )

    budget = JudgeBudget(
        tmp_path / "ledger.json", cap=150, operation_id="test", operation_cap=10, model="gpt-6-luna"
    )
    judge = BenchmarkJudge(RouterBoundary(), tmp_path, budget)
    if failure:
        with pytest.raises(RuntimeError, match="unavailable"):
            asyncio.run(judge("private judge prompt"))
        assert json.loads((tmp_path / "call-0001/error.json").read_text())["type"] == "RuntimeError"
    else:
        result = asyncio.run(judge("private judge prompt"))
        assert result["is_match"] is True
        assert (tmp_path / "call-0001/raw.json").exists()
    record = json.loads(budget.path.read_text())["calls"][0]
    assert record["status"] == "settled"
    assert record["estimated_usd"] is None if failure else record["estimated_usd"] > 0
    assert record["budget_charge_usd"] > 0
    if not failure:
        assert record["usage"]["input_tokens"] == 1000
        assert record["usage"]["cached_input_tokens"] == cached


def test_budget_exhaustion_records_reason_without_calling_api(tmp_path):
    from benchmark_judge import BenchmarkJudge
    from grpo.budget import JudgeBudget

    class RouterBoundary:
        async def complete(self, *args, **kwargs):
            raise AssertionError("Exhausted operation cannot spend")

    budget = JudgeBudget(
        tmp_path / "ledger.json",
        cap=150,
        operation_id="test",
        operation_cap=0.01,
        model="gpt-6-luna",
    )
    with pytest.raises(ValueError, match="budget"):
        asyncio.run(BenchmarkJudge(RouterBoundary(), tmp_path, budget)("prompt"))
    error = json.loads((tmp_path / "call-0001/error.json").read_text())
    assert "operation budget" in error["error"]


@pytest.mark.parametrize("has_incomplete_response", [False, True])
def test_saved_votes_are_consumed_once_without_api_spend(tmp_path, has_incomplete_response):
    from benchmark_judge import BenchmarkJudge, MatchingVerdict
    from grpo.budget import JudgeBudget

    source = tmp_path / "saved" / "call-0001"
    source.mkdir(parents=True)
    verdict = {
        "is_match": True,
        "is_partial_match": False,
        "explanation": "match",
        "severity_from_junior_auditor": "High",
        "severity_from_truth": "High",
        "index_of_finding_from_junior_auditor": 0,
    }
    (source / "request.json").write_text(
        json.dumps(
            {
                "model": "gpt-6-luna",
                "api_effort": "xhigh",
                "prompt": "same prompt",
                "schema": MatchingVerdict.model_json_schema(),
            }
        )
    )
    (source / "raw.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(verdict)}],
                    }
                ],
            }
        )
    )

    if has_incomplete_response:
        incomplete = source.parent / "call-0002"
        incomplete.mkdir()
        (incomplete / "request.json").write_bytes((source / "request.json").read_bytes())
        (incomplete / "raw.json").write_text(json.dumps({"status": "incomplete", "output": []}))

    class NoApi:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("fresh call withheld")

    out = tmp_path / "recovered"
    out.mkdir()
    budget = JudgeBudget(tmp_path / "ledger.json", cap=150, model="gpt-6-luna")
    judge = BenchmarkJudge(NoApi(), out, budget, replay_path=source.parent)
    assert asyncio.run(judge("same prompt"))["is_match"]
    assert not budget.path.exists()
    assert judge.reused == 1
    with pytest.raises(RuntimeError, match="withheld"):
        asyncio.run(judge("same prompt"))
    assert judge.reused == 1


@pytest.fixture
def offline_grading(tmp_path, monkeypatch):
    from benchmark_judge import RecordedJudgeRouter
    from testing.benchmark import free_text_configuration

    class OfflineRouter:
        closed = False

        async def complete(self, *args, **kwargs):
            raise AssertionError("These fixtures must not invoke a paid call")

        async def close(self):
            self.closed = True

    router = OfflineRouter()

    async def create(*args, **kwargs):
        return router

    monkeypatch.setattr(RecordedJudgeRouter, "create", create)
    config = free_text_configuration()
    config.update(judge_ledger=str(tmp_path / "ledger.json"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "private-truth.json").write_text('[{"severity":"High"}]')
    (tmp_path / "judge-prompt.txt").write_text("unused: no findings")
    return router


def test_missing_dotenv_credentials_stop_before_benchmark_grading(tmp_path, offline_grading):
    from benchmark_judge import grade_benchmark

    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["judge_env_file"] = str(tmp_path / "missing.env")
    config_path.write_text(json.dumps(config))
    (tmp_path / "samples.json").write_text(
        json.dumps(
            [
                {
                    "final_text": "an audit",
                    "finish_reason": "stop",
                    "completion_ids": [3],
                }
            ]
            * 4
        )
    )
    with pytest.raises(ValueError, match="gateway credential file"):
        asyncio.run(grade_benchmark(config_path, tmp_path / "samples.json", tmp_path / "grading"))
    assert not (tmp_path / "grading").exists()


@pytest.mark.parametrize(
    "text,finish,expected", [("", "stop", "no_answer"), ("an audit", "length", "truncated")]
)
def test_grading_records_input_identity_without_paid_calls(
    tmp_path, offline_grading, text, finish, expected
):
    from benchmark_judge import grade_benchmark

    (tmp_path / "samples.json").write_text(
        json.dumps([{"final_text": text, "finish_reason": finish, "completion_ids": [3]}] * 4)
    )
    out = tmp_path / "grading"
    asyncio.run(grade_benchmark(tmp_path / "config.json", tmp_path / "samples.json", out))
    measured = json.loads((out / "metrics.json").read_text())
    assert measured["input_sha256"] == {
        "samples": hashlib.sha256((tmp_path / "samples.json").read_bytes()).hexdigest(),
        "private_truth": hashlib.sha256((tmp_path / "private-truth.json").read_bytes()).hexdigest(),
        "judge_prompt": hashlib.sha256((tmp_path / "judge-prompt.txt").read_bytes()).hexdigest(),
    }
    assert [row["status"] for row in measured["draws"]] == [expected] * 4
    # A draw with no answer to grade is unknown, never zero coverage.
    assert [row["exact_coverage"] for row in measured["draws"]] == [None] * 4
    assert measured["judge_calls"] == 0
    assert offline_grading.closed
    assert not (tmp_path / "ledger.json").exists()


def test_invalid_replay_contract_closes_router(tmp_path, offline_grading):
    from benchmark_judge import grade_benchmark

    (tmp_path / "samples.json").write_text("[{}, {}, {}, {}]")
    source = tmp_path / "saved" / "call-0001"
    source.mkdir(parents=True)
    (source / "request.json").write_text('{"model":"different"}')
    (source / "raw.json").write_text('{"status":"completed"}')
    with pytest.raises(ValueError, match="contract differs"):
        asyncio.run(
            grade_benchmark(
                tmp_path / "config.json", tmp_path / "samples.json", tmp_path / "out", source.parent
            )
        )
    assert offline_grading.closed


def test_free_text_draws_are_graded_concurrently(tmp_path, monkeypatch):
    from openai.types.responses import Response

    from benchmark_judge import (
        ExtractedReport,
        MatchingVerdict,
        RecordedJudgeRouter,
        grade_benchmark,
    )
    from testing.benchmark import free_text_configuration

    seen = set()
    released = asyncio.Event()

    class RouterBoundary:
        async def complete(self, model, messages, *, response_model, **settings):
            raw = Response.model_construct(
                id="test",
                model=model,
                status="completed",
                usage={"input_tokens": 100, "output_tokens": 20},
            )
            if response_model is ExtractedReport:
                # Every draw's extraction must be in flight before any of them returns.
                seen.update(i for i in range(4) if f"draw-{i}" in messages[0]["content"])
                if len(seen) == 4:
                    released.set()
                await released.wait()
                finding = {"issue": "GT", "description": "body", "severity": "High"}
                content = ExtractedReport(findings=[dict(finding, contracts=["A.sol"])])
            else:
                content = MatchingVerdict(
                    is_match=True,
                    is_partial_match=False,
                    explanation="same identity",
                    severity_from_junior_auditor="High",
                    severity_from_truth="High",
                    index_of_finding_from_junior_auditor=0,
                )
            return SimpleNamespace(content=content, raw_response=raw)

        async def close(self):
            pass

    async def create(*args, **kwargs):
        return RouterBoundary()

    monkeypatch.setattr(RecordedJudgeRouter, "create", create)
    config = free_text_configuration()
    config.update(grading_concurrency=4, judge_ledger=str(tmp_path / "ledger.json"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    finding = {"issue": "GT", "description": "body", "contracts": ["A.sol"], "severity": "High"}
    (tmp_path / "private-truth.json").write_text(json.dumps([finding]))
    (tmp_path / "judge-prompt.txt").write_text("{ junior_findings } { truth_finding }")
    samples = [
        {
            "final_text": f"audit of draw-{i}",
            "raw_text": "</think>",
            "finish_reason": "stop",
            "completion_ids": [3],
        }
        for i in range(4)
    ]
    (tmp_path / "samples.json").write_text(json.dumps(samples))

    async def run():
        await asyncio.wait_for(
            grade_benchmark(
                tmp_path / "config.json", tmp_path / "samples.json", tmp_path / "grading"
            ),
            timeout=2,
        )

    asyncio.run(run())
    metrics = json.loads((tmp_path / "grading/metrics.json").read_text())
    assert [draw["exact_matches"] for draw in metrics["draws"]] == [1] * 4
    assert metrics["valid_response_rate"] == 1
    assert len(list((tmp_path / "grading/extraction").glob("call-*"))) == 4


def test_judge_accepts_another_response_schema_for_the_same_paid_boundary(tmp_path):
    from pydantic import BaseModel

    from benchmark_judge import BenchmarkJudge
    from grpo.budget import JudgeBudget

    class Other(BaseModel):
        value: str

    seen = {}

    class RouterBoundary:
        async def complete(self, model, messages, **settings):
            seen["model"] = settings["response_model"]
            raise RuntimeError("stop here")

    budget = JudgeBudget(
        tmp_path / "ledger.json", cap=150, operation_id="test", operation_cap=10, model="gpt-6-luna"
    )
    with pytest.raises(RuntimeError):
        asyncio.run(BenchmarkJudge(RouterBoundary(), tmp_path, budget, Other)("prompt"))
    assert seen["model"] is Other
    assert json.loads((tmp_path / "call-0001/request.json").read_text())["schema"] == (
        Other.model_json_schema()
    )


def test_free_text_draws_are_extracted_once_then_matched_with_separate_records(
    tmp_path, monkeypatch
):
    from openai.types.responses import Response

    from benchmark_judge import (
        ExtractedReport,
        MatchingVerdict,
        RecordedJudgeRouter,
        grade_benchmark,
    )
    from testing.benchmark import free_text_configuration

    class RouterBoundary:
        async def complete(self, model, messages, *, response_model, **settings):
            raw = Response.model_construct(
                id="test",
                model=model,
                status="completed",
                usage={"input_tokens": 100, "output_tokens": 20},
            )
            if response_model is ExtractedReport:
                assert "free-text audit of draw" in messages[0]["content"]
                content = ExtractedReport(
                    findings=[
                        {
                            "issue": "Reentrancy",
                            "description": "drains funds",
                            "severity": "N/A",
                            "contracts": [],
                        }
                    ]
                )
            else:
                content = MatchingVerdict(
                    is_match=True,
                    is_partial_match=False,
                    explanation="same",
                    severity_from_junior_auditor="N/A",
                    severity_from_truth="High",
                    index_of_finding_from_junior_auditor=0,
                )
            return SimpleNamespace(content=content, raw_response=raw)

        async def close(self):
            pass

    async def create(*args, **kwargs):
        return RouterBoundary()

    monkeypatch.setattr(RecordedJudgeRouter, "create", create)
    config = free_text_configuration()
    config.update(judge_ledger=str(tmp_path / "ledger.json"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    finding = {"issue": "GT", "description": "body", "contracts": ["A.sol"], "severity": "High"}
    (tmp_path / "private-truth.json").write_text(json.dumps([finding]))
    (tmp_path / "judge-prompt.txt").write_text("{ junior_findings } { truth_finding }")
    samples = [
        {
            "final_text": f"free-text audit of draw {i}",
            "raw_text": "</think>",
            "finish_reason": "stop",
            "completion_ids": [3],
        }
        for i in range(4)
    ]
    (tmp_path / "samples.json").write_text(json.dumps(samples))
    asyncio.run(
        grade_benchmark(tmp_path / "config.json", tmp_path / "samples.json", tmp_path / "g")
    )
    metrics = json.loads((tmp_path / "g/metrics.json").read_text())
    assert [draw["exact_coverage"] for draw in metrics["draws"]] == [1.0] * 4
    assert [draw["answer_source"] for draw in metrics["draws"]] == ["final_answer"] * 4
    assert len(list((tmp_path / "g/extraction").glob("call-*"))) == 4
    assert len(list((tmp_path / "g").glob("call-*"))) == 8  # two agreeing match votes per draw
