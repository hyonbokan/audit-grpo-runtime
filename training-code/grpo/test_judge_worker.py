"""Worker boundary: the evaluation's judges behind a cache, without paid calls."""

import json
from types import SimpleNamespace

import pytest

from testing.judge import FakeRouter, matching, sample

REWARD = {"version": "free-text-recall-volume-v1", "volume_allowance": 3}
REPORTS = {"found": [matching("Missing guard")], "nothing": []}
TEMPLATE = "Truth:\n{ truth_finding }\nFindings:\n{ junior_findings }"


def benchmark_judge():
    from benchmark_free_text import SCOPE_PATH
    from runtime_io import read_json

    return read_json(SCOPE_PATH)["benchmark_judge"]


def plan_for(tmp_path, corpus, **changes):
    return {
        "reward": REWARD,
        "benchmark_judge": benchmark_judge(),
        "dataset_workspace": str(corpus),
        "selected_row_id": "audit/group-123",
        "judge_retry_limit": 1,
        "judge_operation": {"id": "test-op", "cap_usd": 100},
        "access": {
            "local_judge_budget": {"path": str(tmp_path / "ledger.json")},
            "judge_usd_cap": 150,
        },
        **changes,
    }


def write_args(tmp_path, plan, samples):
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "samples.json").write_text(json.dumps(samples))
    return SimpleNamespace(
        plan=tmp_path / "plan.json",
        samples=tmp_path / "samples.json",
        split="train",
        out=tmp_path / "grading",
        cache=tmp_path / "cache",
    )


def patch_router(monkeypatch, router):
    from grpo import judge_worker

    async def create(_config):
        return router

    monkeypatch.setattr(judge_worker.RecordedJudgeRouter, "create", create)
    monkeypatch.setattr(judge_worker, "judge_router_config", lambda *_a, **_k: object())


@pytest.mark.asyncio
async def test_a_group_is_graded_cached_and_summarised(tmp_path, corpus, monkeypatch):
    from grpo import judge_worker

    router = FakeRouter(REPORTS)
    patch_router(monkeypatch, router)
    samples = [sample("found"), sample("nothing"), sample("found"), sample("cut", "length")]
    args = write_args(tmp_path, plan_for(tmp_path, corpus), samples)
    await judge_worker.run(args)
    assert json.loads((args.out / "rewards.json").read_text()) == [0.5, 0.0, 0.5, 0.0]
    metrics = json.loads((args.out / "metrics.json").read_text())
    assert (metrics["caught_gt_ids"], metrics["truncated_draws"]) == (["archive:1"], 1)
    contract = json.loads((args.out / "reward-contract.json").read_text())
    assert contract["reward"] == REWARD and len(contract["sha256"]) == 64
    receipt = json.loads((args.out / "cache-receipt.json").read_text())
    assert receipt["replayed_calls"] == 0 and receipt["paid_calls"] == len(
        router.extract_prompts
    ) + len(router.match_prompts)
    assert router.closed
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["calls"] and all(call["status"] == "settled" for call in ledger["calls"])


@pytest.mark.asyncio
async def test_a_resumed_group_pays_for_nothing_already_graded(tmp_path, corpus, monkeypatch):
    from grpo import judge_worker

    samples = [sample("found"), sample("nothing"), sample("found"), sample("nothing")]
    first = FakeRouter(REPORTS)
    patch_router(monkeypatch, first)
    args = write_args(tmp_path, plan_for(tmp_path, corpus), samples)
    await judge_worker.run(args)
    second = FakeRouter(REPORTS)
    patch_router(monkeypatch, second)
    args.out = tmp_path / "grading-again"
    await judge_worker.run(args)
    assert second.extract_prompts == [] and second.match_prompts == []
    assert json.loads((args.out / "rewards.json").read_text()) == [0.5, 0.0, 0.5, 0.0]
    assert json.loads((args.out / "cache-receipt.json").read_text())["paid_calls"] == 0


@pytest.mark.asyncio
async def test_a_failed_extraction_is_regraded_from_the_cache_without_paying_twice(
    tmp_path, corpus, monkeypatch
):
    from grpo import judge_worker

    router = FakeRouter(REPORTS, failures=1)
    patch_router(monkeypatch, router)
    samples = [sample("found"), sample("nothing"), sample("found"), sample("nothing")]
    args = write_args(tmp_path, plan_for(tmp_path, corpus), samples)
    await judge_worker.run(args)
    assert json.loads((args.out / "rewards.json").read_text()) == [0.5, 0.0, 0.5, 0.0]
    assert sorted(p.name for p in args.out.glob("attempt-*")) == ["attempt-1", "attempt-2"]
    receipt = json.loads((args.out / "cache-receipt.json").read_text())
    assert receipt["replayed_calls"] >= 3


@pytest.mark.asyncio
async def test_a_group_that_stays_unknown_without_a_retry_is_withheld(
    tmp_path, corpus, monkeypatch
):
    from grpo import judge_worker
    from grpo.reward import GroupWithheld

    router = FakeRouter(REPORTS, failures=99)
    patch_router(monkeypatch, router)
    samples = [sample("found")] * 4
    args = write_args(tmp_path, plan_for(tmp_path, corpus, judge_retry_limit=0), samples)
    with pytest.raises(GroupWithheld):
        await judge_worker.run(args)
    assert (args.out / "attempt-1/grades.json").exists()
    assert not (args.out / "attempt-2").exists()
    assert router.closed


@pytest.mark.asyncio
async def test_samples_from_another_row_are_refused_before_any_judge_request(
    tmp_path, corpus, monkeypatch
):
    from grpo import judge_worker

    router = FakeRouter(REPORTS)
    patch_router(monkeypatch, router)
    samples = [sample("found", row_id="another/group")] * 4
    args = write_args(tmp_path, plan_for(tmp_path, corpus), samples)
    with pytest.raises(ValueError, match="selected row"):
        await judge_worker.run(args)
    assert router.extract_prompts == [] and router.match_prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"reward": {**REWARD, "volume_allowance": None}}, "volume_allowance"),
        ({"benchmark_judge": {"model": "other"}}, "benchmark judge"),
    ],
)
async def test_an_unreviewed_reward_or_judge_stops_before_any_credential_or_router(
    tmp_path, corpus, monkeypatch, change, message
):
    from grpo import judge_worker

    async def unexpected(*_args, **_kwargs):
        pytest.fail("Must stop before router initialization")

    monkeypatch.setattr(judge_worker.RecordedJudgeRouter, "create", unexpected)
    args = write_args(tmp_path, plan_for(tmp_path, corpus, **change), [sample("found")] * 4)
    with pytest.raises(ValueError, match=message):
        await judge_worker.run(args)
    assert not args.out.exists()


@pytest.mark.asyncio
async def test_missing_dotenv_credentials_stop_before_training_grading(
    tmp_path, corpus, monkeypatch
):
    from grpo import judge_worker

    async def unexpected(*_args, **_kwargs):
        pytest.fail("Missing gateway credentials must stop before router initialization")

    monkeypatch.setattr(judge_worker.RecordedJudgeRouter, "create", unexpected)
    plan = plan_for(tmp_path, corpus, judge_env_file=str(tmp_path / "missing.env"))
    args = write_args(tmp_path, plan, [sample("found")] * 4)
    with pytest.raises(ValueError, match="gateway credential file"):
        await judge_worker.run(args)
    assert not args.out.exists()
