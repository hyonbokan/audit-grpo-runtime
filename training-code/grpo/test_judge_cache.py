"""Completed judge responses are replayed exactly once per attempt and never repaid."""

import json

import pytest

from grpo.judge_cache import CachedJudge


class Paid:
    def __init__(self):
        self.calls = 0

    async def __call__(self, request):
        self.calls += 1
        return {"answer": self.calls, "echo": request["payload"]}


REQUEST = {"method": "sgr_match", "payload": {"x": 1}}


@pytest.mark.asyncio
async def test_retry_attempt_and_new_process_replay_without_paying(tmp_path):
    paid = Paid()
    judge = CachedJudge(paid, tmp_path)
    first = await judge(REQUEST)
    judge.new_attempt()
    assert await judge(REQUEST) == first
    # A resumed cycle starts a new process with the same cycle-local cache.
    resumed = CachedJudge(paid, tmp_path)
    assert await resumed(REQUEST) == first
    assert paid.calls == 1
    assert (judge.paid_calls, judge.replayed, resumed.replayed) == (1, 1, 1)


@pytest.mark.asyncio
async def test_identical_requests_in_one_attempt_are_paid_independently(tmp_path):
    paid = Paid()
    judge = CachedJudge(paid, tmp_path)
    responses = [await judge(REQUEST), await judge(REQUEST)]
    assert [r["answer"] for r in responses] == [1, 2]
    judge.new_attempt()
    assert [await judge(REQUEST), await judge(REQUEST)] == responses
    assert paid.calls == 2


@pytest.mark.asyncio
async def test_failed_calls_are_not_cached(tmp_path):
    class Flaky(Paid):
        async def __call__(self, request):
            if not self.calls:
                self.calls += 1
                raise TimeoutError("dropped")
            return await super().__call__(request)

    paid = Flaky()
    judge = CachedJudge(paid, tmp_path)
    with pytest.raises(TimeoutError):
        await judge(REQUEST)
    judge.new_attempt()
    assert (await judge(REQUEST))["answer"] == 2
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_tampered_cache_entry_is_rejected(tmp_path):
    judge = CachedJudge(Paid(), tmp_path)
    await judge(REQUEST)
    [entry] = tmp_path.glob("*.json")
    saved = json.loads(entry.read_text())
    saved["request"]["payload"] = {"x": 2}
    entry.write_text(json.dumps(saved))
    judge.new_attempt()
    with pytest.raises(ValueError, match="differs"):
        await judge(REQUEST)
