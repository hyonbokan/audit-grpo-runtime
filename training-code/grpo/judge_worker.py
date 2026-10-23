"""Python 3.14 reward worker: the evaluation's judge path over one rollout group."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from scripts.finetuning.tools.pilot_transport import RecordedJudgeRouter

import benchmark_free_text as free_text
import free_text_corpus
from benchmark_judge import BenchmarkJudge, ExtractedReport
from benchmark_matching import judge_vote
from grpo import reward as free_text_reward
from grpo.budget import JudgeBudget
from grpo.judge_cache import CachedJudge
from grpo.judge_environment import judge_router_config
from grpo.measurement import group_metrics
from grpo.samples import write_new
from runtime_io import read_json

# One router slot per draw: the group's four draws are graded together.
GRADING_CONCURRENCY = 4
JUDGE_MODEL = "gpt-6-luna"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--split", choices=free_text_corpus.SPLITS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, required=True, help="Cycle-local completed-response cache"
    )
    args = parser.parse_args(argv)
    if not args.run_paid:
        raise ValueError("Paid worker requires explicit --run-paid from authorized trainer")
    asyncio.run(run(args))
    return 0


async def run(args):
    plan = json.loads(args.plan.read_text())
    reward = plan["reward"]
    free_text_reward.validate_reward(reward)
    if plan["benchmark_judge"] != read_json(free_text.SCOPE_PATH)["benchmark_judge"]:
        raise ValueError("Worker judge differs from the reviewed benchmark judge")
    router_config = judge_router_config(plan, semaphore_limit=GRADING_CONCURRENCY)
    corpus = Path(plan["dataset_workspace"])
    _, private = free_text_reward.load_split(corpus, args.split)
    template = (corpus / free_text_corpus.JUDGE_TEMPLATE).read_text()
    budget = JudgeBudget(
        Path(plan["access"]["local_judge_budget"]["path"]),
        cap=plan["access"]["judge_usd_cap"],
        operation_id=plan["judge_operation"]["id"],
        operation_cap=plan["judge_operation"]["cap_usd"],
        model=JUDGE_MODEL,
    )
    samples = json.loads(args.samples.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    router = await RecordedJudgeRouter.create(router_config)
    router.event_dir = args.out / "events"
    try:
        cache, extract, match = cached_judges(router, args.out, budget, args.cache)
        rewards = await grade_selected_group(
            samples, private, reward, template, extract, match, cache.new_attempt, args.out, plan
        )
        write_new(args.out / "cache-receipt.json", cache.receipt())
        write_new(args.out / "rewards.json", rewards)
        write_new(
            args.out / "reward-contract.json",
            {
                "reward": reward,
                "judge": plan["benchmark_judge"],
                "sha256": free_text_reward.contract_sha256(reward, template),
            },
        )
        grades = json.loads(sorted(args.out.glob("attempt-*/grades.json"))[-1].read_text())
        write_new(args.out / "metrics.json", group_metrics(private[samples[0]["row_id"]], grades))
    finally:
        await router.close()


async def grade_selected_group(
    samples, private, reward, template, extract, match, new_attempt, out, plan
):
    selected = plan["selected_row_id"]
    if selected not in private:
        raise ValueError(f"Selected row {selected!r} is not in the split")
    if any(sample.get("row_id") != selected for sample in samples):
        raise ValueError("Grading samples differ from selected row")
    return await free_text_reward.grade_group(
        samples,
        private[selected],
        reward,
        template,
        extract,
        match,
        new_attempt,
        out,
        num_generations=len(samples),
        max_retries=plan["judge_retry_limit"],
    )


# --- Judges behind the cycle's response cache


def cached_judges(router: Any, out: Path, budget: JudgeBudget, directory: Path):
    """The extraction and voted-matching judges, each paid call kept for a regrade or a resume."""
    (out / "extraction").mkdir()
    paid = {
        "extract": BenchmarkJudge(router, out / "extraction", budget, ExtractedReport),
        "match": BenchmarkJudge(router, out, budget),
    }

    async def paid_call(request: dict[str, Any]) -> dict[str, Any]:
        return await paid[request["kind"]](request["prompt"])

    cache = CachedJudge(paid_call, directory)

    async def extract(prompt: str) -> dict[str, Any]:
        return await cache({"kind": "extract", "prompt": prompt})

    async def match_once(prompt: str) -> dict[str, Any]:
        return await cache({"kind": "match", "prompt": prompt})

    async def match(prompt: str) -> dict[str, Any]:
        return await judge_vote(prompt, match_once)

    return cache, extract, match


if __name__ == "__main__":
    raise SystemExit(main())
