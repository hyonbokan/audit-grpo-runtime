"""Python 3.14 reward worker using the existing backend router, never the historical B CLI."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from llm_router.errors import RateLimitError, TransientProviderError
from llm_router.types import ThinkingEffort
from scripts.finetuning.tools.pilot_transport import CURRENT_JOB, RecordedJudgeRouter

from grpo.budget import JudgeBudget, response_token_usage
from grpo.judge_cache import CachedJudge
from grpo.judge_environment import judge_router_config
from grpo.measurement import group_metrics
from grpo.policy_prompt import judging_contract
from grpo.reward import grade_group, load_split, response_schema
from grpo.samples import write_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "evaluation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, help="Cycle-local completed-response cache shared across retries"
    )
    args = parser.parse_args(argv)
    if not args.run_paid:
        raise ValueError("Paid worker requires explicit --run-paid from authorized trainer")
    asyncio.run(run(args))
    return 0


async def run(args):
    plan = json.loads(args.plan.read_text())
    if plan.get("judge") != judging_contract(args.workspace):
        raise ValueError("Worker judge contract differs from live plan")
    router_config = judge_router_config(plan, semaphore_limit=1)
    budget = JudgeBudget(
        Path(plan["access"]["local_judge_budget"]["path"]),
        cap=plan["access"]["judge_usd_cap"],
        operation_id=(plan.get("judge_operation") or {}).get("id"),
        operation_cap=(plan.get("judge_operation") or {}).get("cap_usd"),
    )
    _, private = load_split(Path(plan.get("dataset_workspace", args.workspace)), args.split)
    samples = json.loads(args.samples.read_text())
    contract = json.loads((args.workspace / "configs/reward_contract.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    router = await RecordedJudgeRouter.create(router_config)
    router.event_dir = args.out / "events"
    try:
        judge = RouterJudge(
            router,
            args.out,
            budget=budget,
            retries=plan.get("judge_call_retries", DEFAULT_CALL_RETRIES),
        )
        if args.cache:
            judge = CachedJudge(judge, args.cache)
        rewards = await grade_selected_group(
            samples,
            private,
            contract,
            judge,
            args.out,
            plan,
        )
        if args.cache:
            write_new(args.out / "cache-receipt.json", judge.receipt())
        write_new(args.out / "rewards.json", rewards)
        attempts = sorted(args.out.glob("attempt-*/grades.json"))
        grades = json.loads(attempts[-1].read_text())
        identity = samples[0]["row_id"]
        targets = [
            target["id"] for target in private[identity]["scorer_metadata"]["eligible_targets"]
        ]
        write_new(args.out / "metrics.json", group_metrics(identity, targets, grades, samples))
    finally:
        await router.close()


async def grade_selected_group(samples, private, contract, judge, out, plan):
    selected = plan.get("selected_row_id")
    if selected and any(sample.get("row_id") != selected for sample in samples):
        raise ValueError("Grading samples differ from selected row")
    if selected and not private.get(selected, {}).get("scorer_metadata", {}).get(
        "eligible_targets"
    ):
        raise ValueError("Selected row has no eligible private targets")
    return await grade_group(
        samples,
        private,
        contract,
        judge,
        out,
        num_generations=4,
        max_retries=plan.get("judge_retry_limit", 1),
    )


# Transport failures (dropped connections, 5xx, rate limits) are retried with backoff.
# Every attempt keeps its own call folder and budget reservation; anything else
# (schema, budget, programming errors) fails immediately.
DEFAULT_CALL_RETRIES = 3
TRANSIENT_ERRORS = (TransientProviderError, RateLimitError)


class RouterJudge:
    def __init__(self, router, out, *, budget=None, retries=0, backoff_seconds=30.0):
        if type(retries) is not int or not 0 <= retries <= 5:
            raise ValueError("Judge call retries must be an integer from 0 to 5")
        self.router, self.out = router, out
        self.counter = 0
        self.budget = budget
        self.retries, self.backoff_seconds = retries, backoff_seconds

    async def __call__(self, request):
        for attempt in range(self.retries + 1):
            try:
                return await self._call_once(request)
            except TRANSIENT_ERRORS as exc:
                if attempt == self.retries:
                    raise
                delay = self.backoff_seconds * 2**attempt
                delay = max(delay, getattr(exc, "retry_after", None) or 0)
                print(f"Transient judge failure ({type(exc).__name__}); retrying in {delay:.0f}s")
                await asyncio.sleep(delay)

    async def _call_once(self, request):
        if request["model"] != "gpt-6.1-sol" or request["reasoning_effort"] not in (
            "medium",
            "high",
        ):
            raise ValueError("Worker requires selected Sol medium/high judge")
        self.counter += 1
        folder = self.out / f"call-{self.counter:04d}"
        folder.mkdir()
        schema = response_schema(request["method"])
        write_new(folder / "request.json", {**request, "schema": schema.model_json_schema()})
        bound = (
            len(json.dumps({**request, "schema": schema.model_json_schema()}).encode("utf-8"))
            + 8192
        )
        reservation = self.budget.reserve(bound) if self.budget else None
        if self.budget:
            write_new(
                folder / "budget-reservation.json",
                {
                    "reservation_id": reservation,
                    "ledger": str(self.budget.path),
                },
            )
        settled = False
        job = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()
        token = CURRENT_JOB.set(job)
        started = time.monotonic()
        try:
            result = await self.router.complete(
                request["model"],
                [{"role": "user", "content": json.dumps(request["payload"], ensure_ascii=False)}],
                system_prompt=request["system"],
                response_model=schema,
                thinking_effort=ThinkingEffort(request["reasoning_effort"]),
                max_tokens=None,
                timeout=1800,
                max_retries=0,
            )
            raw = result.raw_response
            (folder / "raw.json").write_text(raw.model_dump_json(indent=2, warnings=False))
            usage = {
                "input_tokens": result.input_tokens,
                "cached_input_tokens": result.cache_read_tokens,
                "output_tokens": result.output_tokens,
                **(response_token_usage(raw) or {}),
                "resolved_model": raw.model,
                "response_id": raw.id,
                "status": raw.status,
                "seconds": time.monotonic() - started,
            }
            write_new(folder / "usage.json", usage)
            if self.budget:
                self.budget.settle(reservation, usage if getattr(raw, "usage", None) else None)
                settled = True
            return result.content.model_dump()
        except Exception as exc:
            if self.budget and not settled:
                self.budget.settle(reservation, None)
            write_new(
                folder / "error.json",
                {
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "usage": (getattr(exc, "details", None) or {}).get("usage"),
                    "seconds": time.monotonic() - started,
                },
            )
            if getattr(exc, "raw_response", None) is not None:
                (folder / "raw.json").write_text(
                    exc.raw_response.model_dump_json(indent=2, warnings=False)
                )
            raise
        finally:
            CURRENT_JOB.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
