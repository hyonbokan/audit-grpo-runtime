"""Native benchmark grading in the existing credential-bearing Python 3.14 worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal

from llm_router.types import ThinkingEffort
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr
from scripts.finetuning.tools.pilot_transport import CURRENT_JOB, RecordedJudgeRouter

import benchmark_free_text as free_text
from benchmark_contract import grading_input_hashes, parse_completion
from benchmark_evaluation import validate_benchmark_config
from benchmark_matching import evaluate_findings, judge_vote
from grpo.budget import JudgeBudget, response_token_usage
from grpo.judge_environment import judge_router_config
from grpo.samples import write_new


class MatchingVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_match: StrictBool
    is_partial_match: StrictBool
    explanation: StrictStr
    severity_from_junior_auditor: StrictStr
    severity_from_truth: StrictStr
    index_of_finding_from_junior_auditor: float = Field(allow_inf_nan=False)


class ExtractedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issue: StrictStr
    description: StrictStr
    severity: Literal[free_text.SEVERITIES]
    contracts: list[StrictStr]


class ExtractedReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[ExtractedFinding]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if not args.run_paid:
        raise ValueError("Benchmark judge requires authorized --run-paid")
    asyncio.run(grade_benchmark(args.config, args.samples, args.out, args.replay))
    return 0


async def grade_benchmark(
    config_path: Path, sample_path: Path, out: Path, replay_path: Path | None = None
) -> None:
    config = json.loads(config_path.read_text())
    validate_benchmark_config(config)
    router_config = judge_router_config(config, semaphore_limit=4)
    truth_data = config_path.with_name("private-truth.json").read_bytes()
    template_data = config_path.with_name("judge-prompt.txt").read_bytes()
    sample_data = sample_path.read_bytes()
    truth = json.loads(truth_data)
    template = template_data.decode("utf-8")
    samples = json.loads(sample_data)
    if len(samples) != config["sampling"]["n"]:
        raise ValueError("Declared recorded benchmark draws required")
    out.mkdir(parents=True, exist_ok=False)
    budget = JudgeBudget(
        Path(config["judge_ledger"]),
        cap=config["judge_cap_usd"],
        operation_id=config["run_id"],
        operation_cap=config["judge_operation_cap_usd"],
        model="gpt-6-luna",
    )
    router = await RecordedJudgeRouter.create(router_config)
    router.event_dir = out / "events"
    results = []
    started = time.monotonic()
    try:
        judge = BenchmarkJudge(router, out, budget, replay_path=replay_path)
        semaphore = asyncio.Semaphore(config.get("grading_concurrency", 1))
        # Older configs predate output_format; their answers are native JSON.
        free = config.get("output_format", "json") == "free_text"
        if free:
            (out / "extraction").mkdir()
            extractor = BenchmarkJudge(
                router,
                out / "extraction",
                budget,
                ExtractedReport,
                replay_path=replay_path / "extraction" if replay_path else None,
            )

        async def grade_draw(index, sample):
            async with semaphore:
                if free:
                    result = await free_text.grade_draw(
                        sample, truth, template, extractor, lambda prompt: judge_vote(prompt, judge)
                    )
                    return record_draw(index, sample, result)
                try:
                    findings = parse_completion(
                        sample["final_text"],
                        sample["finish_reason"],
                        output_format=config.get("output_format", "json"),
                    )
                except ValueError as error:
                    result = {
                        "status": "truncated" if sample["finish_reason"] == "length" else "invalid",
                        "error": str(error),
                        "known_gt_count": len(truth),
                        "exact_coverage": None,
                        "coverage_including_partial": None,
                    }
                else:
                    result = await evaluate_findings(
                        findings, truth, template, lambda prompt: judge_vote(prompt, judge)
                    )
                return record_draw(index, sample, result)

        def record_draw(index, sample, result):
            result.update(
                draw_index=index,
                completion_tokens=len(sample["completion_ids"]),
                finish_reason=sample["finish_reason"],
            )
            write_new(out / f"draw-{index}.json", result)
            print(
                f"draw {index}: {result['status']}; exact coverage={result['exact_coverage']}",
                flush=True,
            )
            return result

        # Let every draw finish and save its draw-N.json before reporting a failure,
        # so a rerun with --replay reuses all completed paid responses.
        results = await asyncio.gather(
            *(grade_draw(index, sample) for index, sample in enumerate(samples)),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise failures[0]
        write_new(
            out / "metrics.json",
            {
                "draws": results,
                "judge": config["benchmark_judge"],
                "input_sha256": grading_input_hashes(sample_data, truth_data, template_data),
                "valid_response_rate": sum(
                    row["status"] in ("graded", "unknown") for row in results
                )
                / len(samples),
                "grading_seconds": time.monotonic() - started,
                "judge_calls": judge.counter,
                "reused_paid_responses": judge.reused,
            },
        )
    finally:
        await router.close()


class BenchmarkJudge:
    def __init__(
        self,
        router: Any,
        out: Path,
        budget: JudgeBudget,
        response_model: type[BaseModel] = MatchingVerdict,
        *,
        replay_path: Path | None = None,
    ):
        self.router, self.out, self.budget = router, out, budget
        self.response_model = response_model
        self.counter = 0
        self.reused = 0
        self.replay = {}
        if replay_path is not None:
            for folder in sorted(replay_path.glob("call-*")):
                if not (folder / "raw.json").exists():
                    continue
                raw = json.loads((folder / "raw.json").read_text())
                if raw["status"] != "completed":
                    # Interrupted calls are evidence, not votes; keep other saved votes usable.
                    continue
                request = json.loads((folder / "request.json").read_text())
                if (
                    request["model"] != "gpt-6-luna"
                    or request["api_effort"] != "xhigh"
                    or request["schema"] != response_model.model_json_schema()
                ):
                    raise ValueError("Replay response contract differs")
                messages = [item for item in raw["output"] if item["type"] == "message"]
                texts = [
                    item["text"]
                    for message in messages
                    for item in message["content"]
                    if item["type"] == "output_text"
                ]
                verdict = response_model.model_validate_json("".join(texts)).model_dump()
                self.replay.setdefault(request["prompt"], []).append((folder, verdict))

    async def __call__(self, prompt: str) -> dict[str, Any]:
        self.counter += 1
        folder = self.out / f"call-{self.counter:04d}"
        folder.mkdir()
        schema = self.response_model.model_json_schema()
        write_new(
            folder / "request.json",
            {
                "model": "gpt-6-luna",
                "effort": "max",
                "api_effort": "xhigh",
                "prompt": prompt,
                "schema": schema,
                "service_tier": "account_default",
                "timeout_seconds": 180,
                "retry_limit": 0,
            },
        )
        if self.replay.get(prompt):
            source, verdict = self.replay[prompt].pop(0)
            self.reused += 1
            write_new(
                folder / "replay.json",
                {
                    "source": str(source),
                    "raw_sha256": hashlib.sha256((source / "raw.json").read_bytes()).hexdigest(),
                    "verdict": verdict,
                },
            )
            return verdict
        token = CURRENT_JOB.set(hashlib.sha256(str(folder).encode()).hexdigest())
        started = time.monotonic()
        reservation = None
        settled = False
        try:
            bound = len(prompt.encode()) + len(json.dumps(schema).encode()) + 8192
            reservation = self.budget.reserve(bound)
            write_new(folder / "reservation.json", {"id": reservation})
            # Native UI "max" maps to API "xhigh". Runtime MAX would send a different effort.
            result = await self.router.complete(
                "gpt-6-luna",
                [{"role": "user", "content": prompt}],
                response_model=self.response_model,
                thinking_effort=ThinkingEffort.XHIGH,
                max_tokens=None,
                timeout=180,
                max_retries=0,
            )
            raw = result.raw_response
            (folder / "raw.json").write_text(raw.model_dump_json(indent=2, warnings=False))
            usage = {
                **(response_token_usage(raw) or {}),
                "response_id": raw.id,
                "resolved_model": raw.model,
                "status": raw.status,
                "seconds": time.monotonic() - started,
                "estimate_basis": "Luna Standard; conservative ledger includes uplifts",
            }
            write_new(folder / "usage.json", usage)
            self.budget.settle(reservation, usage if raw.usage is not None else None)
            settled = True
            return result.content.model_dump()
        except Exception as error:
            if reservation is not None and not settled:
                self.budget.settle(reservation, None)
            write_new(
                folder / "error.json",
                {
                    "type": type(error).__name__,
                    "error": str(error),
                    "seconds": time.monotonic() - started,
                },
            )
            raw = getattr(error, "raw_response", None)
            if raw is not None:
                (folder / "raw.json").write_text(raw.model_dump_json(indent=2, warnings=False))
            raise
        finally:
            CURRENT_JOB.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
