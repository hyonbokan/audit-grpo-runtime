"""Separate, symmetric rescue protocol: re-format what each draw already wrote.

A healer model reads one raw output and rewrites its identified issues into the strict
Markdown report format. It sees no source code or labels and may not add findings. Every
healed finding must cite a verbatim quote from the raw output; findings whose quote is
absent are flagged (not dropped) so the report can state how much was invented.
The strict samples are never modified; healed samples are graded as a separate arm.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from benchmark_contract import parse_markdown

QUOTE_LINE = re.compile(r'^[ \t]*Source quote:[ \t]*"(.*)"[ \t]*$', re.M)

PROMPT = """You are a formatting assistant. Below is the complete raw output of a smart
contract security audit. It may be unfinished, may contain exploratory reasoning, or may
already be a report. Rewrite the vulnerabilities that the text itself identifies as
issues into the report format below.

Rules:
- Do not add, invent or judge findings. Do not use outside knowledge of the code.
- Skip ideas the text itself rules out as safe or not exploitable.
- Keep the text's own description of root cause, conditions and impact. Merge duplicates.
- Use exact contract paths as written in the text; if none is given, use the file name
  mentioned for the function.
- End every finding with one line `Source quote: "..."` holding a short verbatim excerpt
  (copied exactly, no ellipsis) from the raw output that supports it.
- If the text identifies no vulnerability, return exactly: No vulnerabilities found.

Report format, one section per finding:

## Finding 1: Issue title
Severity: High | Medium | Low | Info (choose one)
Contracts: `relative/path.sol`, `another/path.sol`

Root cause, relevant functions, conditions and impact.
Source quote: "verbatim excerpt"

Return the report in the `report_markdown` field.

=== RAW AUDIT OUTPUT START ===
{raw}
=== RAW AUDIT OUTPUT END ===
"""


def healing_prompt(raw_text: str) -> str:
    return PROMPT.format(raw=raw_text)


def squash(text: str) -> str:
    return " ".join(text.split())


async def heal_sample(sample: dict[str, Any], healer: Any) -> dict[str, Any]:
    raw = sample["raw_text"]
    markdown = (await healer(healing_prompt(raw)))["report_markdown"].strip()
    findings = parse_markdown(markdown)
    haystack = squash(raw)
    untraceable = []
    for finding in findings:
        quotes = QUOTE_LINE.findall(finding["description"])
        if not quotes or any(squash(quote) not in haystack for quote in quotes):
            untraceable.append(finding["issue"])
    cleaned = QUOTE_LINE.sub("", markdown).strip()
    parse_markdown(cleaned)
    return {
        **sample,
        "final_text": cleaned,
        "finish_reason": "stop",
        "healing": {
            "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "findings": len(findings),
            "untraceable_findings": untraceable,
        },
    }


async def heal_samples(source: Path, out: Path, healer: Any) -> list[dict[str, Any]]:
    samples = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=False)
    healed = list(await asyncio.gather(*(heal_sample(sample, healer) for sample in samples)))
    (out / "samples.json").write_text(json.dumps(healed))
    return healed


async def run(config_path: Path, source: Path, out: Path) -> None:
    from pydantic import BaseModel, ConfigDict, StrictStr
    from scripts.finetuning.tools.pilot_transport import RecordedJudgeRouter

    from benchmark_evaluation import validate_benchmark_config
    from benchmark_judge import BenchmarkJudge
    from grpo.budget import JudgeBudget
    from grpo.judge_environment import judge_router_config

    class HealedReport(BaseModel):
        model_config = ConfigDict(extra="forbid")
        report_markdown: StrictStr

    config = json.loads(config_path.read_text())
    validate_benchmark_config(config)
    budget = JudgeBudget(
        Path(config["judge_ledger"]),
        cap=config["judge_cap_usd"],
        operation_id=config["run_id"],
        operation_cap=config["judge_operation_cap_usd"],
        model="gpt-6-luna",
    )
    router = await RecordedJudgeRouter.create(judge_router_config(config, semaphore_limit=4))
    router.event_dir = out.with_name(out.name + "-events")
    try:
        healer = BenchmarkJudge(router, out.with_name(out.name + "-calls"), budget, HealedReport)
        await heal_samples(source, out, healer)
    finally:
        await router.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-paid", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_paid:
        raise ValueError("Healing requires authorized --run-paid")
    args.out.with_name(args.out.name + "-calls").mkdir(parents=True, exist_ok=False)
    asyncio.run(run(args.config, args.samples, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
