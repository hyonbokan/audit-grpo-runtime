"""Reviewed, source-pinned corpus admission and split isolation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal


def admit(candidate: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    files = candidate["source_files"]
    paths = {f["path"] for f in files}
    if len(paths) != len(files) or not files:
        raise ValueError("Duplicate or empty source scope")
    if any(hashlib.sha256(f["content"].encode()).hexdigest() != f["sha256"] for f in files):
        raise ValueError("Source hash mismatch")
    if candidate["unresolved_imports"] or not set(candidate["scope_paths"]).issubset(paths):
        raise ValueError("Incomplete audited scope or dependencies")
    if not review["scope_verified"] or not review["scope_rationale"].strip():
        raise ValueError("Audited scope review did not pass")
    targets = {t["archive_id"]: t for t in candidate["targets"]}
    votes = {t["archive_id"]: t for t in review["targets"]}
    if len(votes) != len(review["targets"]) or set(votes) != set(targets):
        raise ValueError("Review does not join all candidate targets exactly")
    accepted = []
    for identity, target in targets.items():
        vote = votes[identity]
        if not vote["accepted"]:
            continue
        if vote["final_verdict"] not in ("HIGH", "MEDIUM"):
            raise ValueError("Accepted target has no final high/medium verdict")
        if candidate["revision"] not in target["reference_revisions"]:
            raise ValueError("Finding does not reference recovered immutable revision")
        if not target["contracts"] or not set(target["contracts"]).issubset(paths):
            raise ValueError("Finding references missing code")
        if not vote["rationale"].strip() or any(
            x not in ("summary", "documentation", "invariants") for x in vote["required_context"]
        ):
            raise ValueError("Missing target context review")
        record = target["archive_record"]
        accepted.append(
            {
                "id": f"archive:{identity}",
                "required_context": vote["required_context"],
                "finding": {
                    "issue": record["title"],
                    "description": record["content"],
                    "contracts": target["contracts"],
                    "archive_record": record,
                    "eligibility_review": {
                        "status": "accepted",
                        **vote,
                        "review_origin": "Sol medium structured source/report review; exact immutable-reference gate",
                    },
                },
            }
        )
    if not any(not t["required_context"] for t in accepted):
        raise ValueError("No eligible code-only targets after review")
    return {
        "row_id": candidate["row_id"],
        "repository": candidate["repository"],
        "revision": candidate["revision"],
        "group_id": "all-scope",
        "source_files": files,
        "complementary_context": {},
        "scorer_metadata": {"targets": accepted},
        "profile": "none",
    }


def application_fingerprint(content: str) -> str:
    tokens = re.findall(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
        r"[A-Za-z_$][\w$]*|[^\s]",
        content,
    )
    body = [t for t in tokens if not t.startswith(("//", "/*"))]
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def check_isolation(
    candidates: list[dict[str, Any]], *, vendor_fingerprints: set[str] | None = None
) -> None:
    owners: dict[tuple[str, str], str] = {}
    labels: dict[int, str] = {}
    for c in candidates:
        keys = [("repository", c["repository"].lower().removesuffix(".git"))]
        keys += [("family", f) for f in c["family_ids"]]
        for f in c["source_files"]:
            path = f["path"].lower()
            if not any(
                x in path
                for x in (
                    "node_modules/",
                    "openzeppelin/",
                    "@openzeppelin/",
                    "/lib/",
                    "lib/",
                    "solmate/",
                    "solady/",
                    "interfaces/",
                )
            ):
                fingerprint = application_fingerprint(f["content"])
                if fingerprint not in (vendor_fingerprints or set()):
                    keys.append(("application", f["sha256"]))
                    keys.append(("normalized_application", fingerprint))
        for key in keys:
            if key in owners and owners[key] != c["split"]:
                raise ValueError(f"Cross-split {key[0]} overlap: {key[1]}")
            owners[key] = c["split"]
        for t in c.get("targets", []):
            identity = t["archive_id"]
            if identity in labels and labels[identity] != c["split"]:
                raise ValueError("Cross-split finding overlap")
            labels[identity] = c["split"]


async def review_candidates(config: dict[str, Any], out: Path) -> None:
    # The judge environment owns router dependencies; CPU corpus checks do not import them.
    from pydantic import BaseModel
    from llm_router.types import ThinkingEffort
    from scripts.finetuning.tools.pilot_transport import RecordedJudgeRouter
    from grpo.judge_environment import judge_router_config
    from grpo.budget import JudgeBudget, response_token_usage

    class TargetReview(BaseModel):
        archive_id: int
        accepted: bool
        final_verdict: Literal["HIGH", "MEDIUM", "OTHER", "UNKNOWN"]
        required_context: list[Literal["summary", "documentation", "invariants"]]
        rationale: str

    class ScopeReview(BaseModel):
        scope_verified: bool
        scope_rationale: str
        targets: list[TargetReview]

    out.mkdir(parents=True, exist_ok=False)
    budget = JudgeBudget(
        Path(config["budget_path"]), cap=150, operation_id=config["run_id"], operation_cap=150
    )
    router = await RecordedJudgeRouter.create(judge_router_config(config, semaphore_limit=1))
    router.event_dir = out / "events"
    system = (
        "Review candidate admission for code-only vulnerability training. You are a curation reviewer, not the policy. "
        "Verify the supplied immutable repository revision, README/scope declaration and complete Solidity source scope. "
        "Do not accept partial scopes, findings in missing Rust/Go/non-Solidity code, or labels that concern another revision. "
        "For every supplied archive ID decide the FINAL published HIGH/MEDIUM verdict, defect identity, actual harmful behavior "
        "and whether it follows from visible code alone. Mark external deployment, economic, intended-behavior or frontend "
        "assumptions as requiring documentation/invariants; do not turn assumptions into code-only facts. "
        "Code-only means hypothetical attacker inputs and normal token behavior may be reasoned from code, but external "
        "protocol intentions or actual historical prices/balances cannot be assumed. Read final judge comments rather than "
        "intermediate downgrades. Reject mismatched reports, insufficient evidence and unresolved verdicts. "
        "Return every candidate archive ID exactly once. Give concrete function/path evidence and harm in each rationale. "
        "The archive is a sparse set of known labels; do not invent new IDs or require completeness of the label set. "
        "Scope verification is about completeness of the supplied audited code and dependencies."
    )
    try:
        candidates = [
            json.loads(line) for line in Path(config["candidates"]).read_text().splitlines()
        ]
        for c in candidates:
            folder = out / c["job_id"]
            folder.mkdir()
            request = {
                "model": "gpt-6.1-sol",
                "reasoning_effort": "medium",
                "system": system,
                "payload": c,
                "schema": ScopeReview.model_json_schema(),
                "max_retries": 0,
            }
            (folder / "request.json").write_text(json.dumps(request, ensure_ascii=False) + "\n")
            reservation = budget.reserve(len(json.dumps(request).encode()) + 8192)
            (folder / "budget-reservation.json").write_text(
                json.dumps({"reservation_id": reservation, "ledger": str(budget.path)})
            )
            settled = False
            try:
                result = await router.complete(
                    "gpt-6.1-sol",
                    [{"role": "user", "content": json.dumps(c, ensure_ascii=False)}],
                    system_prompt=system,
                    response_model=ScopeReview,
                    thinking_effort=ThinkingEffort("medium"),
                    max_tokens=None,
                    timeout=900,
                    max_retries=0,
                )
                raw = result.raw_response
                (folder / "raw.json").write_text(raw.model_dump_json(indent=2, warnings=False))
                usage = response_token_usage(raw)
                budget.settle(reservation, usage)
                settled = True
                (folder / "usage.json").write_text(json.dumps(usage))
                review = result.content.model_dump()
                (folder / "review.json").write_text(json.dumps(review, indent=2) + "\n")
                try:
                    row = admit(c, review)
                    (folder / "accepted-row.private.json").write_text(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                "job_id": c["job_id"],
                                "status": "accepted",
                                "targets": len(row["scorer_metadata"]["targets"]),
                            }
                        ),
                        flush=True,
                    )
                except ValueError as exc:
                    (folder / "withheld.json").write_text(json.dumps({"reason": str(exc)}))
                    print(
                        json.dumps(
                            {"job_id": c["job_id"], "status": "withheld", "reason": str(exc)}
                        ),
                        flush=True,
                    )
            except Exception as exc:
                if not settled:
                    budget.settle(reservation, None)
                (folder / "error.json").write_text(
                    json.dumps({"type": type(exc).__name__, "error": str(exc)})
                )
                raise
    finally:
        await router.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-paid", action="store_true")
    args = parser.parse_args()
    if not args.run_paid:
        raise ValueError("Curation requires explicit paid authorization")
    asyncio.run(review_candidates(json.loads(args.config.read_text()), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
