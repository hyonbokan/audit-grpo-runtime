"""Free-text benchmark configs and suite project results shared by the tests."""

import json
from pathlib import Path

from benchmark_free_text import PINNED_KEYS

ROOT = Path(__file__).resolve().parents[1]


def project(name, known, base, trained, label="checkpoint-2"):
    """A finished project whose comparison holds four graded draws per arm.

    `base` and `trained` are (exact, partial, findings) counts for every draw of that arm.
    """

    def draws(counts):
        exact, partial, findings = counts
        return [
            {
                "status": "graded",
                "known_gt_count": known,
                "exact_matches": exact,
                "partial_matches": partial,
                "findings_count": findings,
                "unmatched_findings": findings - exact - partial,
                "verdicts": [],
            }
            for _ in range(4)
        ]

    return {
        "name": name,
        "known_gt_count": known,
        "comparison": {
            "output_format": "free_text",
            "draws": {"base": draws(base), label: draws(trained)},
        },
    }


def free_text_configuration(row_id="solidity/munchables", *, extra_verification=False):
    """A reviewed base-only, xhigh-effort, four-draw free-text benchmark config."""
    scope = json.loads((ROOT / "benchmark_scope.json").read_text())
    prompt_tokens = 30000
    config = {key: scope[key] for key in PINNED_KEYS}
    config.update(
        output_format="free_text",
        row_id=row_id,
        extra_verification=extra_verification,
        run_id="test",
        assets="/assets",
        manifest_sha256="a" * 64,
        model_path="/model",
        prompt_token_count=prompt_tokens,
        absolute_deadline="2026-10-23T00:00:00+00:00",
        seconds_limit=43200,
        judge_seconds_limit=7200,
        judge_operation_cap_usd=40,
        judge_cap_usd=200,
        judge_ledger="/results/ledger.json",
        grading_concurrency=4,
        overlap_grading=False,
        template_options=dict(scope["template_options"], reasoning_effort="xhigh"),
        sampling=dict(
            scope["sampling"], max_tokens=scope["engine"]["max_model_len"] - prompt_tokens
        ),
        engine=dict(scope["engine"], tensor_parallel_size=2),
        cases=[{"label": "base", "initial_step": 0}],
        generation_order=["base"],
    )
    return config


def trained_free_text_configuration(row_id="solidity/munchables", step=7):
    """A reviewed trained-arm free-text config compared against a frozen base run."""
    config = free_text_configuration(row_id)
    case = {
        "label": f"checkpoint-{step}",
        "initial_step": step,
        "checkpoint": f"/results/cp{step}",
        "initial_adapter_sha256": "d" * 64,
    }
    config.update(
        cases=[case],
        generation_order=[case["label"]],
        merged_model_path="/workspace/cache/merged",
        base_reference={
            "run_root": f"/results/base-{row_id.split('/', 1)[1]}",
            "comparison_sha256": "c" * 64,
        },
    )
    return config
