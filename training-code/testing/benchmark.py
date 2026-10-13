"""Native benchmark evaluation configs and suite project results shared by the tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    scope = json.loads((ROOT / "evaluation_scope.json").read_text())
    scope.update(
        evaluation_kind="native_solidity_benchmark",
        engine=dict(scope["engine"], max_model_len=36864),
        row_id="solidity/munchables",
        model_path="/model",
        run_id="test",
        assets="/assets",
        manifest_sha256="a" * 64,
        template_options={
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": True,
            "reasoning_effort": "low",
            "preserve_thinking": True,
        },
        benchmark_judge={
            "model": "gpt-6-luna",
            "effort": "max",
            "api_effort": "xhigh",
            "iterations": 3,
            "batch_size": 30,
            "retry_limit": 0,
        },
        cases=[
            {"label": "base", "initial_step": 0},
            {
                "label": "checkpoint-2",
                "initial_step": 2,
                "checkpoint": "/checkpoint",
                "initial_adapter_sha256": "b" * 64,
            },
        ],
    )
    return scope


def native_context_configuration():
    config = configuration()
    config.update(
        benchmark_suite="solidity1_markdown",
        evaluation_profile="full_native_context",
        output_format="markdown",
        merged_model_path="/merged",
        prompt_token_count=35093,
        absolute_deadline="2026-10-10T12:30:00+00:00",
        seconds_limit=21600,
        judge_seconds_limit=1800,
        judge_operation_cap_usd=150,
    )
    config["sampling"].update(n=1, max_tokens=227051)
    for key in ("lora_dtype", "max_loras", "max_lora_rank"):
        config["engine"].pop(key)
    config["engine"].update(
        max_model_len=262144, enable_lora=False, max_num_seqs=1, gpu_memory_utilization=0.90
    )
    return config


def scheduled_configuration():
    config = native_context_configuration()
    config.update(
        weekend_schedule_id="weekend-grpo-20261009",
        absolute_deadline="2026-10-13T00:00:00+00:00",
        seconds_limit=32400,
        judge_operation_cap_usd=30,
    )
    config["template_options"]["reasoning_effort"] = "xhigh"
    config["cases"][-1].update(label="checkpoint-5", initial_step=5)
    return config


def project(name, known, base, trained):
    def arm(label, counts):
        exact, partial, findings = counts
        return {
            "label": label,
            "draws": [
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
            ],
        }

    return {
        "name": name,
        "known_gt_count": known,
        "comparison": {
            "strict_json_validity": {"base": 1, "checkpoint-2": 1},
            "arms": [arm("base", base), arm("checkpoint-2", trained)],
        },
    }
