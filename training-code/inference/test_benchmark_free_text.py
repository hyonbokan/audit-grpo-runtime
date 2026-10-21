"""Free-text benchmark: no output format, faithful extraction, then the reviewed matcher."""

import asyncio
import json

import pytest

import benchmark_free_text as free_text
from testing.benchmark import free_text_configuration

SOURCES = "// File: A.sol\ncontract A { function f() external {} }"
TRUTH = [
    {
        "issue": "Reentrancy in f",
        "severity": "High",
        "description": "f sends ETH before updating state",
        "contracts": ["A.sol"],
    }
]


def test_prompt_asks_for_an_audit_without_any_output_format():
    prompt = free_text.audit_prompt(SOURCES)
    assert SOURCES in prompt
    for format_cue in ("JSON", "Markdown", "Severity:", "Finding 1", "Output format"):
        assert format_cue not in prompt


def test_base_only_xhigh_effort_four_draw_config_is_reviewed():
    free_text.validate_config(free_text_configuration())


def test_the_five_solidity_repos_are_the_evaluation_set():
    assert free_text.evaluation_row_ids() == [
        "solidity/munchables",
        "solidity/vultisig",
        "solidity/tigris",
        "solidity/backd",
        "solidity/phi",
    ]


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["template_options"].update(reasoning_effort="high"),
        lambda c: c.pop("native_adapter_precision"),
        lambda c: c.update(rollout_adapter_precision="float32"),
        lambda c: c["sampling"].update(n=1),
        lambda c: c["sampling"].update(max_tokens=1000),
        lambda c: c.update(row_id="solidity2/renft"),
        lambda c: c.update(row_id="solidity/unknown"),
        lambda c: c["engine"].update(tensor_parallel_size=3),
        lambda c: c.update(output_format="markdown"),
        lambda c: c.update(seconds_limit=10**9),
        lambda c: c["cases"].append({"label": "checkpoint-4", "initial_step": 4}),
        lambda c: c.update(cases=[{"label": "checkpoint-4", "initial_step": 4}]),
    ],
)
def test_unreviewed_free_text_settings_are_rejected(change):
    config = free_text_configuration()
    change(config)
    with pytest.raises(ValueError):
        free_text.validate_config(config)


def test_extra_verification_repos_need_the_explicit_flag():
    config = free_text_configuration(row_id="solidity2/renft", extra_verification=True)
    free_text.validate_config(config)


def test_a_trained_arm_is_compared_through_its_merged_weights():
    config = free_text_configuration()
    config["cases"].append(
        {
            "label": "checkpoint-4",
            "initial_step": 4,
            "checkpoint": "/checkpoint",
            "initial_adapter_sha256": "b" * 64,
        }
    )
    config["generation_order"] = ["base", "checkpoint-4"]
    with pytest.raises(ValueError, match="merged"):
        free_text.validate_config(config)
    free_text.validate_config(dict(config, merged_model_path="/merged"))


def test_extracted_findings_may_omit_severity_and_contracts():
    findings = [{"issue": "Reentrancy", "description": "...", "severity": "N/A", "contracts": []}]
    assert free_text.validate_extracted({"findings": findings}) == findings


@pytest.mark.parametrize(
    "report",
    [
        {"findings": [{"issue": "x", "description": "y", "severity": "Critical", "contracts": []}]},
        {"findings": [{"issue": "x", "description": "y", "severity": "Low"}]},
        {
            "findings": [
                {"issue": "x", "description": "y", "severity": "Low", "contracts": [], "z": 1}
            ]
        },
        {"vulnerabilities": []},
    ],
)
def test_malformed_extraction_is_rejected_not_scored(report):
    with pytest.raises(ValueError):
        free_text.validate_extracted(report)


def judges(extracted, verdict):
    calls = {"extract": [], "match": []}

    async def extract(prompt):
        calls["extract"].append(prompt)
        return {"findings": extracted}

    async def match(prompt):
        calls["match"].append(prompt)
        return dict(verdict)

    return calls, extract, match


MATCH = {
    "is_match": True,
    "is_partial_match": False,
    "explanation": "same root cause and impact",
    "severity_from_junior_auditor": "N/A",
    "severity_from_truth": "High",
    "index_of_finding_from_junior_auditor": 0,
}


def test_a_free_text_report_is_extracted_then_matched_against_the_truth():
    report = "I think f is reentrant: it sends ETH before writing state, so funds drain."
    extracted = [{"issue": "Reentrancy", "description": report, "severity": "N/A", "contracts": []}]
    calls, extract, match = judges(extracted, MATCH)
    sample = {"final_text": report, "finish_reason": "stop"}
    result = asyncio.run(free_text.grade_draw(sample, TRUTH, "{ junior_findings }", extract, match))
    assert report in calls["extract"][0]
    assert json.dumps(TRUTH[0]["issue"]) not in calls["extract"][0]  # no truth leaks in
    assert result["status"] == "graded" and result["exact_coverage"] == 1.0
    assert result["extracted_findings"] == extracted


def test_a_report_with_no_vulnerabilities_is_a_real_zero():
    calls, extract, match = judges([], MATCH)
    sample = {"final_text": "The contracts look safe.", "finish_reason": "stop"}
    result = asyncio.run(free_text.grade_draw(sample, TRUTH, "", extract, match))
    assert result["status"] == "graded" and result["exact_coverage"] == 0.0
    assert calls["match"] == []


@pytest.mark.parametrize(
    "sample,status",
    [
        ({"final_text": "half an answer", "finish_reason": "length"}, "truncated"),
        ({"final_text": "", "finish_reason": "stop"}, "no_answer"),
    ],
)
def test_a_cut_off_or_empty_answer_stays_unknown(sample, status):
    calls, extract, match = judges([], MATCH)
    result = asyncio.run(free_text.grade_draw(sample, TRUTH, "", extract, match))
    assert result["exact_coverage"] is None and result["status"] == status
    assert calls["extract"] == []


def draw(status, exact=None, tokens=100):
    return {
        "status": status,
        "known_gt_count": 4,
        "exact_coverage": exact,
        "coverage_including_partial": exact,
        "completion_tokens": tokens,
    }


def test_summary_averages_graded_draws_and_counts_the_rest():
    base = {"label": "base", "draws": [draw("graded", 0.25), draw("graded", 0.5)]}
    base["draws"] += [draw("truncated"), draw("graded", 0.75)]
    summary = free_text.summarize([base])
    arm = summary["arms"]["base"]
    assert arm["mean_exact_coverage"] == pytest.approx(0.5)
    assert arm["status_counts"] == {"graded": 3, "truncated": 1}
    assert summary["status"] == "incomplete" and "deltas" not in summary


def test_summary_of_two_complete_arms_reports_the_trained_delta():
    base = {"label": "base", "draws": [draw("graded", 0.25)] * 4}
    trained = {"label": "checkpoint-4", "draws": [draw("graded", 0.5)] * 4}
    summary = free_text.summarize([base, trained])
    assert summary["status"] == "complete"
    assert summary["deltas"]["checkpoint-4"]["mean_exact_coverage"] == pytest.approx(0.25)


def test_a_draw_that_stops_inside_its_reasoning_is_graded_on_that_reasoning():
    thoughts = "<think>f sends ETH before writing state; that is a real reentrancy bug"
    extracted = [{"issue": "Reentrancy", "description": "...", "severity": "N/A", "contracts": []}]
    _, extract, match = judges(extracted, MATCH)
    sample = {"final_text": thoughts, "raw_text": thoughts, "finish_reason": "stop"}
    result = asyncio.run(free_text.grade_draw(sample, TRUTH, "", extract, match))
    assert result["answer_source"] == "unclosed_reasoning" and result["exact_coverage"] == 1.0
    closed = dict(sample, raw_text="<think>...</think>" + thoughts)
    result = asyncio.run(free_text.grade_draw(closed, TRUTH, "", extract, match))
    assert result["answer_source"] == "final_answer"
