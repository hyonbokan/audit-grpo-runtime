"""One new evaluation arm reuses matched saved scores and preserves strict validity."""

import copy
import hashlib
import json

import pytest

from testing.benchmark import configuration


def test_continuation_generates_only_the_new_checkpoint_with_shared_judge_capacity():
    from benchmark_evaluation import validate_benchmark_config

    config = configuration()
    config.update(
        continuation_reference="/results/reference",
        reference_comparison_sha256="c" * 64,
        content_panel="conditional_content_after_sole_outer_json_fence_removed",
        judge_operation_cap_usd=150,
        judge_seconds_limit=1800,
        cases=[
            {
                "label": "checkpoint-8",
                "initial_step": 8,
                "checkpoint": "/checkpoint-8",
                "initial_adapter_sha256": "d" * 64,
            }
        ],
    )
    validate_benchmark_config(config)
    for step in (2, 9):
        changed = copy.deepcopy(config)
        changed["cases"][0].update(label=f"checkpoint-{step}", initial_step=step)
        with pytest.raises(ValueError):
            validate_benchmark_config(changed)


def test_conditional_normalization_preserves_raw_evidence_and_strict_failures():
    from benchmark_evaluation import normalize_samples

    original = [
        {
            "final_text": '```json\n{"vulnerabilities": []}\n```',
            "finish_reason": "stop",
            "raw_text": "raw fenced answer",
            "completion_ids": [1, 2],
            "reasoning": "trace",
        },
        {"final_text": '{"vulnerabilities": []}', "finish_reason": "stop"},
        {"final_text": '```json\n{"vulnerabilities": []}\n```', "finish_reason": "length"},
        {"final_text": 'prose\n```json\n{"vulnerabilities": []}\n```', "finish_reason": "stop"},
    ]
    saved = copy.deepcopy(original)
    normalized, report = normalize_samples(original)
    assert original == saved
    assert normalized[0]["final_text"] == '{"vulnerabilities": []}'
    assert normalized[0]["raw_text"] == "raw fenced answer"
    assert normalized[0]["completion_ids"] == [1, 2]
    assert normalized[0]["reasoning"] == "trace"
    assert normalized[3] == saved[3]
    assert report["strict_valid_fraction"] == 0.25
    assert report["fences_removed"] == [True, False, False, False]


def test_saved_reference_reuse_rejects_a_different_sampling_profile(tmp_path):
    from benchmark_evaluation import load_reference

    config = configuration()
    config.update(continuation_reference=str(tmp_path), reference_comparison_sha256="a" * 64)
    changed = copy.deepcopy(config)
    changed["sampling"]["max_tokens"] = 8192
    (tmp_path / "prepared.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="sampling"):
        load_reference(config, tmp_path / "fresh")


@pytest.mark.parametrize("outcome", ["improved", "regressed", "unknown", "tampered"])
def test_final_continuation_decision_uses_matched_saved_arms(tmp_path, outcome):
    from benchmark_contract import grading_input_hashes
    from benchmark_evaluation import finish_benchmark

    reference = tmp_path / "reference"
    out = tmp_path / "new"
    reference.mkdir()
    out.mkdir()
    config = configuration()
    (reference / "prepared.json").write_text(json.dumps(config))
    config.update(
        continuation_reference=str(reference),
        content_panel="conditional_content_after_sole_outer_json_fence_removed",
        judge_operation_cap_usd=150,
        judge_seconds_limit=1800,
        cases=[
            {
                "label": "checkpoint-8",
                "initial_step": 8,
                "checkpoint": "/checkpoint-8",
                "initial_adapter_sha256": "d" * 64,
            }
        ],
    )
    request = {
        "prompt_ids": [1, 2],
        "model": {"precision": "bf16"},
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "max_model_len": 36864,
        "full_completion_ceiling": 16384,
    }
    for root in (reference, out):
        (root / "request.json").write_text(json.dumps(request))
        (root / "private-truth.json").write_text('[{"issue":"known"}]')
        (root / "judge-prompt.txt").write_text("judge template")
        (root / "source-scope.json").write_text('[{"path":"A.sol","sha256":"source"}]')

    def draws(coverage):
        return [
            {
                "status": "graded",
                "known_gt_count": 11,
                "exact_coverage": coverage,
                "coverage_including_partial": coverage + 0.1,
            }
            for _ in range(4)
        ]

    saved = {
        "panel": config["content_panel"],
        "strict_json_validity": {"base": 0, "checkpoint-2": 0},
        "arms": {"base": {"draws": draws(0.1)}, "checkpoint-2": {"draws": draws(0.2)}},
    }
    comparison = reference / "normalized/comparison.json"
    comparison.parent.mkdir()
    comparison.write_text(json.dumps(saved))
    config["reference_comparison_sha256"] = hashlib.sha256(comparison.read_bytes()).hexdigest()
    (out / "prepared.json").write_text(json.dumps(config))
    case = config["cases"][0]
    (out / "checkpoint-8-config.json").write_text(
        json.dumps(dict(config, **case, request=str(out / "request.json")))
    )
    sample_path = out / "conditional/checkpoint-8/samples.json"
    sample_path.parent.mkdir(parents=True)
    samples = [
        {
            "prompt_ids": [1, 2],
            "weight_version": f"step-8:{'d' * 64}",
            "completion_ids": [3],
            "final_text": '{"vulnerabilities":[]}',
            "finish_reason": "stop",
        }
        for _ in range(4)
    ]
    sample_path.write_text(json.dumps(samples))
    metrics = {
        "input_sha256": grading_input_hashes(
            sample_path.read_bytes(),
            (out / "private-truth.json").read_bytes(),
            (out / "judge-prompt.txt").read_bytes(),
        ),
        "judge": config["benchmark_judge"],
        "draws": draws(0.3 if outcome != "regressed" else 0.1),
    }
    if outcome == "unknown":
        metrics["draws"][0].update(status="unknown", exact_coverage=None)
    grading = out / "checkpoint-8-grading"
    grading.mkdir()
    (grading / "metrics.json").write_text(json.dumps(metrics))
    (out / "strict-validity.json").write_text('{"strict_valid_fraction":0.25}')
    if outcome == "tampered":
        samples[0]["completion_ids"] = [4]
        sample_path.write_text(json.dumps(samples))
        with pytest.raises(ValueError, match="Grading receipt"):
            finish_benchmark(config, out)
        assert not (out / "comparison.json").exists()
        return
    finish_benchmark(config, out)
    result = json.loads((out / "comparison.json").read_text())
    assert result["positive_quality_signal"] == (
        None if outcome == "unknown" else outcome == "improved"
    )
    # Measured 0.3 (or 0.1 when regressed) against saved checkpoint-2 at 0.2 and base at 0.1.
    deltas = {"improved": (0.1, 0.2), "regressed": (-0.1, 0.0), "unknown": (None, None)}
    previous, base = deltas[outcome]
    assert result["mean_exact_coverage_delta"] == pytest.approx(previous)
    assert result["versus_base"]["mean_exact_coverage_delta"] == pytest.approx(base)
    assert result["quality_gain_established"] is False
    assert result["strict_json_validity"]["checkpoint-8"] == 0.25
    assert set(result["arms"]) == {"base", "checkpoint-2", "checkpoint-8"}
