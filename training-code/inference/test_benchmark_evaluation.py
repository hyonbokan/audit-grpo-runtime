"""Native benchmark configuration and paired summaries cannot accept unknown results."""

import copy
import hashlib
import json

import pytest

from testing.benchmark import configuration, native_context_configuration


def test_markdown_one_shot_config_admits_one_answer_but_rejects_mismatched_sampling():
    from benchmark_evaluation import validate_benchmark_config

    config = configuration()
    config.update(benchmark_suite="solidity1_markdown", output_format="markdown")
    config["sampling"]["n"] = 1
    validate_benchmark_config(config)
    config["sampling"]["n"] = 4
    with pytest.raises(ValueError):
        validate_benchmark_config(config)


def test_merged_evaluation_allows_bounded_deadline_and_checkpoint_first():
    from benchmark_evaluation import validate_benchmark_config

    config = configuration()
    config.update(
        benchmark_suite="solidity1_markdown",
        output_format="markdown",
        merged_model_path="/merged",
        evaluation_deadline_override="2026-10-09T12:23:06+00:00",
        generation_order=["checkpoint-2", "base"],
    )
    config["sampling"]["n"] = 1
    validate_benchmark_config(config)
    config["evaluation_deadline_override"] = "2026-10-09T12:23:07+00:00"
    with pytest.raises(ValueError, match="deadline"):
        validate_benchmark_config(config)
    config["evaluation_deadline_override"] = "2026-10-09T12:23:06+00:00"
    config["generation_order"] = ["checkpoint-2", "checkpoint-2"]
    with pytest.raises(ValueError, match="generation order"):
        validate_benchmark_config(config)


def test_full_native_context_accepts_remaining_window_for_both_arms():
    from benchmark_evaluation import validate_benchmark_config

    config = native_context_configuration()
    validate_benchmark_config(config)
    for case in config["cases"]:
        validate_benchmark_config(dict(config, **case))


@pytest.mark.parametrize("arms", [["checkpoint-2", "base"], ["checkpoint-2"]])
def test_explicit_repeat_uses_fresh_seed_and_only_requested_arms(arms):
    from benchmark_evaluation import validate_benchmark_config

    config = native_context_configuration()
    config.update(repeat_evaluation={"seed": 3408, "arms": arms}, generation_order=arms)
    config["sampling"]["seed"] = 3408
    config["engine"]["seed"] = 3408
    validate_benchmark_config(config)
    config["sampling"]["seed"] = 3407
    with pytest.raises(ValueError):
        validate_benchmark_config(config)


@pytest.mark.parametrize(
    "arms,seed",
    [
        ([], 3408),
        (["other"], 3408),
        (["checkpoint-2"] * 2, 3408),
        (["checkpoint-2"], 3407),
        (["checkpoint-2"], True),
    ],
)
def test_repeat_rejects_empty_duplicate_unknown_arms_or_reused_seed(arms, seed):
    from benchmark_evaluation import validate_benchmark_config

    config = native_context_configuration()
    config.update(repeat_evaluation={"seed": seed, "arms": arms}, generation_order=arms)
    config["sampling"]["seed"] = seed
    config["engine"]["seed"] = seed
    with pytest.raises(ValueError):
        validate_benchmark_config(config)


@pytest.mark.parametrize("corrupt_receipt", [False, True])
def test_single_arm_repeat_preserves_grades_without_inventing_paired_gain(
    tmp_path, corrupt_receipt
):
    from benchmark_evaluation import finish_benchmark

    config = native_context_configuration()
    config.update(
        repeat_evaluation={"seed": 3408, "arms": ["checkpoint-2"]},
        generation_order=["checkpoint-2"],
    )
    config["sampling"]["seed"] = config["engine"]["seed"] = 3408
    (tmp_path / "prepared.json").write_text(json.dumps(config))
    request = {
        "prompt_ids": [1, 2],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "model": {"precision": "bf16"},
        "max_model_len": 262144,
        "full_completion_ceiling": 227051,
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    truth = b'[{"issue":"known"}]'
    template = b"judge template"
    (tmp_path / "private-truth.json").write_bytes(truth)
    (tmp_path / "judge-prompt.txt").write_bytes(template)
    case = config["cases"][1]
    (tmp_path / "checkpoint-2-config.json").write_text(
        json.dumps(dict(config, **case, request=str(tmp_path / "request.json")))
    )
    arm = tmp_path / "checkpoint-2"
    arm.mkdir()
    samples = json.dumps(
        [
            {
                "prompt_ids": [1, 2],
                "completion_ids": [3],
                "weight_version": f"step-2:{case['initial_adapter_sha256']}",
            }
        ]
    ).encode()
    (arm / "samples.json").write_bytes(samples)
    grading = tmp_path / "checkpoint-2-grading"
    grading.mkdir()
    hashes = {
        "samples": hashlib.sha256(samples).hexdigest(),
        "private_truth": hashlib.sha256(truth).hexdigest(),
        "judge_prompt": hashlib.sha256(template).hexdigest(),
    }
    if corrupt_receipt:
        hashes["samples"] = "0" * 64
    (grading / "metrics.json").write_text(
        json.dumps(
            {
                "input_sha256": hashes,
                "judge": config["benchmark_judge"],
                "draws": [
                    {
                        "status": "graded",
                        "known_gt_count": 23,
                        "exact_matches": 1,
                        "partial_matches": 4,
                    }
                ],
            }
        )
    )
    if corrupt_receipt:
        with pytest.raises(ValueError, match="Grading"):
            finish_benchmark(config, tmp_path)
        assert not (tmp_path / "comparison.json").exists()
    else:
        finish_benchmark(config, tmp_path)
        result = json.loads((tmp_path / "comparison.json").read_text())
        assert result["status"] == "single_arm_completed"
        assert [arm["label"] for arm in result["arms"]] == ["checkpoint-2"]
        assert result["arms"][0]["draws"][0]["partial_matches"] == 4
        assert result["mean_exact_coverage_delta"] is None
        assert result["quality_gain_established"] is False
        assert not (tmp_path / "base").exists()


@pytest.mark.parametrize("change", ["output", "context", "deadline", "sampling", "budget"])
def test_full_native_context_rejects_inconsistent_or_unbounded_configuration(change):
    from benchmark_evaluation import validate_benchmark_config

    config = native_context_configuration()
    if change == "output":
        config["sampling"]["max_tokens"] = 227052
    elif change == "context":
        config["engine"]["max_model_len"] = 1000000
    elif change == "deadline":
        config["absolute_deadline"] = "2026-10-11T12:30:00+00:00"
    elif change == "sampling":
        config["sampling"]["temperature"] = 0
    else:
        config["seconds_limit"] = 999999
    with pytest.raises(ValueError):
        validate_benchmark_config(config)


def test_completion_allowance_uses_verified_actual_prompt_length():
    from benchmark_evaluation import completion_allowance

    config = native_context_configuration()
    assert completion_allowance(config, 35093) == 227051
    with pytest.raises(ValueError, match="prompt token count"):
        completion_allowance(config, 35092)
    config["sampling"]["max_tokens"] = 16384
    with pytest.raises(ValueError, match="remaining context"):
        completion_allowance(config, 35093)


@pytest.mark.parametrize("change", ["judge", "sampling", "arms", "budget", "revision"])
def test_native_configuration_rejects_unmatched_settings(change):
    from benchmark_evaluation import validate_benchmark_config

    config = configuration()
    validate_benchmark_config(config)
    changed = copy.deepcopy(config)
    if change == "judge":
        changed["benchmark_judge"]["api_effort"] = "max"
    elif change == "sampling":
        changed["sampling"]["n"] = 1
    elif change == "arms":
        changed["cases"].reverse()
    elif change == "budget":
        changed["seconds_limit"] = 999999
    else:
        changed["model_revision"] = "other"
    with pytest.raises(ValueError):
        validate_benchmark_config(changed)


@pytest.mark.parametrize("name", ["munchables", "vultisig", "tigris", "backd", "phi"])
def test_solidity1_configuration_admits_complete_scopes_with_matched_low_effort(name):
    from benchmark_evaluation import validate_benchmark_config

    config = configuration()
    config.update(
        benchmark_suite="solidity1",
        row_id=f"solidity/{name}",
        engine=dict(config["engine"], max_model_len=53248),
        seconds_limit=5400,
        judge_seconds_limit=1800,
        judge_operation_cap_usd=150,
        content_panel="conditional_content_after_sole_outer_json_fence_removed",
    )
    validate_benchmark_config(config)
    changed = copy.deepcopy(config)
    changed["row_id"] = "solidity/not-in-the-reviewed-suite"
    with pytest.raises(ValueError):
        validate_benchmark_config(changed)
    changed = copy.deepcopy(config)
    changed["engine"]["max_model_len"] = 999999
    with pytest.raises(ValueError):
        validate_benchmark_config(changed)


def test_normalization_selects_checkpoint_arm_and_preserves_raw_fenced_answer(tmp_path):
    from benchmark_evaluation import main

    config = configuration()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    arm = tmp_path / "checkpoint-2"
    arm.mkdir()
    original = {"final_text": '```json\n{"vulnerabilities":[]}\n```', "finish_reason": "stop"}
    (arm / "samples.json").write_text(json.dumps([original] * 4))
    assert (
        main(
            ["normalize", "--config", str(path), "--out", str(tmp_path), "--label", "checkpoint-2"]
        )
        == 0
    )
    copied = json.loads((tmp_path / "conditional/checkpoint-2/samples.json").read_text())
    assert copied[0]["final_text"] == '{"vulnerabilities":[]}'
    assert json.loads((arm / "samples.json").read_text())[0] == original
    validity = json.loads((tmp_path / "conditional/checkpoint-2/strict-validity.json").read_text())
    assert validity["strict_valid_fraction"] == 0


@pytest.mark.parametrize("sample_count", [1, 4])
@pytest.mark.parametrize("unread", ["unknown", "invalid", "truncated"])
def test_paired_summary_reports_coverage_delta_and_keeps_ungraded_draws_unknown(
    sample_count, unread
):
    from benchmark_evaluation import summarize_benchmark

    def arm(exact):
        row = {
            "status": "graded",
            "known_gt_count": 11,
            "exact_coverage": exact,
            "coverage_including_partial": exact + 0.1,
        }
        return {"draws": [dict(row) for _ in range(sample_count)]}

    summary = summarize_benchmark(arm(0.25), arm(0.5), sample_count=sample_count)
    assert summary["mean_exact_coverage_delta"] == 0.25
    assert summary["mean_coverage_including_partial_delta"] == pytest.approx(0.25)
    assert summary["quality_gain_established"] is False
    right = arm(0.5)
    right["draws"][0].update(status=unread, exact_coverage=None, coverage_including_partial=None)
    summary = summarize_benchmark(arm(0.25), right, sample_count=sample_count)
    assert summary["status"] == "inconclusive"
    assert summary["mean_exact_coverage_delta"] is None
    assert summary["mean_coverage_including_partial_delta"] is None


@pytest.mark.parametrize(
    "corruption", [None, "missing_receipt", "samples", "truth", "template", "judge", "swapped"]
)
@pytest.mark.parametrize("panel", ["json", "conditional", "markdown"])
def test_finish_requires_grading_receipts_for_exact_inputs(tmp_path, corruption, panel):
    from benchmark_evaluation import finish_benchmark

    config = configuration()
    conditional = panel == "conditional"
    if panel == "markdown":
        config.update(benchmark_suite="solidity1_markdown", output_format="markdown")
        config["sampling"]["n"] = 1
    if conditional:
        config.update(
            benchmark_suite="solidity1",
            row_id="solidity/vultisig",
            content_panel="conditional_content_after_sole_outer_json_fence_removed",
        )
    (tmp_path / "prepared.json").write_text(json.dumps(config))
    request = {
        "prompt_ids": [1, 2],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "model": {"precision": "bf16"},
        "max_model_len": 36864,
        "full_completion_ceiling": 16384,
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    truth = tmp_path / "private-truth.json"
    truth.write_text('[{"issue":"known"}]')
    template = tmp_path / "judge-prompt.txt"
    template.write_text("judge template")
    for case in config["cases"]:
        label = case["label"]
        (tmp_path / f"{label}-config.json").write_text(
            json.dumps(dict(config, **case, request=str(tmp_path / "request.json")))
        )
        version = (
            f"base:{config['model_revision']}"
            if label == "base"
            else f"step-2:{case['initial_adapter_sha256']}"
        )
        sample_path = tmp_path / label / "samples.json"
        sample_path.parent.mkdir()
        samples = [
            {
                "prompt_ids": [1, 2],
                "weight_version": version,
                "completion_ids": [3],
                "final_text": "No vulnerabilities found."
                if panel == "markdown"
                else '{"vulnerabilities":[]}',
                "finish_reason": "stop",
            }
            for _ in range(config["sampling"]["n"])
        ]
        sample_path.write_text(json.dumps(samples))
        if conditional:
            raw = copy.deepcopy(samples)
            for sample in raw:
                sample["final_text"] = '```json\n{"vulnerabilities":[]}\n```'
            sample_path.write_text(json.dumps(raw))
            sample_path = tmp_path / "conditional" / label / "samples.json"
            sample_path.parent.mkdir(parents=True)
            sample_path.write_text(json.dumps(samples))
            (sample_path.parent / "strict-validity.json").write_text(
                json.dumps({"strict_valid_fraction": 0})
            )
        measured = {
            "input_sha256": {
                "samples": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                "private_truth": hashlib.sha256(truth.read_bytes()).hexdigest(),
                "judge_prompt": hashlib.sha256(template.read_bytes()).hexdigest(),
            },
            "judge": config["benchmark_judge"],
            "draws": [
                {
                    "status": "graded",
                    "known_gt_count": 1,
                    "exact_coverage": 0,
                    "coverage_including_partial": 0,
                }
                for _ in range(config["sampling"]["n"])
            ],
        }
        if label == "checkpoint-2":
            if corruption == "missing_receipt":
                del measured["input_sha256"]
            elif corruption == "samples":
                samples[0]["final_text"] = "changed answer"
                sample_path.write_text(json.dumps(samples))
            elif corruption == "truth":
                truth.write_text('[{"issue":"different"}]')
            elif corruption == "template":
                template.write_text("different judge template")
            elif corruption == "judge":
                measured["judge"] = dict(config["benchmark_judge"], api_effort="low")
        grading_path = tmp_path / f"{label}-grading" / "metrics.json"
        grading_path.parent.mkdir()
        grading_path.write_text(json.dumps(measured))
    if corruption == "swapped":
        (tmp_path / "checkpoint-2-grading/metrics.json").write_bytes(
            (tmp_path / "base-grading/metrics.json").read_bytes()
        )
    if corruption is not None:
        with pytest.raises(ValueError, match="Grading"):
            finish_benchmark(config, tmp_path)
        assert not (tmp_path / "comparison.json").exists()
    else:
        finish_benchmark(config, tmp_path)
        result = json.loads((tmp_path / "comparison.json").read_text())
        assert result["status"] == "matched_benchmark_completed"
        assert result["mean_exact_coverage_delta"] == 0
        if conditional:
            assert result["strict_json_validity"] == {"base": 0, "checkpoint-2": 0}
            assert result["panel"] == config["content_panel"]
            assert "vultisig" in result["scope"]
