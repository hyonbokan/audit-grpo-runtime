"""Reject unfair comparisons and keep unknown grades from becoming an improvement."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from testing.clock import freeze


def configuration():
    return json.loads((Path(__file__).parents[1] / "fixtures/evaluation.json").read_text())


@pytest.mark.parametrize("held_out", [True, False])
def test_preparation_uses_complete_policy_only_held_out_input(tmp_path, monkeypatch, held_out):
    import evaluation
    from evaluation import prepare_evaluation

    config = configuration()
    freeze(monkeypatch, evaluation, "2026-10-09T00:00:00+00:00")
    config["workspace"] = str(tmp_path / "workspace")
    policy = Path(config["workspace"]) / "data/policy"
    policy.mkdir(parents=True)
    row = {"row_id": config["row_id"], "messages": [{"role": "user", "content": "full code"}]}
    for split in ("train", "calibration", "evaluation"):
        selected = "evaluation" if held_out else "train"
        (policy / f"{split}.jsonl").write_text(json.dumps(row) + "\n" if split == selected else "")
    config["model_path"] = str(tmp_path / "model")
    Path(config["model_path"]).mkdir()
    config["source_plan"] = str(tmp_path / "plan.json")
    Path(config["source_plan"]).write_text(
        json.dumps(
            {
                "model": {
                    "id": config["model_id"],
                    "revision": config["model_revision"],
                    "precision": "bf16",
                },
                "sampling_profile": {"id": "schema-low-16k-v1", "policy_reasoning_effort": "low"},
                "judge": {"model": "fixed-judge"},
            }
        )
    )
    for case in config["cases"]:
        checkpoint = tmp_path / case["label"]
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_bytes(case["label"].encode())
        (checkpoint / "adapter_config.json").write_text('{"r":16,"lora_alpha":32}')
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": case["initial_step"]})
        )
        case["checkpoint"] = str(checkpoint)
        case["initial_adapter_sha256"] = hashlib.sha256(case["label"].encode()).hexdigest()

    class TokenizerBoundary:
        chat_template = "fixed template"

        def apply_chat_template(self, messages, **options):
            assert messages == [{"role": "user", "content": "full code"}]
            assert options["reasoning_effort"] == "low"
            return [10, 20, 30]

    out = tmp_path / "result"
    out.mkdir()
    if not held_out:
        with pytest.raises(ValueError, match="held-out"):
            prepare_evaluation(config, out, tokenizer=TokenizerBoundary())
        assert not (out / "request.json").exists()
        return
    prepare_evaluation(config, out, tokenizer=TokenizerBoundary())
    request = json.loads((out / "request.json").read_text())
    assert request["prompt_ids"] == [10, 20, 30]
    assert request["full_completion_ceiling"] == 16384
    assert request["template_options"]["reasoning_effort"] == "low"
    assert json.loads((out / "checkpoint-2-config.json").read_text())["initial_step"] == 2


def metrics(coverage, caught, hits, unmatched):
    return {
        "row_id": "held-out",
        "draws": 4,
        "eligible_gt_ids": ["a", "b"],
        "caught_gt_ids": caught,
        "known_gt_coverage": coverage,
        "unknown_draws": 0,
        "valid_final_trace_rate": 1.0,
        "mean_findings": (hits + unmatched) / 4,
        "quality_per_hit": 1.0,
        "matched_hits": hits,
        "quality_sum": float(hits),
        "unmatched_findings": unmatched,
    }


def test_comparison_reports_coverage_and_false_positive_changes_without_promotion():
    from evaluation import summarize_metrics

    result = summarize_metrics(
        metrics(0.5, ["a"], 2, 2),
        metrics(1.0, ["a", "b"], 4, 0),
        configuration()["thresholds"],
    )
    assert result["coverage_delta"] == 0.5
    assert result["matched_hits_delta"] == 2
    assert result["unmatched_findings_delta"] == -2
    assert result["quality_gain_established"] is False
    assert result["comparison"]["production_promotion"] is False


@pytest.mark.parametrize("change", ["unknown", "denominator", "draw_count", "row"])
def test_comparison_rejects_unmatched_or_unknown_measurements(change):
    from evaluation import summarize_metrics

    left = metrics(0.5, ["a"], 2, 2)
    right = copy.deepcopy(left)
    key, value = {
        "unknown": ("unknown_draws", 1),
        "denominator": ("eligible_gt_ids", ["a"]),
        "draw_count": ("draws", 3),
        "row": ("row_id", "other"),
    }[change]
    right[key] = value
    with pytest.raises(ValueError):
        summarize_metrics(left, right, configuration()["thresholds"])


@pytest.mark.parametrize("change", ["precision", "sampling", "budget", "support"])
def test_evaluation_configuration_rejects_silent_contract_changes(change):
    from evaluation import validate_evaluation_config

    config = configuration()
    if change == "precision":
        config["engine"]["dtype"] = "float16"
    elif change == "sampling":
        config["sampling"]["n"] = 1
    elif change == "budget":
        config["seconds_limit"] = 2701
    else:
        config["engine"]["return_sampling_mask"] = True
    with pytest.raises(ValueError):
        validate_evaluation_config(config)


@pytest.mark.parametrize("corruption", [None, "prompt", "version", "sampling", "reward", "metrics"])
def test_finish_saves_only_a_complete_matched_comparison(tmp_path, corruption):
    from evaluation import finish_evaluation

    config = configuration()
    (tmp_path / "prepared.json").write_text(json.dumps(config))
    request = {
        "prompt_ids": [1, 2],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "model": {"precision": "bf16"},
        "max_model_len": 73728,
        "full_completion_ceiling": 16384,
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    for case in config["cases"]:
        label = case["label"]
        arm = dict(config, **case, request=str(tmp_path / "request.json"))
        if corruption == "sampling" and label == "checkpoint-2":
            arm["sampling"] = dict(arm["sampling"], seed=123)
        (tmp_path / f"{label}-config.json").write_text(json.dumps(arm))
        samples = [
            {
                "row_id": config["row_id"],
                "prompt_ids": [1, 2],
                "completion_ids": [3],
                "logprobs": [-0.1],
                "weight_version": f"step-{case['initial_step']}:{case['initial_adapter_sha256']}",
                "logprob_semantics": "sampling_processed",
                "finish_reason": "stop",
            }
            for _ in range(4)
        ]
        if corruption in ("prompt", "version") and label == "checkpoint-2":
            key, value = (
                ("prompt_ids", [1]) if corruption == "prompt" else ("weight_version", "step-1")
            )
            samples[0][key] = value
        (tmp_path / label).mkdir()
        (tmp_path / label / "samples.json").write_text(json.dumps(samples))
        grading = tmp_path / f"{label}-grading"
        grading.mkdir()
        measured = metrics(0.5, ["a"], 2, 2)
        measured["row_id"] = config["row_id"]
        if corruption == "metrics":
            measured["row_id"] = "other"
        (grading / "metrics.json").write_text(json.dumps(measured))
        rewards = [0, 1, 0, 1] if label == "checkpoint-1" else [1, 1, 1, 1]
        if corruption == "reward" and label == "checkpoint-2":
            rewards[0] = None
        (grading / "rewards.json").write_text(json.dumps(rewards))
    if corruption:
        with pytest.raises(ValueError):
            finish_evaluation(config, tmp_path)
        assert not (tmp_path / "comparison.json").exists()
    else:
        result = finish_evaluation(config, tmp_path)
        assert result["mean_reward_delta"] == 0.5
        assert result["coverage_delta"] == 0
        assert result["optimizer_steps"] == 0
        assert (
            json.loads((tmp_path / "comparison.json").read_text())["arms"][1]["truncated_draws"]
            == 0
        )


def test_unknown_evaluation_kind_is_rejected_instead_of_running_the_legacy_comparison():
    from evaluation import launcher_settings, validate_evaluation_config

    config = configuration()
    config["evaluation_kind"] = "grpo_heldout_panel_typo"
    with pytest.raises(ValueError, match="Unknown evaluation_kind"):
        validate_evaluation_config(config)
    with pytest.raises(ValueError, match="Unknown evaluation_kind"):
        launcher_settings(config)


def test_launcher_settings_select_the_module_and_grading_mode_for_the_kind():
    from evaluation import launcher_settings
    from testing.benchmark import native_context_configuration

    config = native_context_configuration()
    config.update(
        generation_order=["checkpoint-2", "base"], overlap_grading=True, judge_seconds_limit=900
    )
    assert launcher_settings(config) == [
        config["native_image"],
        config["rollout_image"],
        str(config["seconds_limit"]),
        "900",
        "benchmark_evaluation",
        "checkpoint-2 base",
        "strict",
        "evaluation",
        "overlap",
    ]
