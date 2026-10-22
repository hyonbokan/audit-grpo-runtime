"""One free-text benchmark evaluation: launcher settings, frozen inputs and the comparison."""

import hashlib
import json

import pytest

import benchmark_free_text as free_text
from testing.benchmark import free_text_configuration


def test_settings_name_the_images_budgets_arms_and_grading_mode(capsys, tmp_path):
    from benchmark_evaluation import main

    config = dict(free_text_configuration(), overlap_grading=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert main(["settings", "--config", str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        config["native_image"],
        config["rollout_image"],
        str(config["seconds_limit"]),
        str(config["judge_seconds_limit"]),
        "base",
        "overlap",
    ]


def test_settings_refuse_a_config_outside_the_free_text_contract(tmp_path):
    from benchmark_evaluation import main

    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(free_text_configuration(), benchmark_suite="solidity1")))
    with pytest.raises(ValueError, match="benchmark_suite"):
        main(["settings", "--config", str(path)])


def test_completion_allowance_uses_verified_actual_prompt_length():
    from benchmark_evaluation import completion_allowance

    config = free_text_configuration()
    assert completion_allowance(config, 30000) == 262144 - 30000
    with pytest.raises(ValueError, match="prompt token count"):
        completion_allowance(config, 29999)
    config["sampling"]["max_tokens"] = 16384
    with pytest.raises(ValueError, match="remaining context"):
        completion_allowance(config, 30000)


@pytest.mark.parametrize(
    "corruption", [None, "missing_receipt", "samples", "truth", "template", "judge", "swapped"]
)
def test_finish_requires_grading_receipts_for_exact_inputs(tmp_path, corruption):
    from benchmark_evaluation import finish_benchmark

    trained = {
        "label": "checkpoint-2",
        "initial_step": 2,
        "checkpoint": "/checkpoint",
        "initial_adapter_sha256": "b" * 64,
    }
    config = free_text_configuration()
    config.update(
        cases=[*config["cases"], trained],
        generation_order=["base", "checkpoint-2"],
        merged_model_path="/merged",
    )
    (tmp_path / "prepared.json").write_text(json.dumps(config))
    request = {
        "prompt_ids": [1, 2],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "model": {"precision": "bf16"},
        "max_model_len": 262144,
        "full_completion_ceiling": config["sampling"]["max_tokens"],
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
                "final_text": "No vulnerabilities found.",
                "finish_reason": "stop",
            }
            for _ in range(config["sampling"]["n"])
        ]
        sample_path.write_text(json.dumps(samples))
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
                    "completion_tokens": 1,
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
        assert result["status"] == "complete"
        assert result["deltas"] == {
            "checkpoint-2": {"mean_exact_coverage": 0, "mean_coverage_including_partial": 0}
        }


def test_free_text_base_arm_finishes_with_per_arm_means_and_no_invented_gain(tmp_path):
    from benchmark_evaluation import finish_benchmark
    from testing.benchmark import free_text_configuration

    config = free_text_configuration()
    free_text.validate_config(config)
    (tmp_path / "prepared.json").write_text(json.dumps(config))
    request = {
        "prompt_ids": [1, 2],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "model": {"precision": "bf16"},
        "max_model_len": 262144,
        "full_completion_ceiling": config["sampling"]["max_tokens"],
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    truth, template = b'[{"issue":"known"}]', b"judge template"
    (tmp_path / "private-truth.json").write_bytes(truth)
    (tmp_path / "judge-prompt.txt").write_bytes(template)
    (tmp_path / "base-config.json").write_text(
        json.dumps(dict(config, **config["cases"][0], request=str(tmp_path / "request.json")))
    )
    version = f"base:{config['model_revision']}"
    samples = json.dumps(
        [{"prompt_ids": [1, 2], "completion_ids": [3, 4], "weight_version": version}] * 4
    ).encode()
    (tmp_path / "base").mkdir()
    (tmp_path / "base/samples.json").write_bytes(samples)
    (tmp_path / "base-grading").mkdir()
    draws = [
        {
            "status": "graded",
            "exact_coverage": c,
            "coverage_including_partial": c,
            "completion_tokens": 2,
        }
        for c in (0.0, 0.5, 0.5, 1.0)
    ]
    (tmp_path / "base-grading/metrics.json").write_text(
        json.dumps(
            {
                "draws": draws,
                "judge": config["benchmark_judge"],
                "input_sha256": {
                    "samples": hashlib.sha256(samples).hexdigest(),
                    "private_truth": hashlib.sha256(truth).hexdigest(),
                    "judge_prompt": hashlib.sha256(template).hexdigest(),
                },
            }
        )
    )
    finish_benchmark(config, tmp_path)
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert result["status"] == "complete" and result["output_format"] == "free_text"
    assert result["arms"]["base"]["mean_exact_coverage"] == 0.5
    assert result["quality_gain_established"] is False and "deltas" not in result


def test_only_trained_arms_have_checkpoints_to_verify(tmp_path):
    from benchmark_evaluation import verify_checkpoints

    verify_checkpoints([{"label": "base", "initial_step": 0}])
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 4}')
    arm = {
        "label": "checkpoint-4",
        "initial_step": 4,
        "checkpoint": str(checkpoint),
        "initial_adapter_sha256": hashlib.sha256(b"weights").hexdigest(),
    }
    verify_checkpoints([{"label": "base", "initial_step": 0}, arm])
    with pytest.raises(ValueError, match="identity"):
        verify_checkpoints([dict(arm, initial_adapter_sha256="0" * 64)])
    with pytest.raises(ValueError, match="trainer state"):
        verify_checkpoints([dict(arm, initial_step=5)])


# --- Trained arm against the frozen base


def frozen_base_and_trained_run(tmp_path):
    """A finished base-only run, and a trained-arm run over the same frozen inputs."""
    from testing.benchmark import free_text_configuration

    base_config = free_text_configuration()
    base_root, out = tmp_path / "base-run", tmp_path / "trained-run"
    request = {
        "prompt_ids": [1, 2],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "model": {"precision": "bf16"},
        "max_model_len": 262144,
        "full_completion_ceiling": base_config["sampling"]["max_tokens"],
    }
    inputs = {
        "request.json": json.dumps(request).encode(),
        "private-truth.json": b'[{"issue":"known"}]',
        "judge-prompt.txt": b"judge template",
        "source-scope.json": b'[{"path":"A.sol","sha256":"0"}]',
    }
    for root in (base_root, out):
        root.mkdir()
        for name, data in inputs.items():
            (root / name).write_bytes(data)
    (base_root / "prepared.json").write_text(json.dumps(base_config))
    base_draws = [
        {"status": "graded", "exact_coverage": c, "coverage_including_partial": c}
        for c in (0.0, 0.25, 0.25, 0.5)
    ]
    for draw in base_draws:
        draw["completion_tokens"] = 2
    comparison = json.dumps({"output_format": "free_text", "draws": {"base": base_draws}})
    (base_root / "comparison.json").write_text(comparison)
    trained = {
        "label": "checkpoint-4",
        "initial_step": 4,
        "checkpoint": "/checkpoint",
        "initial_adapter_sha256": "b" * 64,
    }
    config = dict(
        base_config,
        run_id="trained",
        cases=[trained],
        generation_order=["checkpoint-4"],
        merged_model_path="/merged",
        base_reference={
            "run_root": str(base_root),
            "comparison_sha256": hashlib.sha256(comparison.encode()).hexdigest(),
        },
    )
    return config, out, base_root, base_draws


def test_frozen_base_reference_returns_its_graded_base_draws(tmp_path):
    from benchmark_evaluation import load_base_reference

    config, out, _base_root, base_draws = frozen_base_and_trained_run(tmp_path)
    assert load_base_reference(config, out) == base_draws


@pytest.mark.parametrize(
    "change",
    [
        lambda base, _out: (base / "comparison.json").write_text('{"draws": {}}'),
        lambda base, _out: (base / "private-truth.json").write_bytes(b"[]"),
        lambda base, _out: (base / "judge-prompt.txt").write_bytes(b"other judge"),
        lambda base, _out: (base / "source-scope.json").write_bytes(b"[]"),
        lambda _base, out: (out / "request.json").write_text(
            json.dumps({"prompt_ids_sha256": "0" * 64})
        ),
        lambda base, _out: (base / "prepared.json").write_text(
            json.dumps(
                {
                    **json.loads((base / "prepared.json").read_text()),
                    "sampling": {"seed": 1},
                }
            )
        ),
        lambda base, _out: (base / "prepared.json").write_text(
            json.dumps(
                {
                    **json.loads((base / "prepared.json").read_text()),
                    "template_options": {"reasoning_effort": "medium"},
                }
            )
        ),
    ],
)
def test_frozen_base_must_match_the_trained_evaluation(tmp_path, change):
    from benchmark_evaluation import load_base_reference

    config, out, base_root, _ = frozen_base_and_trained_run(tmp_path)
    change(base_root, out)
    with pytest.raises(ValueError, match="base"):
        load_base_reference(config, out)


def test_trained_arm_finishes_with_deltas_against_the_frozen_base(tmp_path):
    from benchmark_evaluation import finish_benchmark

    config, out, _base_root, base_draws = frozen_base_and_trained_run(tmp_path)
    free_text.validate_config(config)
    (out / "prepared.json").write_text(json.dumps(config))
    trained = config["cases"][0]
    (out / "checkpoint-4-config.json").write_text(
        json.dumps(dict(config, **trained, request=str(out / "request.json")))
    )
    version = f"step-4:{'b' * 64}"
    samples = json.dumps(
        [{"prompt_ids": [1, 2], "completion_ids": [3, 4], "weight_version": version}] * 4
    ).encode()
    (out / "checkpoint-4").mkdir()
    (out / "checkpoint-4/samples.json").write_bytes(samples)
    (out / "checkpoint-4-grading").mkdir()
    draws = [
        {"status": "graded", "exact_coverage": c, "coverage_including_partial": c}
        for c in (0.5, 0.5, 0.5, 0.5)
    ]
    for draw in draws:
        draw["completion_tokens"] = 2
    (out / "checkpoint-4-grading/metrics.json").write_text(
        json.dumps(
            {
                "draws": draws,
                "judge": config["benchmark_judge"],
                "input_sha256": {
                    "samples": hashlib.sha256(samples).hexdigest(),
                    "private_truth": hashlib.sha256(b'[{"issue":"known"}]').hexdigest(),
                    "judge_prompt": hashlib.sha256(b"judge template").hexdigest(),
                },
            }
        )
    )
    finish_benchmark(config, out)
    result = json.loads((out / "comparison.json").read_text())
    assert result["status"] == "complete"
    assert result["deltas"]["checkpoint-4"]["mean_exact_coverage"] == pytest.approx(0.25)
    assert result["draws"]["base"] == base_draws
    assert result["base_reference"] == config["base_reference"]
