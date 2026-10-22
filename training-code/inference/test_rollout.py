"""Exercise artifact capture with real SDK data and a CPU-only engine boundary."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput, SamplingMask

from rollout import run_rollout


@pytest.mark.parametrize("mode", ["rollout", "evaluate", "base", "merged", "native_base"])
def test_versioned_groups_capture_actual_sdk_tokens(tmp_path, monkeypatch, mode):
    phase = "evaluate" if mode in ("base", "merged", "native_base") else mode
    configuration = "fixtures/evaluation.json" if phase == "evaluate" else "fixtures/cycle.json"
    config = json.loads((Path(__file__).parents[1] / configuration).read_text())
    step = 2 if phase == "evaluate" else 1
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    save_file(
        {
            "base_model.model.model.layers.0.x.lora_A.weight": torch.ones(16, 2),
            "base_model.model.model.layers.0.x.lora_B.weight": torch.ones(2, 16),
        },
        str(checkpoint / "adapter_model.safetensors"),
    )
    (checkpoint / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    config["checkpoint"] = str(checkpoint)
    config["initial_adapter_sha256"] = hashlib.sha256(
        (checkpoint / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    if phase == "evaluate":
        config["initial_step"] = step
        config["cases"] = [
            dict(
                label="test",
                **{
                    name: config[name]
                    for name in ("initial_step", "checkpoint", "initial_adapter_sha256")
                },
            )
        ]
    if mode == "native_base":
        from testing.benchmark import native_context_configuration

        config = native_context_configuration()
        config.update(initial_step=0, label="base", prompt_token_count=2)
        config["sampling"]["max_tokens"] = 262142
    if mode == "base":
        config.pop("checkpoint")
        config.pop("initial_adapter_sha256")
        config["initial_step"] = 0
        config["label"] = "base"
        config["cases"] = [{"label": "base", "initial_step": 0}]
    if mode == "merged":
        merged = tmp_path / "merged"
        merged.mkdir()
        config["merged_model_path"] = str(merged)
        (merged / "MERGE_MANIFEST.json").write_text(
            json.dumps(
                {
                    "status": "merged_and_verified",
                    "base_revision": config["model_revision"],
                    "adapter_sha256": config["initial_adapter_sha256"],
                    "stored_weight_precision": "bfloat16",
                }
            )
        )
    request = {
        "row_id": "test-row",
        "prompt_ids": [1, 2],
        "prompt_ids_sha256": hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest(),
        "model": {
            "id": config["model_id"],
            "revision": config["model_revision"],
            "precision": "bf16",
        },
        "full_completion_ceiling": config["sampling"]["max_tokens"],
        "max_model_len": config["engine"]["max_model_len"],
    }
    config["request"] = str(tmp_path / "request.json")
    Path(config["request"]).write_text(json.dumps(request))
    outputs = [
        CompletionOutput(
            index=index,
            text="test",
            token_ids=[4, 5],
            cumulative_logprob=-2 * math.log(2),
            logprobs=[{4: Logprob(-math.log(2), 1)}, {5: Logprob(-math.log(2), 1)}],
            finish_reason="stop",
            stop_reason=None,
            sampling_mask=None if phase == "evaluate" else SamplingMask([[4, 6], [5, 6]]),
        )
        for index in range(config["sampling"]["n"])
    ]
    response = RequestOutput("request", None, [1, 2], None, outputs, True)

    class EngineBoundary:
        # Loading the actual H100 engine is the external boundary; SDK artifacts,
        # adapter mapping, validation and file persistence remain real in this test.
        def __init__(self, **settings):
            assert settings["return_sampling_mask"] is (phase != "evaluate")
            if mode == "native_base":
                assert settings["enable_lora"] is False
                assert "worker_extension_cls" not in settings
            if mode == "merged":
                assert settings["model"] == str(merged)
                assert settings["tokenizer"] == config["model_path"]
                assert settings["enable_lora"] is False
                assert "worker_extension_cls" not in settings
                assert "lora_dtype" not in settings

        def collective_rpc(self, method, args=()):
            if mode in ("base", "merged", "native_base"):
                raise AssertionError("Unchanged base must never load an adapter")
            if method == "load_adapter":
                return [True]
            if method == "verify_adapter":
                return [{"cast_transfer_verified": True, "source_tensor_count": 2}]
            raise ValueError("Corrected replay must not install native sampler hooks")

        def generate(self, prompts, params, *, use_tqdm, **request_options):
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            assert (params.n, params.max_tokens, params.top_k, params.top_p) == (
                (1, 262142, 20, 0.95) if mode == "native_base" else (4, 16384, 20, 0.95)
            )
            if mode in ("base", "native_base"):
                assert request_options["lora_request"] is None
            elif mode == "merged":
                assert not request_options
            else:
                assert request_options["lora_request"].lora_int_id == step
            return [response]

        def get_tokenizer(self):
            return SimpleNamespace(decode=lambda ids, **options: 'reason</think>{"findings":[]}')

    monkeypatch.setattr("vllm.LLM", EngineBoundary)
    out = tmp_path / "rollout"
    receipt = run_rollout(phase, config, out)
    samples = json.loads((out / "samples.json").read_text())
    assert receipt["status"] == "versioned_rollout_saved"
    assert receipt["sampler_backend"] == "vllm_upstream"
    if mode == "merged":
        assert receipt["weight_representation"] == "merged_bf16"
        assert receipt["numerical_agreement_verified"] is False
    assert len(samples) == config["sampling"]["n"]
    for sample in samples:
        if mode == "merged":
            assert sample["weight_representation"] == "merged_bf16"
        assert sample["completion_ids"] == [4, 5]
        assert sample["sampling_support_ids"] == (None if phase == "evaluate" else [[4, 6], [5, 6]])
        assert sample["logprobs"] == pytest.approx([-0.69314718056, -0.69314718056])
        assert sample["raw_logprobs"] is None
        if mode in ("base", "native_base"):
            assert sample["weight_version"] == f"base:{config['model_revision']}"
        else:
            assert sample["weight_version"].startswith(f"step-{step}:")


# --- Every project's arm in one engine


def merged_arm(tmp_path, name, prompt_ids):
    """One project's merged-weight evaluation arm with its own request."""
    config = json.loads((Path(__file__).parents[1] / "fixtures/evaluation.json").read_text())
    adapter = b"merged adapter"
    checkpoint = tmp_path / "checkpoint-2"
    if not checkpoint.exists():
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_bytes(adapter)
        merged = tmp_path / "merged"
        merged.mkdir()
        manifest = {
            "status": "merged_and_verified",
            "base_revision": config["model_revision"],
            "adapter_sha256": hashlib.sha256(adapter).hexdigest(),
            "stored_weight_precision": "bfloat16",
        }
        (merged / "MERGE_MANIFEST.json").write_text(json.dumps(manifest))
    case = {
        "label": "checkpoint-2",
        "initial_step": 2,
        "checkpoint": str(checkpoint),
        "initial_adapter_sha256": hashlib.sha256(adapter).hexdigest(),
    }
    request = {
        "row_id": name,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
        "model": {
            "id": config["model_id"],
            "revision": config["model_revision"],
            "precision": "bf16",
        },
        "full_completion_ceiling": config["sampling"]["max_tokens"],
        "max_model_len": config["engine"]["max_model_len"],
    }
    (tmp_path / f"{name}-request.json").write_text(json.dumps(request))
    config.update(
        case,
        cases=[case],
        merged_model_path=str(tmp_path / "merged"),
        request=str(tmp_path / f"{name}-request.json"),
    )
    path = tmp_path / f"{name}-config.json"
    path.write_text(json.dumps(config))
    return {"config": str(path), "out": str(tmp_path / name / "checkpoint-2")}


def engine_output(prompt_ids, n, token):
    outputs = [
        CompletionOutput(
            index=index,
            text="audit",
            token_ids=[token, token + 1],
            cumulative_logprob=-2 * math.log(2),
            logprobs=[{token: Logprob(-math.log(2), 1)}, {token + 1: Logprob(-math.log(2), 1)}],
            finish_reason="stop",
            stop_reason=None,
        )
        for index in range(n)
    ]
    return RequestOutput(f"request-{token}", None, prompt_ids, None, outputs, True)


def test_every_project_is_sampled_in_one_engine_with_its_own_prompt(tmp_path, monkeypatch):
    from rollout import run_evaluation_suite

    prompts = {"munchables": [1, 2], "vultisig": [7, 8, 9], "phi": [3]}
    entries = [merged_arm(tmp_path, name, ids) for name, ids in prompts.items()]
    starts = []

    class EngineBoundary:
        def __init__(self, **settings):
            starts.append(settings)
            assert settings["model"] == str(tmp_path / "merged")
            assert settings["enable_lora"] is False

        def collective_rpc(self, method, args=()):
            raise AssertionError("Merged weights never load an adapter")

        def generate(self, requests, params, *, use_tqdm, **request_options):
            assert requests == [{"prompt_token_ids": ids} for ids in prompts.values()]
            assert [p.seed for p in params] == [3407] * 3 and not request_options
            return [
                engine_output(ids, p.n, 10 * (index + 1))
                for index, (ids, p) in enumerate(zip(prompts.values(), params, strict=True))
            ]

        def get_tokenizer(self):
            return SimpleNamespace(decode=lambda ids, **options: "reasoning</think>the audit")

    monkeypatch.setattr("vllm.LLM", EngineBoundary)
    receipt_path = tmp_path / "rollout-receipt.json"
    suite = run_evaluation_suite({"arms": entries, "receipt": str(receipt_path)})
    assert len(starts) == 1
    assert suite["status"] == "versioned_rollout_saved"
    assert suite["completion_tokens"] == 3 * 4 * 2
    assert json.loads(receipt_path.read_text())["arms"] == [e["out"] for e in entries]
    for index, (name, ids) in enumerate(prompts.items()):
        out = tmp_path / name / "checkpoint-2"
        samples = json.loads((out / "samples.json").read_text())
        assert [s["row_id"] for s in samples] == [name] * 4
        assert {tuple(s["prompt_ids"]) for s in samples} == {tuple(ids)}
        assert samples[0]["completion_ids"] == [10 * (index + 1), 10 * (index + 1) + 1]
        assert samples[0]["weight_representation"] == "merged_bf16"
        receipt = json.loads((out / "receipt.json").read_text())
        assert receipt["status"] == "versioned_rollout_saved"
        assert receipt["shared_generation_receipt"] == str(receipt_path)
        assert "generation_seconds" not in receipt
        assert json.loads((out / "batch.json").read_text())["shared_generation_arms"] == 3


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["engine"].update(max_num_seqs=8),
        lambda c: c["sampling"].update(temperature=0.6),
        lambda c: c.update(merged_model_path=None),
    ],
)
def test_arms_that_differ_beyond_their_prompt_cannot_share_an_engine(tmp_path, change):
    from rollout import resolve_arm, validate_suite_arms

    entries = [merged_arm(tmp_path, name, [1, 2]) for name in ("munchables", "phi")]
    arms = [resolve_arm("evaluate", json.loads(Path(e["config"]).read_text())) for e in entries]
    validate_suite_arms(arms)
    changed = json.loads(Path(entries[1]["config"]).read_text())
    change(changed)
    arms[1] = dict(arms[1], config=changed, runtime_adapter=not changed["merged_model_path"])
    with pytest.raises(ValueError, match=r"engine|sampling|adapter"):
        validate_suite_arms(arms)
