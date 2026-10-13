"""Offline trainer boundary checks; no Torch, GPU, downloads or paid judge required."""

import copy
import json
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[5]


def test_preflight_requests_flat_ids_from_dictionary_default_tokenizer():
    from grpo import trainer

    class Tokenizer:
        def apply_chat_template(self, _messages, *, return_dict=True, **_options):
            ids = [11, 12, 13, 14, 15]
            return {"input_ids": ids, "attention_mask": [1] * 5} if return_dict else ids

    result = trainer.preflight(
        [{"row_id": "a", "prompt": [{"role": "user", "content": "hi"}]}],
        Tokenizer(),
        max_prompt_tokens=4,
        max_completion_tokens=8,
        context_limit=12,
    )
    assert result == [{"row_id": "a", "prompt_tokens": 5, "total_tokens": 13, "eligible": False}]


def test_preflight_counts_native_template_and_withholds_whole_oversize_rows():
    from grpo import trainer

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert options["enable_thinking"] is True
            assert options["reasoning_effort"] == "xhigh"
            assert options["add_generation_prompt"] is True
            return list(range(len(messages[0]["content"]) + 5))

    records = [
        {"row_id": "short", "prompt": [{"role": "user", "content": "a"}]},
        {"row_id": "long", "prompt": [{"role": "user", "content": "a" * 20}]},
    ]
    result = trainer.preflight(
        records, Tokenizer(), max_prompt_tokens=10, max_completion_tokens=8, context_limit=18
    )
    assert result == [
        {"row_id": "short", "prompt_tokens": 6, "total_tokens": 14, "eligible": True},
        {"row_id": "long", "prompt_tokens": 25, "total_tokens": 33, "eligible": False},
    ]


def test_trainer_parameters_disable_std_scaling_and_keep_full_completion_loss():
    from grpo import trainer

    plan = trainer.default_plan()
    parameters = trainer.trainer_parameters(plan, "output", max_steps=1)
    assert parameters["scale_rewards"] is False
    assert parameters["num_generations"] == 4
    assert parameters["per_device_train_batch_size"] == 1
    assert parameters["gradient_accumulation_steps"] == 4
    assert parameters["use_vllm"] is False
    assert parameters["bf16"] is True
    assert parameters["remove_unused_columns"] is False
    assert parameters["save_steps"] == 1
    assert parameters["max_steps"] == 1
    assert parameters["beta"] == 0.0
    assert parameters["mask_truncated_completions"] is False


def test_live_gate_requires_caps_access_baseline_and_thresholds(tmp_path):
    from grpo import trainer

    plan = trainer.default_plan()
    with pytest.raises(ValueError, match="access"):
        trainer.validate_live(plan, phase="baseline")
    plan["access"] = {
        "established": True,
        "billing_project": "company",
        "provider": "hyperstack",
        "machine_usd_cap": 30,
        "judge_usd_cap": 30,
        "wall_hours_cap": 6,
        "provider_budget_enforced": True,
        "judge_budget_enforced": True,
    }
    assert trainer.validate_live(plan, phase="baseline") is None
    with pytest.raises(ValueError, match="baseline"):
        trainer.validate_live(plan, phase="step")
    plan["baseline"] = {"path": str(tmp_path / "missing.json"), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="baseline"):
        trainer.validate_live(plan, phase="pilot")


def test_live_gate_accepts_revised_budget_and_still_requires_untuned_baseline():
    from grpo import trainer

    plan = _revised_budget_plan()
    assert trainer.validate_live(plan, phase="baseline") is None
    with pytest.raises(ValueError, match="baseline"):
        trainer.validate_live(plan, phase="step")


def test_explicit_training_first_scope_allows_fit_and_pilot_but_keeps_caps_and_comparison_gate():
    from grpo import trainer

    plan = _revised_budget_plan()
    plan["training_first"] = {
        "authorized": True,
        "source": "User requested first fine-tuning instead of repeating prior benchmarks",
    }
    assert trainer.validate_live(plan, phase="step") is None
    assert trainer.validate_live(plan, phase="pilot") is None
    with pytest.raises(ValueError, match="baseline"):
        trainer.validate_live(plan, phase="tuned")
    plan["access"]["judge_usd_cap"] = 0
    with pytest.raises(ValueError, match="judge_usd_cap"):
        trainer.validate_live(plan, phase="step")


@pytest.mark.parametrize(
    "authorization", ({"authorized": True}, {"authorized": False, "source": "user"})
)
def test_training_first_gate_requires_explicit_recorded_user_direction(authorization):
    from grpo import trainer

    plan = _revised_budget_plan()
    plan["training_first"] = authorization
    with pytest.raises(ValueError, match="baseline"):
        trainer.validate_live(plan, phase="step")


def test_live_gate_accepts_explicit_manual_provider_stop_with_local_judge_guard():
    from grpo import trainer

    plan = _revised_budget_plan()
    plan["access"].update(
        provider_budget_enforced=False,
        judge_budget_enforced=False,
        manual_provider_stop_authorized=True,
        local_experiment_stop={"timer": "audit-grpo-experiment-deadline.timer"},
        local_judge_budget={"path": "/results/judge-budget.json"},
    )
    assert trainer.validate_live(plan, phase="baseline") is None
    plan["access"]["manual_provider_stop_authorized"] = False
    with pytest.raises(ValueError, match="budget"):
        trainer.validate_live(plan, phase="baseline")


@pytest.mark.parametrize("key", ("machine_usd_cap", "judge_usd_cap", "wall_hours_cap"))
@pytest.mark.parametrize("value", (None, "48", True, False, 0, -1, float("inf"), float("nan")))
def test_live_gate_rejects_nonfinite_nonpositive_or_nonnumeric_budget(key, value):
    from grpo import trainer

    plan = _revised_budget_plan()
    plan["access"].update(machine_usd_cap=30, judge_usd_cap=30, wall_hours_cap=6)
    plan["access"][key] = value
    with pytest.raises(ValueError, match=key):
        trainer.validate_live(plan, phase="baseline")


def test_reward_callback_rejects_trainer_token_or_row_reordering():
    from grpo import trainer

    callback = trainer.RewardCallback()
    callback.pending = [
        {"row_id": "r", "completion_ids": [3], "reward": 1.2},
        {"row_id": "r", "completion_ids": [4], "reward": 0.95},
    ]
    with pytest.raises(ValueError, match="join"):
        callback(completion_ids=[[4], [3]], row_id=["r", "r"])
    assert callback(completion_ids=[[3], [4]], row_id=["r", "r"]) == [1.2, 0.95]
    with pytest.raises(ValueError, match="join"):
        callback(completion_ids=[[3], [4]], row_id=["r", "r"])


def test_revision_and_mutated_scalar_fail_preparation():
    from grpo import trainer

    plan = trainer.default_plan()
    bad = copy.deepcopy(plan)
    bad["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        trainer.validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["training"]["scale_rewards"] = "group"
    with pytest.raises(ValueError, match="scaling"):
        trainer.validate_plan(bad)
    assert (
        len(
            json.loads((WORKSPACE / "data/policy/train.jsonl").read_text().splitlines()[0])[
                "messages"
            ]
        )
        == 2
    )


def test_rollout_hook_handles_pinned_trl_repeated_prompt_slice(monkeypatch):
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    from grpo import trainer

    module = ModuleType("trl.models")
    module.unwrap_model_for_generation = lambda *_args: nullcontext("active-policy")
    monkeypatch.setitem(__import__("sys").modules, "trl.models", module)
    prompt = [{"role": "user", "content": "code only"}]

    class Sampler:
        def __call__(self, identity, messages, count):
            assert self.model == "active-policy"
            assert identity == "r" and messages == prompt and count == 4
            return [
                {"row_id": "r", "prompt_ids": [1], "completion_ids": [i], "logprobs": [-0.1]}
                for i in (7, 8, 9, 10)
            ]

    callback = trainer.RewardCallback()
    hook = trainer.RolloutHook(
        {"train": [{"row_id": "r", "prompt": prompt}]},
        Sampler(),
        lambda _group, _split: [1.2, 0.95, 0, -2],
        callback,
        trainer.default_plan(),
    )
    output = hook(
        [prompt, prompt, prompt, prompt],
        SimpleNamespace(model_wrapped="wrapped", accelerator="accelerator"),
    )
    assert output["completion_ids"] == [[7], [8], [9], [10]]
    assert callback(completion_ids=output["completion_ids"], row_id=["r"] * 4) == [1.2, 0.95, 0, -2]


def test_expired_compute_window_stops_before_more_work():
    from grpo import trainer

    with pytest.raises(TimeoutError, match="window"):
        trainer.remaining_seconds(0)


def test_feasibility_rollout_stops_before_loss_when_all_rewards_are_identical(monkeypatch):
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    from grpo import trainer

    module = ModuleType("trl.models")
    module.unwrap_model_for_generation = lambda *_args: nullcontext("active-policy")
    monkeypatch.setitem(__import__("sys").modules, "trl.models", module)
    prompt = [{"role": "user", "content": "code only"}]

    class Sampler:
        def __call__(self, *_args):
            return [
                {"row_id": "r", "prompt_ids": [1], "completion_ids": [i], "logprobs": [-0.1]}
                for i in (7, 8, 9, 10)
            ]

    plan = trainer.default_plan()
    plan["feasibility_only"] = {"authorized": True}
    callback = trainer.RewardCallback()
    hook = trainer.RolloutHook(
        {"train": [{"row_id": "r", "prompt": prompt}]},
        Sampler(),
        lambda _group, _split: [-2, -2, -2, -2],
        callback,
        plan,
    )
    with pytest.raises(RuntimeError, match="reward contrast"):
        hook([prompt] * 4, SimpleNamespace(model_wrapped="wrapped", accelerator="accelerator"))
    assert callback.pending == []


def test_generation_device_uses_gpu_after_offloaded_cpu_embeddings():
    from types import SimpleNamespace

    from grpo import trainer

    cpu, gpu = SimpleNamespace(type="cpu"), SimpleNamespace(type="cuda")
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=cpu), SimpleNamespace(device=gpu)])
    )
    assert trainer.generation_device(model) is gpu
    model.parameters = lambda: iter([SimpleNamespace(device=cpu)])
    with pytest.raises(ValueError, match="CUDA"):
        trainer.generation_device(model)


def test_rollout_hook_check_accepts_forwarding_constructor_wrapper():
    from grpo import trainer

    class HookTrainer:
        def __init__(self, *, rollout_func):
            self.rollout_func = rollout_func

    class WrappedTrainer(HookTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    hook = object()
    assert trainer.validate_rollout_hook(WrappedTrainer) is None
    assert WrappedTrainer(rollout_func=hook).rollout_func is hook


def test_rollout_hook_check_rejects_wrapper_without_supported_constructor():
    from grpo import trainer

    class NoHookTrainer:
        def __init__(self, *, model):
            self.model = model

    class WrappedTrainer(NoHookTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    with pytest.raises(ValueError, match="rollout_func"):
        trainer.validate_rollout_hook(WrappedTrainer)


def test_rollout_hook_check_rejects_constructor_that_cannot_forward_hook():
    from grpo import trainer

    class HookTrainer:
        def __init__(self, *, rollout_func):
            self.rollout_func = rollout_func

    class NoForwardTrainer(HookTrainer):
        def __init__(self, *, model):
            self.model = model

    with pytest.raises(ValueError, match="rollout_func"):
        trainer.validate_rollout_hook(NoForwardTrainer)


def _revised_budget_plan():
    from grpo import trainer

    plan = trainer.default_plan()
    plan["access"] = {
        "established": True,
        "billing_project": "company",
        "machine_usd_cap": 120.48,
        "judge_usd_cap": 150,
        "wall_hours_cap": 48,
        "provider_budget_enforced": True,
        "judge_budget_enforced": True,
    }
    return plan


def test_judge_bridge_worker_preserves_virtualenv_site_packages(tmp_path):
    import subprocess
    import sys
    import venv

    from grpo.trainer import JudgeBridge

    environment = tmp_path / "judge-environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    packages = (
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (packages / "pilot_worker_marker.py").write_text('value = "judge-venv-visible"\n')
    bridge = JudgeBridge(tmp_path, environment / "bin/python", tmp_path / "out")
    child = subprocess.run(
        [
            str(bridge.interpreter),
            "-c",
            "import pilot_worker_marker; print(pilot_worker_marker.value)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "judge-venv-visible"


def test_judge_worker_preflight_exposes_import_failure_before_generation(tmp_path):
    import sys

    from grpo import trainer

    package = tmp_path / "code/backend/scripts/finetuning/training"
    package.mkdir(parents=True)
    for directory in [package, package.parent, package.parent.parent]:
        (directory / "__init__.py").write_text("")
    (package / "judge_worker.py").write_text("import missing_pilot_worker_dependency\n")
    with pytest.raises(RuntimeError, match="missing_pilot_worker_dependency"):
        trainer.validate_judge_worker(tmp_path, Path(sys.executable))


def test_judge_worker_preflight_uses_worker_help_without_paid_execution(tmp_path):
    import sys

    from grpo import trainer

    package = tmp_path / "code/backend/scripts/finetuning/training"
    package.mkdir(parents=True)
    for directory in [package, package.parent, package.parent.parent]:
        (directory / "__init__.py").write_text("")
    (package / "judge_worker.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.parse_args()\n"
        'raise RuntimeError("Paid execution was entered")\n'
    )
    assert trainer.validate_judge_worker(tmp_path, Path(sys.executable)) is None


def test_live_startup_reports_worker_failure_before_importing_gpu_stack(tmp_path):
    import sys

    from grpo import trainer

    plan = _revised_budget_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    policy = tmp_path / "data/policy"
    policy.mkdir(parents=True)
    for split in ["train", "calibration", "evaluation"]:
        (policy / f"{split}.jsonl").write_text(
            json.dumps({"row_id": split, "messages": [{"role": "user", "content": "code"}]}) + "\n"
        )
    package = tmp_path / "code/backend/scripts/finetuning/training"
    package.mkdir(parents=True)
    for directory in [package, package.parent, package.parent.parent]:
        (directory / "__init__.py").write_text("")
    (package / "judge_worker.py").write_text("import missing_pilot_worker_dependency\n")
    with pytest.raises(RuntimeError, match="missing_pilot_worker_dependency"):
        trainer.main(
            [
                "baseline",
                "--run-live",
                "--workspace",
                str(tmp_path),
                "--plan",
                str(plan_path),
                "--judge-python",
                sys.executable,
                "--out",
                str(tmp_path / "output"),
            ]
        )


def test_versioned_sampling_profile_reaches_prompt_preflight_and_sampling_record():
    from grpo import trainer

    class ProfileTokenizer:
        def apply_chat_template(self, _messages, **options):
            return [1, 2, 3] if options["reasoning_effort"] == "medium" else [1, 2, 3, 4, 5]

    plan = trainer.default_plan()
    plan["version"] = "feasibility-schema-medium-8k-v1"
    plan["sampling_profile"] = {"id": "schema-medium-8k-v1", "policy_reasoning_effort": "medium"}
    plan["training"]["max_completion_tokens"] = 8192
    options = trainer.policy_template_options(plan)
    fit = trainer.preflight(
        [{"row_id": "scope", "prompt": [{"role": "user", "content": "code"}]}],
        ProfileTokenizer(),
        max_prompt_tokens=4,
        max_completion_tokens=8,
        context_limit=12,
        template_options=options,
    )
    assert fit == [{"row_id": "scope", "prompt_tokens": 3, "total_tokens": 11, "eligible": True}]
    assert trainer.sampling(plan)["template_options"]["reasoning_effort"] == "medium"
    assert trainer.sampling(plan)["max_completion_tokens"] == 8192


def test_sampling_profile_rejects_effort_unsupported_by_pinned_native_template():
    from grpo import trainer

    plan = trainer.default_plan()
    plan["sampling_profile"] = {"id": "unsupported-effort", "policy_reasoning_effort": "high"}
    with pytest.raises(ValueError, match="reasoning effort"):
        trainer.validate_plan(plan)


def test_phase_event_journal_retains_prior_measurements(tmp_path):
    from grpo.trainer import record_event

    path = tmp_path / "events.jsonl"
    record_event(path, "judge_start", group="group-0001")
    first = path.read_bytes()
    record_event(path, "judge_end", group="group-0001", returncode=0)
    assert path.read_bytes().startswith(first)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["judge_start", "judge_end"]
    assert events[1]["returncode"] == 0
    assert events[0]["monotonic_seconds"] <= events[1]["monotonic_seconds"]


def test_sampling_record_changes_when_output_schema_contract_changes():
    from grpo.trainer import default_plan, sampling

    first = default_plan()
    first["sampling_profile"] = {
        "id": "medium-8k",
        "policy_reasoning_effort": "medium",
        "output_schema_sha256": "a" * 64,
    }
    changed = copy.deepcopy(first)
    changed["sampling_profile"]["output_schema_sha256"] = "b" * 64
    assert sampling(first) != sampling(changed)
