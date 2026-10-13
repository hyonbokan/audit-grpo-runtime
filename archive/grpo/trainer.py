"""Offline preparation and explicitly enabled BF16 LoRA GRPO pilot stages."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import subprocess
import time
from pathlib import Path

from grpo.samples import capture_sample, write_new
from grpo.sampling import SamplingRecorder

WORKSPACE = Path(__file__).resolve().parents[5]
TEMPLATE_OPTIONS = {
    "add_generation_prompt": True,
    "enable_thinking": True,
    "reasoning_effort": "xhigh",
    "preserve_thinking": True,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "preflight", "baseline", "step", "pilot", "tuned")
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-python", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--run-live", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text()) if args.plan else default_plan()
    validate_plan(plan)
    if plan.get("judge") is not None and plan["judge"] != judging_contract(args.workspace):
        raise ValueError("Live judge contract differs from reviewed plan")
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / "plan.json", plan)
    records = load_policy(args.workspace)
    if args.phase == "prepare":
        write_new(
            args.out / "prepared.json",
            {
                "status": "offline_only",
                "rows": {split: [row["row_id"] for row in rows] for split, rows in records.items()},
                "policy_calls": 0,
                "judge_calls": 0,
                "optimizer_steps": 0,
            },
        )
        return 0
    if args.phase != "preflight":
        if not args.run_live:
            raise ValueError("Live stages require --run-live and established access/caps")
        validate_live(plan, phase=args.phase)
        if args.judge_python is None:
            raise ValueError("Separate backend/judge interpreter required")
        if args.adapter and args.phase != "tuned":
            raise ValueError("Saved adapter is only used for tuned evaluation")
        validate_judge_worker(args.workspace, args.judge_python)
        import unsloth  # noqa: F401
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["id"], revision=plan["model"]["revision"], local_files_only=not args.run_live
    )
    fits = {
        split: preflight(
            rows,
            tokenizer,
            max_prompt_tokens=plan["training"]["max_prompt_tokens"],
            max_completion_tokens=plan["training"]["max_completion_tokens"],
            context_limit=plan["model"]["context_limit"],
            template_options=policy_template_options(plan),
        )
        for split, rows in records.items()
    }
    write_new(args.out / "preflight.json", fits)
    write_new(
        args.out / "tokenizer.json",
        {
            "chat_template_sha256": digest(tokenizer.chat_template.encode()),
            "revision": plan["model"]["revision"],
            "template_options": policy_template_options(plan),
            "vocabulary_sha256": digest(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()),
        },
    )
    if args.phase == "preflight":
        return 0
    return run_gpu(args, plan, records, fits, tokenizer)


def default_plan():
    return {
        "version": "grpo-pilot-v2-2026-10-07",
        "model": {
            "id": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "precision": "bf16",
            "context_limit": 163840,
        },
        "training": {
            "preferred_stack": "unsloth",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 5e-6,
            "num_generations": 4,
            "max_steps": 8,
            "max_prompt_tokens": 98304,
            "max_completion_tokens": 65536,
            "gradient_accumulation_steps": 4,
            "scale_rewards": False,
        },
        "access": {"established": False},
        "baseline": None,
        "thresholds": None,
        "baseline_row_ids": [
            "2022-10-paladin/all-scope",
            "2023-03-wenwin/all-scope",
            "2024-06-krystal-defi/all-scope",
        ],
        "fit_train_row_ids": ["2022-12-caviar/all-scope"],
        "package_versions": {
            "unsloth": "2026.10.1",
            "unsloth_zoo": "2026.10.1",
            "trl": "1.13.0",
            "transformers": "5.17.0",
            "peft": "0.21.2",
            "torch": "2.13.0",
            "datasets": "4.8.5",
            "accelerate": "1.15.0",
        },
    }


def validate_plan(plan):
    policy_template_options(plan)
    if not re.fullmatch(r"[0-9a-f]{40}", plan["model"]["revision"]):
        raise ValueError("Immutable model revision required")
    training = plan["training"]
    if training["scale_rewards"] is not False:
        raise ValueError("Selected reward disables standard-deviation scaling")
    if plan["model"]["precision"] != "bf16" or training["num_generations"] != 4:
        raise ValueError("First pilot requires BF16 and four complete draws")
    if training["gradient_accumulation_steps"] != 4 or not 1 <= training["max_steps"] <= 8:
        raise ValueError("Pilot step/batch cap differs")
    if (
        training["max_prompt_tokens"] + training["max_completion_tokens"]
        > plan["model"]["context_limit"]
    ):
        raise ValueError("Full prompt plus completion exceeds context")


def validate_live(plan, *, phase):
    access = plan["access"]
    if not access.get("established") or not access.get("billing_project"):
        raise ValueError("Machine access and company billing project must be established")
    for key in ("machine_usd_cap", "judge_usd_cap", "wall_hours_cap"):
        if (
            type(access.get(key)) not in (int, float)
            or not math.isfinite(access[key])
            or access[key] <= 0
        ):
            raise ValueError(f"Missing/invalid approved {key}")
    provider_control = access.get("provider_budget_enforced") or (
        access.get("manual_provider_stop_authorized") is True
        and (access.get("local_experiment_stop") or {}).get("timer")
    )
    judge_control = access.get("judge_budget_enforced") or (
        access.get("local_judge_budget") or {}
    ).get("path")
    if not provider_control or not judge_control:
        raise ValueError("Provider and judge account budget enforcement must be established")
    training_first = plan.get("training_first") or {}
    source = training_first.get("source")
    if (
        phase in ("step", "pilot")
        and training_first.get("authorized") is True
        and isinstance(source, str)
        and source.strip()
    ):
        return
    if phase in ("step", "pilot", "tuned"):
        baseline = plan.get("baseline")
        if not baseline or not Path(baseline["path"]).is_file():
            raise ValueError("Frozen same-checkpoint baseline required before weights change")
        if digest(Path(baseline["path"]).read_bytes()) != baseline["sha256"]:
            raise ValueError("Frozen baseline hash differs")
        measured = json.loads(Path(baseline["path"]).read_text())
        if measured.get("judge") != plan.get("judge"):
            raise ValueError("Frozen baseline judge contract differs")
        if measured["model"] != plan["model"] or measured["optimizer_steps"] != 0:
            raise ValueError("Baseline checkpoint/precision differs")
        if measured["sampling"] != sampling(plan):
            raise ValueError("Baseline generation budget differs")
        thresholds = plan.get("thresholds") or {}
        required = ("min_recall_delta", "max_validity_drop", "max_quality_drop", "max_volume_ratio")
        if any(key not in thresholds or not math.isfinite(thresholds[key]) for key in required):
            raise ValueError("Freeze numerical success/regression thresholds before training")


def validate_rollout_hook(trainer_class):
    for cls in trainer_class.__mro__:
        parameters = inspect.signature(inspect.unwrap(cls.__init__)).parameters
        if "rollout_func" in parameters:
            return
        if not any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()):
            break
    raise ValueError("Installed patched trainer lacks required rollout_func")


def trainer_parameters(plan, output, *, max_steps):
    train = plan["training"]
    return {
        "output_dir": str(output),
        "learning_rate": train["learning_rate"],
        "max_steps": max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": 4,
        "generation_batch_size": 4,
        "max_completion_length": train["max_completion_tokens"],
        "scale_rewards": False,
        "beta": 0.0,
        "loss_type": "dapo",
        "mask_truncated_completions": False,
        "use_vllm": False,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "logging_steps": 1,
        "save_steps": 1,
        "save_total_limit": 3,
        "seed": 3407,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }


def load_policy(workspace):
    splits = {}
    identities = set()
    for split in ("train", "calibration", "evaluation"):
        rows = [
            json.loads(line)
            for line in (workspace / f"data/policy/{split}.jsonl").read_text().splitlines()
        ]
        for row in rows:
            if set(row) != {"row_id", "messages"} or row["row_id"] in identities:
                raise ValueError("Private fields or duplicate row in policy records")
            identities.add(row["row_id"])
        splits[split] = [{"row_id": row["row_id"], "prompt": row["messages"]} for row in rows]
    return splits


def preflight(
    records,
    tokenizer,
    *,
    max_prompt_tokens,
    max_completion_tokens,
    context_limit,
    template_options=None,
):
    result = []
    for record in records:
        ids = prompt_token_ids(tokenizer, record["prompt"], template_options=template_options)
        total = len(ids) + max_completion_tokens
        result.append(
            {
                "row_id": record["row_id"],
                "prompt_tokens": len(ids),
                "total_tokens": total,
                "eligible": len(ids) <= max_prompt_tokens and total <= context_limit,
            }
        )
    return result


def prompt_token_ids(tokenizer, messages, *, template_options=None):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        **(TEMPLATE_OPTIONS if template_options is None else template_options),
    )


class RewardCallback:
    """Verify TRL's decoded view still refers to the exact samples already graded."""

    __name__ = "selected_scalar"

    def __init__(self):
        self.pending = []

    def __call__(self, completion_ids, row_id, **_kwargs):
        if len(completion_ids) != len(self.pending) or len(row_id) != len(self.pending):
            raise ValueError("Trainer/sample join differs")
        if any(
            ids != sample["completion_ids"] or identity != sample["row_id"]
            for ids, identity, sample in zip(completion_ids, row_id, self.pending, strict=True)
        ):
            raise ValueError("Trainer/sample token or row join differs")
        values = [sample["reward"] for sample in self.pending]
        self.pending = []
        return values


def digest(data):
    return hashlib.sha256(data).hexdigest()


def judging_contract(workspace):
    data = (workspace / "configs/reward_contract.json").read_bytes()
    judge = json.loads(data)["judge"]
    return {
        "model": judge["model"],
        "reasoning_effort": judge["reasoning_effort"],
        "reward_contract_sha256": digest(data),
    }


def sampling(plan):
    result = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "draws_per_row": 4,
        "max_completion_tokens": plan["training"]["max_completion_tokens"],
        "template_options": policy_template_options(plan),
    }
    if plan.get("sampling_profile"):
        result.update(
            profile_id=plan["sampling_profile"]["id"],
            output_schema_sha256=plan["sampling_profile"].get("output_schema_sha256"),
        )
    return result


def policy_template_options(plan):
    options = dict(TEMPLATE_OPTIONS)
    profile = plan.get("sampling_profile")
    if profile is not None:
        effort = profile.get("policy_reasoning_effort")
        if not profile.get("id") or effort not in ("low", "medium", "xhigh"):
            raise ValueError("Versioned profile needs a supported policy reasoning effort")
        options["reasoning_effort"] = effort
    return options


def remaining_seconds(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Approved compute window expired")
    return remaining


def validate_judge_worker(workspace, interpreter):
    result = subprocess.run(
        [str(interpreter.absolute()), "-m", "grpo.judge_worker", "--help"],
        cwd=workspace / "code/backend",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Judge worker startup failed: {result.stderr}")


def record_event(path, event, **fields):
    if path is None:
        return
    row = {
        "event": event,
        "wall_seconds": time.time(),
        "monotonic_seconds": time.monotonic(),
        **fields,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def make_timing_callback(deadline, path):
    from transformers import TrainerCallback

    class TimingCallback(TrainerCallback):
        def on_step_begin(self, _args, state, _control, **_kwargs):
            remaining_seconds(deadline)
            record_event(path, "step_begin", global_step=state.global_step)

        def on_substep_end(self, _args, state, _control, **_kwargs):
            record_event(path, "substep_end", global_step=state.global_step)

        def on_step_end(self, _args, state, _control, **_kwargs):
            record_event(path, "step_end", global_step=state.global_step)

        def on_save(self, _args, state, _control, **_kwargs):
            record_event(path, "checkpoint_saved", global_step=state.global_step)

    return TimingCallback()


# --- GPU stages (imports deferred; preparation never loads weights)


def run_gpu(args, plan, records, fits, tokenizer):
    deadline = time.monotonic() + plan["access"]["wall_hours_cap"] * 3600
    import importlib.metadata

    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    actual = {name: importlib.metadata.version(name) for name in plan["package_versions"]}
    write_new(args.out / "packages.json", actual)
    if actual != plan["package_versions"]:
        raise ValueError("GPU environment differs from candidate pins; review a new plan")
    validate_rollout_hook(GRPOTrainer)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 device required")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("First fit probe uses one process; DDP is not memory sharding")
    from unsloth import FastModel

    model, _ = FastModel.from_pretrained(
        model_name=plan["model"]["id"],
        revision=plan["model"]["revision"],
        use_exact_model_name=True,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        fast_inference=False,
        text_only=True,
        return_logits=True,
        max_seq_length=plan["model"]["context_limit"],
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    elif args.phase in ("step", "pilot"):
        model = FastModel.get_peft_model(
            model,
            r=plan["training"]["lora_rank"],
            lora_alpha=plan["training"]["lora_alpha"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
    if args.phase == "tuned" and not args.adapter:
        raise ValueError("Tuned evaluation requires saved adapter")
    callback = RewardCallback()
    timing_path = args.out / "phase-timing.jsonl"
    sampler = NativeSampler(model, tokenizer, plan, deadline=deadline, timing_path=timing_path)
    bridge = JudgeBridge(
        args.workspace, args.judge_python, args.out, deadline=deadline, plan=args.plan
    )
    hook = RolloutHook(records, sampler, bridge, callback, plan)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(3407)
    if args.phase in ("baseline", "tuned"):
        chosen = set(plan.get("baseline_row_ids") or [])
        if not chosen:
            raise ValueError("Select the small baseline row set before generation")
        measurements = []
        for split in ("calibration", "evaluation"):
            for row, fit in zip(records[split], fits[split], strict=True):
                if row["row_id"] in chosen:
                    if not fit["eligible"]:
                        raise ValueError("Baseline row does not fit; preserve full input")
                    samples = sampler(row["row_id"], row["prompt"], 4)
                    values = bridge(samples, split)
                    measurements.append(
                        {
                            "row_id": row["row_id"],
                            "split": split,
                            "rewards": values,
                            "metrics": bridge.last_metrics,
                        }
                    )
        if len(measurements) != len(chosen):
            raise ValueError("Baseline rows must belong to fixed calibration/evaluation splits")
        write_new(
            args.out / "measurement.json",
            {
                "model": plan["model"],
                "sampling": sampling(plan),
                "judge": judging_contract(args.workspace),
                "optimizer_steps": 0,
                "adapter": str(args.adapter) if args.adapter else None,
                "rows": measurements,
                "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
        )
    else:
        chosen = set(plan.get("fit_train_row_ids") or [])
        if not chosen:
            raise ValueError("Explicitly select complete fit-tested training rows")
        selected = [
            row
            for row, fit in zip(records["train"], fits["train"], strict=True)
            if row["row_id"] in chosen and fit["eligible"]
        ]
        if len(selected) != len(chosen):
            raise ValueError("Selected training row missing or too long; never truncate")
        steps = 1 if args.phase == "step" else plan["training"]["max_steps"]
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=GRPOConfig(**trainer_parameters(plan, args.out / "checkpoints", max_steps=steps)),
            train_dataset=Dataset.from_list(selected),
            reward_funcs=callback,
            rollout_func=hook,
            callbacks=[make_timing_callback(deadline, timing_path)],
        )
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        record_event(timing_path, "final_adapter_save_begin")
        trainer.save_model(str(args.out / "adapter"))
        trainer.save_state()
        record_event(timing_path, "final_adapter_save_end")
        changed = any(
            not torch.equal(before[name], parameter.detach().cpu())
            for name, parameter in model.named_parameters()
            if name in before
        )
        write_new(
            args.out / "step_metrics.json",
            {
                "global_step": trainer.state.global_step,
                "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "resume_from": str(args.resume) if args.resume else None,
                "log_history": trainer.state.log_history,
                "adapter_weights_changed": changed,
            },
        )
        if not changed:
            raise RuntimeError("No adapter weights changed; inspect reward contrast and gradients")
    return 0


class RolloutHook:
    def __init__(self, records, sampler, bridge, callback, plan):
        self.rows = {
            json.dumps(row["prompt"], sort_keys=True): row["row_id"] for row in records["train"]
        }
        self.sampler, self.bridge, self.callback, self.plan = sampler, bridge, callback, plan

    def __call__(self, prompts, trainer):
        from trl.models import unwrap_model_for_generation

        if self.callback.pending:
            raise ValueError("Previous sampled group was not consumed")
        samples = []
        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as active:
            self.sampler.model = active
            count = self.plan["training"]["num_generations"]
            if len(prompts) % count:
                raise ValueError("Pinned TRL prompt slice contains an incomplete group")
            for start in range(0, len(prompts), count):
                prompt = prompts[start]
                if any(item != prompt for item in prompts[start : start + count]):
                    raise ValueError("Pinned TRL prompt group is not contiguous and identical")
                identity = self.rows[json.dumps(prompt, sort_keys=True)]
                group = self.sampler(identity, prompt, count)
                values = self.bridge(group, "train")
                if (self.plan.get("feasibility_only") or {}).get("authorized") is True and len(
                    set(values)
                ) < 2:
                    raise RuntimeError("Feasibility group has no reward contrast; stop before loss")
                samples.extend(
                    [
                        {**sample, "reward": value}
                        for sample, value in zip(group, values, strict=True)
                    ]
                )
        self.callback.pending = samples
        return {
            "prompt_ids": [sample["prompt_ids"] for sample in samples],
            "completion_ids": [sample["completion_ids"] for sample in samples],
            "logprobs": [sample["logprobs"] for sample in samples],
        }


class NativeSampler:
    def __init__(
        self, model, tokenizer, plan, *, deadline=float("inf"), timing_path=None,
        weight_version: str | None = None,
    ):
        self.model, self.tokenizer, self.plan = model, tokenizer, plan
        self.deadline = deadline
        self.timing_path = timing_path
        self.group_index = 0
        self.weight_version = weight_version

    def __call__(self, row_id, messages, count):
        import torch

        ids = prompt_token_ids(
            self.tokenizer, messages, template_options=policy_template_options(self.plan)
        )
        train = self.plan["training"]
        if (
            len(ids) > train["max_prompt_tokens"]
            or len(ids) + train["max_completion_tokens"] > self.plan["model"]["context_limit"]
        ):
            raise ValueError("Whole prompt does not fit")
        tensor = torch.tensor([ids], device=generation_device(self.model))
        samples = []
        self.group_index += 1
        was_training = self.model.training
        self.model.eval()
        try:
            for draw_index in range(1, count + 1):
                remaining_seconds(self.deadline)
                recorder = SamplingRecorder()
                started = time.monotonic()
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=tensor,
                        attention_mask=torch.ones_like(tensor),
                        max_new_tokens=train["max_completion_tokens"],
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        min_p=None,
                        repetition_penalty=1.0,
                        logits_processor=[recorder],
                        return_dict_in_generate=False,
                        output_scores=False,
                        use_cache=True,
                        max_time=remaining_seconds(self.deadline),
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                generated = output[0, len(ids) :].tolist()
                recorder.finish(generated[-1])
                generated_at = time.monotonic()
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                samples.append(
                    capture_sample(
                        row_id,
                        ids,
                        generated,
                        recorder.logprobs,
                        self.tokenizer,
                        finish_reason="stop" if generated[-1] in eos else "length",
                        raw_logprobs=recorder.raw_logprobs,
                        sampling_support_ids=recorder.sampling_support_ids,
                        weight_version=self.weight_version,
                    )
                )
                if self.timing_path is not None:
                    folder = self.timing_path.parent / "draws" / f"group-{self.group_index:04d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    write_new(folder / f"draw-{draw_index:04d}.json", samples[-1])
                record_event(
                    self.timing_path,
                    "draw_completed",
                    row_id=row_id,
                    group_index=self.group_index,
                    draw_index=draw_index,
                    prompt_tokens=len(ids),
                    completion_tokens=len(generated),
                    generation_seconds=generated_at - started,
                    capture_seconds=time.monotonic() - generated_at,
                    tokens_per_second=len(generated) / (generated_at - started),
                    finish_reason=samples[-1]["finish_reason"],
                )
        finally:
            self.model.train(was_training)
        return samples


def generation_device(model):
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return parameter.device
    raise ValueError("Native generation requires a CUDA parameter device")


class JudgeBridge:
    def __init__(self, workspace, interpreter, out, *, deadline=float("inf"), plan=None):
        self.workspace, self.interpreter, self.out = (
            workspace.resolve(),
            interpreter.absolute(),
            out.resolve(),
        )
        self.counter = 0
        self.deadline = deadline
        self.plan = plan

    def __call__(self, samples, split):
        self.counter += 1
        folder = self.out / f"group-{self.counter:04d}"
        folder.mkdir()
        write_new(folder / "input.json", samples)
        command = [
            str(self.interpreter),
            "-m",
            "grpo.judge_worker",
            "--workspace",
            str(self.workspace),
            "--samples",
            str(folder / "input.json"),
            "--split",
            split,
            "--out",
            str(folder / "grading"),
            "--run-paid",
        ]
        if self.plan is not None:
            command.extend(["--plan", str(self.plan.resolve())])
        with (folder / "worker.log").open("x") as log:
            record_event(self.out / "phase-timing.jsonl", "judge_start", group=folder.name)
            result = subprocess.run(
                command,
                cwd=self.workspace / "code/backend",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=min(7200, remaining_seconds(self.deadline)),
                check=False,
            )
            record_event(
                self.out / "phase-timing.jsonl",
                "judge_end",
                group=folder.name,
                returncode=result.returncode,
            )
        if result.returncode:
            raise RuntimeError(f"Whole group withheld; inspect {folder}")
        values = json.loads((folder / "grading/rewards.json").read_text())
        self.last_metrics = json.loads((folder / "grading/metrics.json").read_text())
        if len(values) != len(samples) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
        ):
            raise ValueError("Judge bridge returned partial/unknown rewards")
        return values


if __name__ == "__main__":
    raise SystemExit(main())
