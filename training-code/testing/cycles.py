"""Cycle configs for a registered schedule, built on the synthetic cycle fixture."""

import json
from pathlib import Path

import schedules

ROOT = Path(__file__).resolve().parents[1]
PROMPT_TOKENS = 18625


def schedule_cycle(schedule_id: str, initial_step: int | None = None) -> dict:
    """A cycle the schedule permits: its limits, full remaining context and one update."""
    reviewed = schedules.schedule(schedule_id)
    training = reviewed["training"]
    step = training["initial_step"] if initial_step is None else initial_step
    config = json.loads((ROOT / "fixtures/cycle.json").read_text())
    config.update(
        schedule_id=schedule_id,
        initial_step=step,
        target_step=step + 1,
        row_id="tiny/all-scope",
        dataset_workspace="/dataset",
        policy_reasoning_effort=training["policy_reasoning_effort"],
        absolute_deadline=reviewed["training_cutoff"],
        seconds_limit=training["seconds_limit"],
        judge_seconds_limit=training["judge_seconds_limit"],
        judge_operation_cap_usd=training["judge_operation_cap_usd"],
        prompt_token_count=PROMPT_TOKENS,
        offload_layers=training["offload_layers"],
        prefetch_depth=training["prefetch_depth"],
    )
    if "update_placement" in training:
        config["update_placement"] = training["update_placement"]
    if "tensor_parallel_size" in training:
        config["engine"]["tensor_parallel_size"] = training["tensor_parallel_size"]
    config["sampling"]["max_tokens"] = reviewed["maximum_context"] - PROMPT_TOKENS
    config["engine"]["max_model_len"] = reviewed["maximum_context"]
    config["engine"]["gpu_memory_utilization"] = training["gpu_memory_utilization"]
    return config
