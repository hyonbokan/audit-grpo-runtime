"""A training run's config, source plan and ledgers come only from its reviewed schedule."""

import copy
import json

import pytest

import cycle_contract
import schedules
import training_config
from runtime_io import read_json, sha256_file
from testing.checkpoints import trained_checkpoint

SCHEDULE = "free-text-seg1-20261023"
BENCHMARK_MANIFEST = "bench/assets/BENCHMARK_MANIFEST.json"
HOST = {
    "runtime_host": "/srv/runtime",
    "artifact_host": "/srv/cache/runs",
    "hf_repo": "owner/private-backups",
    "judge_env_file": "/opt/secrets/judge.env",
}
LOCK = """accelerate==1.15.0 \\
    --hash=sha256:00
datasets==4.8.5 \\
peft==0.21.2 \\
torch==2.13.0+cu129 \\
transformers==5.17.0 \\
trl==1.13.0 \\
unsloth==2026.10.1 \\
unsloth-zoo==2026.10.1 \\
vllm==0.20.0 \\
"""


@pytest.fixture
def results(tmp_path, monkeypatch):
    """A runs/ tree with the launch inputs, and the schedule registered against them."""
    root = tmp_path / "results"
    (root / "corpus").mkdir(parents=True)
    (root / "corpus/manifest.json").write_text('{"roles": {}}')
    (root / "bench/assets").mkdir(parents=True)
    (root / BENCHMARK_MANIFEST).write_text('{"projects": []}')
    (root / "prior").mkdir()
    (root / "prior/analysis.json").write_text('{"repeat": {"raw_max_abs_logprob_difference": 0.6}}')
    trained_checkpoint(root / "october/checkpoint-13")
    (root / SCHEDULE).mkdir()
    registry = json.loads(schedules.REGISTRY.read_text())
    registry["schedules"][SCHEDULE]["launch"] = {
        "campaign": "campaign",
        "corpus": "corpus",
        "corpus_manifest_sha256": sha256_file(root / "corpus/manifest.json"),
        "benchmark_assets": "bench/assets",
        "benchmark_manifest_sha256": sha256_file(root / BENCHMARK_MANIFEST),
        "prior_analysis": "prior/analysis.json",
        "start": {"template": "october/checkpoint-13", "seed": 7},
    }
    path = tmp_path / "schedules.json"
    path.write_text(json.dumps(registry))
    monkeypatch.setattr(schedules, "REGISTRY", path)
    (tmp_path / "gpu.lock").write_text(LOCK)
    return root


def write(results, schedule_id=SCHEDULE):
    return training_config.write_run(
        schedule_id, results=results, lock=results.parent / "gpu.lock", host=HOST
    )


def test_the_first_visit_built_from_the_written_config_passes_the_cycle_contract(results):
    config = read_json(write(results))
    cycle = copy.deepcopy(config["cycle"])
    # What controller.next_cycle adds per visit.
    cycle.update(row_id="row/all-scope", request="/results/r.json", prompt_token_count=31006)
    cycle["sampling"]["max_tokens"] = 65536 - 31006
    cycle_contract.validate_config(cycle)
    assert cycle["learning_rate"] == 2e-5
    assert cycle["engine"]["max_model_len"] == 65536
    assert cycle["engine"]["tensor_parallel_size"] == 2
    assert cycle["absolute_deadline"] == config["training_cutoff"] == "2026-10-29T12:00:00+00:00"


def test_training_starts_from_a_fresh_step_zero_adapter_whose_hash_the_config_pins(results):
    config = read_json(write(results))
    start = results / f"{SCHEDULE}-start/checkpoint-0"
    assert config["initial_checkpoint"] == f"/results/{SCHEDULE}-start/checkpoint-0"
    assert config["initial_adapter_sha256"] == sha256_file(start / "adapter_model.safetensors")
    assert config["cycle"]["checkpoint"] == config["initial_checkpoint"]
    assert read_json(start / "continuation_schedule.json")["learning_rate"] == 2e-5
    assert "training_row_ids" not in config


def test_paths_name_the_host_for_the_controller_and_results_for_the_workers(results):
    config = read_json(write(results))
    assert config["dataset_host"] == "/srv/runtime/runs/corpus"
    assert config["dataset_container"] == config["cycle"]["dataset_workspace"] == "/results/corpus"
    assert config["benchmark_assets"] == "/srv/runtime/runs/bench/assets"
    assert config["run_host"] == f"/srv/runtime/runs/{SCHEDULE}"
    assert config["artifact_host"] == "/srv/cache/runs"
    assert config["hf_repo"] == "owner/private-backups"
    assert config["judge_env_file"] == config["cycle"]["judge_env_file"] == "/opt/secrets/judge.env"
    assert config["cycle"]["source_plan"] == f"/results/{SCHEDULE}/source-plan.json"
    assert (results / SCHEDULE / "prior-analysis.json").read_text() == (
        results / "prior/analysis.json"
    ).read_text()


def test_the_campaign_ledgers_carry_the_reviewed_caps_and_the_plan_points_at_them(results):
    config = read_json(write(results))
    assert config["judge_ledger"] == "/results/campaign/judge-training.json"
    assert read_json(results / "campaign/judge-training.json") == {"cap_usd": 120, "calls": []}
    assert read_json(results / "campaign/judge-evaluation.json") == {"cap_usd": 30, "calls": []}
    plan = read_json(results / SCHEDULE / "source-plan.json")
    assert plan["access"] == {
        "judge_usd_cap": 120,
        "local_judge_budget": {"path": "/results/campaign/judge-training.json"},
    }
    assert plan["sampling_profile"] == {"id": SCHEDULE, "policy_reasoning_effort": "medium"}


def test_the_plan_pins_the_update_packages_from_the_image_lock(results):
    write(results)
    plan = read_json(results / SCHEDULE / "source-plan.json")
    assert plan["package_versions"] == {
        "unsloth": "2026.10.1",
        "unsloth_zoo": "2026.10.1",
        "trl": "1.13.0",
        "transformers": "5.17.0",
        "peft": "0.21.2",
        "torch": "2.13.0+cu129",
        "datasets": "4.8.5",
        "accelerate": "1.15.0",
    }


def test_a_relaunch_reuses_the_step_zero_adapter_and_the_ledgers_spent_so_far(results):
    first = read_json(write(results))
    spent = {"cap_usd": 120, "calls": [{"budget_charge_usd": 0.4}]}
    (results / "campaign/judge-training.json").write_text(json.dumps(spent))
    for path in (results / SCHEDULE).iterdir():
        path.unlink()
    second = read_json(write(results))
    assert second["initial_adapter_sha256"] == first["initial_adapter_sha256"]
    assert read_json(results / "campaign/judge-training.json") == spent


@pytest.mark.parametrize(
    "breakage, message",
    [
        (lambda root: (root / SCHEDULE / "config.json").write_text("{}"), "empty run directory"),
        (lambda root: (root / "corpus/manifest.json").write_text("{}"), "Corpus manifest"),
        (lambda root: (root / BENCHMARK_MANIFEST).write_text("{}"), "Benchmark"),
        (
            lambda root: (
                (root / "campaign").mkdir()
                or (root / "campaign/judge-training.json").write_text(
                    '{"cap_usd": 150, "calls": []}'
                )
            ),
            "cap",
        ),
    ],
)
def test_changed_inputs_or_a_used_run_directory_are_refused(results, breakage, message):
    breakage(results)
    with pytest.raises((ValueError, FileExistsError), match=message):
        write(results)


def test_a_schedule_without_launch_inputs_cannot_start_a_run(results):
    (results / "medium-day-20261010").mkdir()
    with pytest.raises(ValueError, match="launch"):
        write(results, "medium-day-20261010")


@pytest.fixture
def selected_rows(results):
    rows = ["second/all-scope", "first/all-scope"]
    manifest = results / "corpus/manifest.json"
    manifest.write_text(json.dumps({"roles": {"train": [*rows, "unused/all-scope"]}}))
    path = results / "selected-rows.json"
    path.write_text(json.dumps(rows))
    registry = read_json(schedules.REGISTRY)
    launch = registry["schedules"][SCHEDULE]["launch"]
    launch.update(
        corpus_manifest_sha256=sha256_file(manifest),
        training_rows="selected-rows.json",
        training_rows_sha256=sha256_file(path),
    )
    schedules.REGISTRY.write_text(json.dumps(registry))
    return rows


def test_selected_training_rows_keep_the_reviewed_order(results, selected_rows):
    config = read_json(write(results))
    assert config["training_row_ids"] == selected_rows


def test_changed_training_row_selection_is_refused(results, selected_rows):
    (results / "selected-rows.json").write_text(json.dumps(list(reversed(selected_rows))))
    with pytest.raises(ValueError, match="Training row selection identity"):
        write(results)


@pytest.mark.parametrize("rows", [[], ["unknown/all-scope"], ["first/all-scope"] * 2, "first"])
def test_invalid_training_row_selection_is_refused(results, selected_rows, rows):
    path = results / "selected-rows.json"
    path.write_text(json.dumps(rows))
    registry = read_json(schedules.REGISTRY)
    registry["schedules"][SCHEDULE]["launch"]["training_rows_sha256"] = sha256_file(path)
    schedules.REGISTRY.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="unique admitted training rows"):
        write(results)


def test_continuation_keeps_the_accepted_checkpoint_and_step(results):
    registry = read_json(schedules.REGISTRY)
    entry = registry["schedules"][SCHEDULE]
    entry["launch"]["start"] = {"checkpoint": "october/checkpoint-13"}
    entry["training"]["initial_step"] = 14
    schedules.REGISTRY.write_text(json.dumps(registry))
    config = read_json(write(results))
    assert config["initial_checkpoint"] == "/results/october/checkpoint-13"
    assert config["initial_adapter_sha256"] == sha256_file(
        results / "october/checkpoint-13/adapter_model.safetensors"
    )
    assert config["cycle"]["initial_step"] == 14


def test_training_row_selection_cannot_exceed_the_reviewed_group_limit(results, selected_rows):
    registry = read_json(schedules.REGISTRY)
    registry["schedules"][SCHEDULE]["training"]["maximum_groups"] = 1
    schedules.REGISTRY.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="unique admitted training rows"):
        write(results)
