"""Fresh xhigh comparisons must identify actual final weights, preserving old profiles."""

import pytest

from benchmark_evaluation import validate_benchmark_config
from testing.benchmark import scheduled_configuration as configuration


def test_weekend_evaluation_accepts_actual_final_identity_and_xhigh_both_arms():
    config = configuration()
    validate_benchmark_config(config)
    for case in config["cases"]:
        validate_benchmark_config(dict(config, **case))


def test_medium_checkpoint_evaluation_requires_medium_for_both_arms():
    config = configuration()
    config["weekend_schedule_id"] = "medium-checkpoint-20261010"
    config["template_options"]["reasoning_effort"] = "medium"
    config["cases"][-1].update(label="checkpoint-3", initial_step=3)
    validate_benchmark_config(config)
    for case in config["cases"]:
        validate_benchmark_config(dict(config, **case))
    for effort in ("low", "xhigh"):
        wrong = dict(
            config, template_options=dict(config["template_options"], reasoning_effort=effort)
        )
        with pytest.raises(ValueError):
            validate_benchmark_config(wrong)


@pytest.mark.parametrize(
    "change",
    [
        {"template_options": {"reasoning_effort": "low"}},
        {"absolute_deadline": "2026-10-14T00:00:00+00:00"},
        {"sampling": {"n": 1, "max_tokens": 16384}},
    ],
)
def test_weekend_evaluation_rejects_mismatched_profile(change):
    config = configuration()
    for key, value in change.items():
        config[key] = dict(config[key], **value) if isinstance(value, dict) else value
    with pytest.raises(ValueError):
        validate_benchmark_config(config)


def test_final_step_cannot_be_mislabeled_as_checkpoint_two():
    config = configuration()
    config["cases"][-1]["label"] = "checkpoint-2"
    with pytest.raises(ValueError):
        validate_benchmark_config(config)


def test_unregistered_schedule_id_is_rejected_instead_of_skipping_its_limits():
    config = configuration()
    config["weekend_schedule_id"] = "typo-schedule"
    with pytest.raises(ValueError, match="Unreviewed schedule"):
        validate_benchmark_config(config)
