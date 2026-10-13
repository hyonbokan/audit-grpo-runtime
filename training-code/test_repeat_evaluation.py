"""Extra evaluation draws reuse the one-shot design with fresh seeds and never alter originals."""

import pytest

import schedules
from benchmark_evaluation import validate_benchmark_config
from testing.benchmark import scheduled_configuration as configuration

SCHEDULE = "medium-day-20261010-repeat"


def repeat_config(seed=3408, arms=("base", "checkpoint-13")):
    reviewed = schedules.schedule(SCHEDULE)
    config = configuration()
    config["cases"][-1].update(label="checkpoint-13", initial_step=13)
    config.update(
        weekend_schedule_id=SCHEDULE,
        absolute_deadline=reviewed["final_deadline"],
        seconds_limit=reviewed["evaluation_seconds_limit"],
        repeat_evaluation={"seed": seed, "arms": list(arms)},
        generation_order=list(arms),
    )
    config["template_options"]["reasoning_effort"] = "medium"
    config["sampling"]["seed"] = config["engine"]["seed"] = seed
    return config


def test_repeat_scope_is_registered_medium_and_full_context():
    assert schedules.is_registered(SCHEDULE)
    scope = schedules.schedule(SCHEDULE)
    assert scope["evaluation_effort"] == "medium" and scope["maximum_context"] == 262144


def test_repeat_accepts_the_real_checkpoint_arm_and_fresh_seed():
    validate_benchmark_config(repeat_config())
    validate_benchmark_config(repeat_config(seed=3409, arms=("checkpoint-13",)))


@pytest.mark.parametrize(
    "seed,arms",
    [(3407, ("base", "checkpoint-13")), (3408, ("base", "checkpoint-2")), (3408, ("base", "base"))],
)
def test_repeat_rejects_reused_seed_unknown_arm_or_duplicates(seed, arms):
    config = repeat_config(seed, arms)
    with pytest.raises(ValueError):
        validate_benchmark_config(config)


def test_child_config_changes_only_identity_seed_and_schedule():
    from repeat_evaluation import repeat_child_config

    source = repeat_config(seed=3407)
    source.pop("repeat_evaluation")
    source["sampling"]["seed"] = source["engine"]["seed"] = 3407
    source.update(run_id="orig-evaluation-phi", weekend_schedule_id="medium-day-20261010")
    child = repeat_child_config(source, "rep1-evaluation-phi", 3408, SCHEDULE)
    assert child["run_id"] == "rep1-evaluation-phi"
    assert child["repeat_evaluation"] == {"seed": 3408, "arms": ["base", "checkpoint-13"]}
    assert child["sampling"]["seed"] == child["engine"]["seed"] == 3408
    assert child["weekend_schedule_id"] == SCHEDULE
    assert child["absolute_deadline"] == schedules.schedule(SCHEDULE)["final_deadline"]
    untouched = {
        k
        for k in source
        if k
        not in {
            "run_id",
            "weekend_schedule_id",
            "absolute_deadline",
            "seconds_limit",
            "repeat_evaluation",
            "sampling",
            "engine",
            "generation_order",
        }
    }
    assert all(child[k] == source[k] for k in untouched)
    assert source["sampling"]["seed"] == 3407 and "repeat_evaluation" not in source
    validate_benchmark_config(child)
