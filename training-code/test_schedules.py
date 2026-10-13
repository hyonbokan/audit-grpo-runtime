"""The schedule registry resolves only reviewed schedules, and the cycle contract enforces them."""

import copy
import json

import pytest

import schedules
from cycle_contract import validate_config
from testing.cycles import schedule_cycle

REGISTERED = sorted(json.loads(schedules.REGISTRY.read_text())["schedules"])


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "schedules.json"
    path.write_text(
        json.dumps(
            {
                "defaults": {
                    "maximum_context": 1000,
                    "training": {"seconds_limit": 10800, "policy_reasoning_effort": "medium"},
                },
                "schedules": {
                    "short": {"training": {"initial_step": 5, "maximum_step": 7}},
                    "long": {"maximum_context": 2000, "training": {"seconds_limit": 600}},
                },
            }
        )
    )
    monkeypatch.setattr(schedules, "REGISTRY", path)


def test_override_replaces_only_the_fields_it_names(registry):
    short = schedules.schedule("short")
    assert short["maximum_context"] == 1000
    assert short["training"] == {
        "seconds_limit": 10800,
        "policy_reasoning_effort": "medium",
        "initial_step": 5,
        "maximum_step": 7,
        "schedule_id": "short",
    }
    long = schedules.schedule("long")
    assert (long["maximum_context"], long["training"]["seconds_limit"]) == (2000, 600)


def test_run_cycle_and_panel_configs_resolve_their_own_schedule(registry):
    assert schedules.scope({"schedule_id": "short"})["schedule_id"] == "short"
    assert schedules.scope({"weekend_schedule_id": "long"})["schedule_id"] == "long"


@pytest.mark.parametrize("config", [{"schedule_id": "missing"}, {"weekend_schedule_id": "x"}, {}])
def test_unregistered_or_unnamed_schedule_is_rejected(registry, config):
    with pytest.raises(ValueError, match=r"Unreviewed|names no schedule"):
        schedules.scope(config)


@pytest.mark.parametrize("schedule_id", REGISTERED)
def test_contract_accepts_each_single_step_up_to_the_cap_and_nothing_beyond(schedule_id):
    training = schedules.schedule(schedule_id)["training"]
    first, maximum = training["initial_step"], training["maximum_step"]
    validate_config(schedule_cycle(schedule_id, first))
    validate_config(schedule_cycle(schedule_id, maximum - 1))
    other_effort = "xhigh" if training["policy_reasoning_effort"] == "medium" else "medium"
    for changes in (
        {"initial_step": maximum, "target_step": maximum + 1},
        {"initial_step": first - 1, "target_step": first},
        {"target_step": first + 2},
        {"policy_reasoning_effort": other_effort},
        {"seconds_limit": training["seconds_limit"] + 1},
        {"absolute_deadline": "2030-01-01T00:00:00+00:00"},
        {"offload_layers": training["offload_layers"] + 1},
    ):
        with pytest.raises(ValueError):
            validate_config(dict(schedule_cycle(schedule_id), **changes))
    shortened = copy.deepcopy(schedule_cycle(schedule_id))
    shortened["sampling"]["max_tokens"] -= 1
    with pytest.raises(ValueError):
        validate_config(shortened)
