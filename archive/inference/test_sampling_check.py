"""The acceptance job cannot silently expand its agreed budget or retries."""

import importlib

import pytest


def implementation():
    try:
        return importlib.import_module("sampling_check")
    except ModuleNotFoundError:
        pytest.fail("Bounded sampler acceptance application is missing")


@pytest.mark.parametrize(
    "field, value",
    [("gpu_seconds_limit", 121), ("retry_limit", 1), ("atol", 1e-3), ("gpu_seconds_limit", 0)],
)
def test_configuration_rejects_expanded_acceptance(field, value):
    config = {"gpu_seconds_limit": 120, "retry_limit": 0, "atol": 1e-6}
    config[field] = value
    with pytest.raises(ValueError, match="budget|retry|tolerance"):
        implementation().validate_limits(config)
