"""No individual integration failure may disappear from the final decision."""

import importlib
from dataclasses import fields

import pytest

CONDITIONS = (
    "sampling_semantics",
    "raw_probability_parity",
    "processed_probability_parity",
    "repeat_stability",
    "adapter_transfer",
    "adapter_precision",
    "trainer_refresh",
    "changed_weight_progression",
)


def implementation():
    try:
        return importlib.import_module("acceptance")
    except ModuleNotFoundError:
        pytest.fail("Complete integration acceptance gate is missing")


@pytest.mark.parametrize("failed", CONDITIONS)
def test_each_required_condition_blocks_integration(failed):
    module = implementation()
    values = dict.fromkeys(CONDITIONS, True)
    values[failed] = False
    decision = module.evaluate_integration(module.IntegrationEvidence(**values))
    assert decision.approved is False
    assert decision.failed_conditions == (failed,)


def test_all_conditions_pass_and_unknown_is_rejected():
    module = implementation()
    evidence = module.IntegrationEvidence(**dict.fromkeys(CONDITIONS, True))
    assert {field.name for field in fields(evidence)} == set(CONDITIONS)
    assert module.evaluate_integration(evidence).approved
    with pytest.raises(ValueError, match="boolean"):
        module.evaluate_integration(
            module.IntegrationEvidence(
                **{**dict.fromkeys(CONDITIONS, True), "trainer_refresh": None}
            )
        )
