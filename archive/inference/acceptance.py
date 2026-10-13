"""A sampler fix cannot bypass the remaining complete GRPO acceptance conditions."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class IntegrationEvidence:
    sampling_semantics: bool
    raw_probability_parity: bool
    processed_probability_parity: bool
    repeat_stability: bool
    adapter_transfer: bool
    adapter_precision: bool
    trainer_refresh: bool
    changed_weight_progression: bool


@dataclass(frozen=True)
class IntegrationDecision:
    approved: bool
    failed_conditions: tuple[str, ...]


def evaluate_integration(evidence: IntegrationEvidence) -> IntegrationDecision:
    names = tuple(field.name for field in fields(evidence))
    if any(type(getattr(evidence, name)) is not bool for name in names):
        raise ValueError("Every integration condition requires an explicit boolean result")
    failures = tuple(name for name in names if not getattr(evidence, name))
    return IntegrationDecision(approved=not failures, failed_conditions=failures)
