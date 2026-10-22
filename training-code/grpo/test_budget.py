"""Persistent budget reservations across separate judge workers."""

import json

import pytest


def test_luna_uses_its_rates_in_shared_operation_ledger(tmp_path):
    from grpo.budget import JudgeBudget

    budget = JudgeBudget(
        tmp_path / "ledger.json",
        cap=150,
        operation_id="benchmark",
        operation_cap=10,
        model="gpt-6-luna",
    )
    call = budget.reserve(1000)
    budget.settle(call, {"input_tokens": 1000, "cached_input_tokens": 100, "output_tokens": 100})
    record = json.loads(budget.path.read_text())["calls"][0]
    assert record["model"] == "gpt-6-luna"
    assert record["estimated_usd"] == pytest.approx(0.000141)
    assert record["budget_charge_usd"] == pytest.approx(0.000715)


def test_reservations_survive_worker_restart_and_prevent_overspending(tmp_path):
    from grpo.budget import JudgeBudget

    path = tmp_path / "ledger.json"
    first = JudgeBudget(path, cap=6)
    call = first.reserve(1000)
    second = JudgeBudget(path, cap=6)
    with pytest.raises(ValueError, match="budget"):
        second.reserve(1000)
    first.settle(call, {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 100})
    second.reserve(1000)
    ledger = json.loads(path.read_text())
    assert ledger["calls"][0]["estimated_usd"] == pytest.approx(0.003)
    assert ledger["calls"][0]["budget_charge_usd"] == pytest.approx(0.0143)
    assert ledger["calls"][1]["status"] == "reserved"


def test_missing_usage_keeps_reservation_and_duplicate_settlement_is_rejected(tmp_path):
    from grpo.budget import JudgeBudget

    path = tmp_path / "ledger.json"
    budget = JudgeBudget(path, cap=6)
    call = budget.reserve(1000)
    budget.settle(call, None)
    with pytest.raises(ValueError, match="budget"):
        JudgeBudget(path, cap=6).reserve(1000)
    with pytest.raises(ValueError, match="settled"):
        budget.settle(call, {"input_tokens": 1, "output_tokens": 1})


def test_malformed_usage_is_not_allowed_to_release_funds(tmp_path):
    from grpo.budget import JudgeBudget

    budget = JudgeBudget(tmp_path / "ledger.json", cap=6)
    call = budget.reserve(1000)
    with pytest.raises(ValueError, match="usage"):
        budget.settle(call, {"input_tokens": -1, "output_tokens": 100})
    with pytest.raises(ValueError, match="budget"):
        budget.reserve(1000)


def test_operation_cap_survives_restart_inside_shared_ledger(tmp_path):
    from grpo.budget import JudgeBudget

    path = tmp_path / "ledger.json"
    first = JudgeBudget(path, cap=150, operation_id="cycle-1", operation_cap=10)
    first.reserve(0)
    first.reserve(0)
    restarted = JudgeBudget(path, cap=150, operation_id="cycle-1", operation_cap=10)
    with pytest.raises(ValueError, match="operation budget"):
        restarted.reserve(0)
    assert len(json.loads(path.read_text())["calls"]) == 2
    JudgeBudget(path, cap=150, operation_id="cycle-2", operation_cap=10).reserve(0)


def test_operations_in_one_ledger_reserve_and_settle_independently(tmp_path):
    from grpo.budget import JudgeBudget

    path = tmp_path / "ledger.json"
    first = JudgeBudget(path, cap=150, operation_id="cycle-1", operation_cap=5)
    second = JudgeBudget(path, cap=150, operation_id="cycle-2", operation_cap=5)
    held = first.reserve(0)
    other = second.reserve(0)
    first.settle(held, {"input_tokens": 1, "output_tokens": 1})
    first.reserve(0)
    calls = {call["id"]: call for call in json.loads(path.read_text())["calls"]}
    assert calls[held]["budget_charge_usd"] == pytest.approx(44e-6)
    assert calls[other]["status"] == "reserved"
    assert calls[other]["budget_charge_usd"] == calls[other]["reserved_usd"] > 0
