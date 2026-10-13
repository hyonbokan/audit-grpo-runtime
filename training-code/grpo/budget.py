"""Shared Sol judge ledger; unresolved calls retain their conservative reservation."""

from __future__ import annotations

import fcntl
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class BudgetExhausted(ValueError):
    """A spending cap stops grading; it is never a transient judge failure."""


def response_token_usage(raw) -> dict[str, int] | None:
    """Responses usage includes cached input; router input counts exclude it."""
    usage = raw.usage
    if usage is None:
        return None

    def field(value, name, default=None):
        return (
            value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
        )

    details = field(usage, "input_tokens_details")
    return {
        "input_tokens": field(usage, "input_tokens"),
        "cached_input_tokens": field(details, "cached_tokens", 0) or 0,
        "output_tokens": field(usage, "output_tokens"),
    }


class JudgeBudget:
    def __init__(self, path, *, cap, operation_id=None, operation_cap=None, model="gpt-6.1-sol"):
        if model not in ("gpt-6.1-sol", "gpt-6-luna"):
            raise ValueError("Judge budget requires a declared pricing model")
        self.model = model
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("Invalid judge budget cap")
        self.path, self.cap = path, cap
        if (operation_id is None) != (operation_cap is None) or (
            operation_id is not None
            and (
                not isinstance(operation_id, str)
                or not operation_id
                or type(operation_cap) not in (int, float)
                or not math.isfinite(operation_cap)
                or not 0 < operation_cap <= cap
            )
        ):
            raise ValueError("Invalid judge operation budget")
        self.operation_id, self.operation_cap = operation_id, operation_cap

    def reserve(self, input_token_bound):
        if type(input_token_bound) is not int or input_token_bound < 0:
            raise ValueError("Invalid input token bound")
        # Full output limit, long-context cache writes, Fast and regional uplifts.
        # Published rates: https://developers.openai.com/api/docs/models/gpt-6.1-sol
        input_rate, output_rate = (0.55, 1.65) if self.model == "gpt-6-luna" else (11, 33)
        amount = (
            min(input_token_bound, 1_050_000) * input_rate + 128_000 * output_rate
        ) / 1_000_000
        with self._ledger() as ledger:
            spent = sum(call["budget_charge_usd"] for call in ledger["calls"])
            if spent + amount > self.cap:
                raise BudgetExhausted("Local judge budget exhausted; next call withheld")
            if self.operation_id is not None:
                operation_spent = sum(
                    call["budget_charge_usd"]
                    for call in ledger["calls"]
                    if call.get("operation_id") == self.operation_id
                )
                if operation_spent + amount > self.operation_cap:
                    raise BudgetExhausted("Judge operation budget exhausted; next call withheld")
            call_id = uuid4().hex
            ledger["calls"].append(
                {
                    "id": call_id,
                    "model": self.model,
                    "status": "reserved",
                    "reserved_usd": amount,
                    "budget_charge_usd": amount,
                    "estimated_usd": None,
                    **({"operation_id": self.operation_id} if self.operation_id else {}),
                }
            )
        return call_id

    def settle(self, call_id, usage):
        estimate = charge = None
        if usage is not None:
            incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
            cached = usage.get("cached_input_tokens", 0) or 0
            if (
                any(type(value) is not int or value < 0 for value in (incoming, outgoing, cached))
                or cached > incoming
            ):
                raise ValueError("Invalid judge usage; reservation retained")
            long_context = incoming > 272_000
            estimate = (
                (incoming - cached) * (4 if long_context else 2)
                + cached * (0.2 if long_context else 0.1)
                + outgoing * (15 if long_context else 10)
            ) / 1_000_000
            charge = (incoming * 11 + outgoing * 33) / 1_000_000
            if self.model == "gpt-6-luna":
                # Luna Standard rates; conservative charge includes long-context/Fast/region.
                # https://developers.openai.com/api/docs/models/gpt-6-luna
                estimate = (
                    (incoming - cached) * (0.2 if long_context else 0.1)
                    + cached * (0.02 if long_context else 0.01)
                    + outgoing * (0.75 if long_context else 0.5)
                ) / 1_000_000
                charge = (incoming * 0.55 + outgoing * 1.65) / 1_000_000
        with self._ledger() as ledger:
            call = next((item for item in ledger["calls"] if item["id"] == call_id), None)
            if call is None:
                raise ValueError(f"Unknown judge reservation: {call_id}")
            if call.get("model", "gpt-6.1-sol") != self.model:
                raise ValueError("Judge settlement pricing model differs")
            if call["status"] != "reserved":
                raise ValueError("Judge call already settled")
            call.update(status="settled", usage=usage, estimated_usd=estimate)
            if charge is not None:
                call["budget_charge_usd"] = charge
            ledger["estimated_usd"] = sum(item["estimated_usd"] or 0 for item in ledger["calls"])
            ledger["budget_charge_usd"] = sum(item["budget_charge_usd"] for item in ledger["calls"])

    @contextmanager
    def _ledger(self):
        # A separate lock file lets the ledger be replaced atomically: a crash mid-write
        # leaves the previous complete ledger instead of a truncated shared file.
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_name(f"{path.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            content = path.read_text() if path.exists() else ""
            ledger = json.loads(content) if content else {"cap_usd": self.cap, "calls": []}
            if ledger["cap_usd"] != self.cap:
                raise ValueError("Shared judge budget cap changed")
            yield ledger
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("w") as stream:
                json.dump(ledger, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                # The root judge container must not take host ownership of a shared ledger.
                previous = path.stat()
                temporary.chmod(previous.st_mode & 0o7777)
                if (previous.st_uid, previous.st_gid) != (os.getuid(), os.getgid()):
                    os.chown(temporary, previous.st_uid, previous.st_gid)
            temporary.replace(path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
