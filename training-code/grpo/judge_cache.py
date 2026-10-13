"""Persist completed judge responses so retries and resumed cycles never pay twice.

Responses are keyed by the exact request and by its occurrence within one grading
attempt, so identical requests in a group still receive independently paid responses.
A cache directory belongs to one cycle; it never crosses samples or reward contracts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


def request_identity(request: dict) -> str:
    return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CachedJudge:
    def __init__(self, paid: Callable[[dict], Awaitable[dict]], directory: Path):
        self.paid = paid
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.occurrences: Counter[str] = Counter()
        self.replayed = 0
        self.paid_calls = 0

    def new_attempt(self) -> None:
        """A whole-group retry asks the same requests again; replay them from the start."""
        self.occurrences.clear()

    async def __call__(self, request: dict) -> dict:
        identity = request_identity(request)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        occurrence = self.occurrences[digest]
        self.occurrences[digest] += 1
        path = self.directory / f"{digest}-{occurrence}.json"
        if path.is_file():
            saved = json.loads(path.read_text())
            if request_identity(saved["request"]) != identity:
                raise ValueError(f"Cached judge request differs: {path}")
            self.replayed += 1
            return saved["response"]
        response = await self.paid(request)
        self.paid_calls += 1
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps({"request": request, "response": response}))
        temporary.replace(path)
        return response

    def receipt(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "replayed_calls": self.replayed,
            "paid_calls": self.paid_calls,
        }
