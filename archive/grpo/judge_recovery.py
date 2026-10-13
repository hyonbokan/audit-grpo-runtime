"""Replay exact completed requests once, with one bounded missing-call recovery."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from typing import Any


def request_identity(request: dict) -> str:
    return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class RecoveryJudge:
    def __init__(self, grades: list[dict], paid: Any):
        self.paid = paid
        self.responses = defaultdict(deque)
        self.missing = []
        self.replayed = 0
        self.paid_calls = 0
        for grade in grades:
            requests, responses = grade["judge_requests"], grade["judge_responses"]
            if len(responses) > len(requests):
                raise ValueError("Saved judge response count differs")
            for request, response in zip(requests, responses):
                self.responses[request_identity(request)].append(copy.deepcopy(response))
            self.missing.extend(request_identity(request) for request in requests[len(responses) :])
        if len(self.missing) != 1:
            raise ValueError("Recovery permits exactly one known missing judge call")

    async def __call__(self, request: dict) -> dict:
        identity = request_identity(request)
        if self.responses[identity]:
            self.replayed += 1
            return self.responses[identity].popleft()
        if identity not in self.missing or self.paid_calls:
            raise ValueError("Unexpected or already-consumed judge recovery request")
        self.missing.remove(identity)
        self.paid_calls += 1
        return await self.paid(request)

    def finish(self) -> None:
        if self.missing or any(self.responses.values()) or self.paid_calls != 1:
            raise ValueError("Saved judge responses or missing call were not consumed exactly once")
