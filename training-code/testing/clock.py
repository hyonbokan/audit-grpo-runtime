"""Freeze `datetime.now` inside a module so deadline checks do not depend on today's date."""

from datetime import UTC, datetime


def freeze(monkeypatch, module, iso: str) -> None:
    frozen = datetime.fromisoformat(iso)

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen.astimezone(tz or UTC)

    monkeypatch.setattr(module, "datetime", FixedClock)
