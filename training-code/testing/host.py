"""Host settings for launcher tests: a runtime tree's host.env and its cache directory."""

from pathlib import Path


def write_host_settings(runtime: Path, *, peer_access: str = "enabled") -> Path:
    """Write `runtime/host.env` naming a fresh cache directory beside the runtime tree."""
    cache = runtime.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (runtime / "host.env").write_text(
        f"AUDIT_GRPO_CACHE={cache}\nAUDIT_GRPO_PEER_ACCESS={peer_access}\n"
    )
    return cache
