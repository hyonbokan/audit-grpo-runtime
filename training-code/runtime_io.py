"""JSON files, hashes, deadlines and result directories shared by the controllers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def save_json(path: Path | str, value: Any) -> None:
    """Write atomically, so a crash never leaves a half-written receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def seconds_until(deadline: str) -> float:
    return (datetime.fromisoformat(deadline) - utc_now()).total_seconds()


def new_phase_dir(out: Path) -> None:
    """A phase's output directory, owned like its cycle directory.

    Containers write as root; the host launcher archives a failed attempt by moving this
    directory, which needs write access to it.
    """
    out.mkdir(parents=True, exist_ok=False)
    owner = out.parent.stat()
    os.chown(out, owner.st_uid, owner.st_gid)


def new_result_dir(config: dict[str, Any], name: str) -> Path:
    """A fresh result directory on the artifact disk, linked under runs/, with frozen source."""
    host = Path(config["runtime_host"]) / "runs" / name
    physical = Path(config["artifact_host"]) / name
    physical.mkdir(parents=True, exist_ok=False)
    # Containers write as root; the host launcher, as run_host's owner, writes its logs here.
    owner = Path(config["run_host"]).stat()
    os.chown(physical, owner.st_uid, owner.st_gid)
    host.symlink_to(physical, target_is_directory=True)
    shutil.copyfile(Path(config["run_host"]) / "SOURCE_SHA256SUMS", physical / "SOURCE_SHA256SUMS")
    return host
