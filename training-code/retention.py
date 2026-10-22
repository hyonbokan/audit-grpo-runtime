"""Disk retention: drop state no later phase reads, and refuse to start work on a full disk."""

from __future__ import annotations

import shutil
from pathlib import Path

from hf_backup import RESUME_ONLY_NAMES
from runtime_io import read_json, save_json, utc_now

MERGE_MANIFEST = "MERGE_MANIFEST.json"
GIB = 2**30


# --- Visits


def prune_visits(state: dict, resume_states_kept: int) -> list[dict]:
    """Remove adapter copies from every visit and resume state from all but the newest steps."""
    if type(resume_states_kept) is not int or resume_states_kept < 1:
        raise ValueError("storage.resume_states_kept must be an integer of at least 1")
    removed = []
    for visit in state["visits"]:
        # The copy of the adapter that vLLM loaded; the rollout receipt keeps its hashes.
        removed += remove_tree(Path(visit["root"]) / "rollout/mapped-adapter")
    checkpoints = [
        Path(visit["root"]) / "update" / f"checkpoint-{visit['update']['global_step']}"
        for visit in state["visits"]
        if visit["decision"] == "update"
    ]
    for checkpoint in checkpoints[:-resume_states_kept]:
        for name in sorted(RESUME_ONLY_NAMES):
            removed += remove_file(checkpoint / name)
    return removed


def remove_tree(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    removed = [record(path) for path in sorted(root.rglob("*")) if path.is_file()]
    shutil.rmtree(root)
    return removed


def remove_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    removed = [record(path)]
    path.unlink()
    return removed


def record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size}


# --- Merged models


def release_merge(merged: Path, receipt: Path) -> dict | None:
    """Delete an evaluated merge and keep its manifest; the adapter can rebuild it."""
    if not merged.exists():
        return None
    manifest_path = merged / MERGE_MANIFEST
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if manifest.get("status") != "merged_and_verified":
        raise ValueError(f"Not a verified merge, left in place: {merged}")
    removed = remove_tree(merged)
    released = {
        "merged_model_path": str(merged),
        "bytes": sum(entry["bytes"] for entry in removed),
        "released_utc": utc_now().isoformat(),
        "manifest": manifest,
    }
    save_json(receipt, released)
    return released


# --- Free space


def require_free_space(paths: list[Path], minimum_gib: float) -> None:
    """Fail before a phase starts when any of its disks is below the reviewed minimum."""
    for path in paths:
        existing = nearest_existing(path)
        free = shutil.disk_usage(existing)[2]
        if free < minimum_gib * GIB:
            raise OSError(
                f"Disk below storage.minimum_free_gib at {existing}: "
                f"{free / GIB:.1f} GiB free, {minimum_gib} GiB required"
            )


def nearest_existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path
