"""Split isolation and exact archive joins for frozen training corpora."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from corpus_sources import SOLIDITY_TOKENS
from runtime_io import sha256_bytes


def application_fingerprint(content: str) -> str:
    tokens = SOLIDITY_TOKENS.findall(content)
    body = [t for t in tokens if not t.startswith(("//", "/*"))]
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def check_isolation(
    candidates: list[dict[str, Any]], *, vendor_fingerprints: set[str] | None = None
) -> None:
    owners: dict[tuple[str, str], str] = {}
    labels: dict[int, str] = {}
    for c in candidates:
        keys = [("repository", c["repository"].lower().removesuffix(".git"))]
        keys += [("family", f) for f in c["family_ids"]]
        for f in c["source_files"]:
            path = f["path"].lower()
            if not any(
                x in path
                for x in (
                    "node_modules/",
                    "openzeppelin/",
                    "@openzeppelin/",
                    "/lib/",
                    "lib/",
                    "solmate/",
                    "solady/",
                    "interfaces/",
                )
            ):
                fingerprint = application_fingerprint(f["content"])
                if fingerprint not in (vendor_fingerprints or set()):
                    keys.append(("application", f["sha256"]))
                    keys.append(("normalized_application", fingerprint))
        for key in keys:
            if key in owners and owners[key] != c["split"]:
                raise ValueError(f"Cross-split {key[0]} overlap: {key[1]}")
            owners[key] = c["split"]
        for t in c.get("targets", []):
            identity = t["archive_id"]
            if identity in labels and labels[identity] != c["split"]:
                raise ValueError("Cross-split finding overlap")
            labels[identity] = c["split"]


def check_benchmark_overlap(
    rows: list[dict[str, Any]], benchmark: dict[str, Any], benchmark_row_ids: list[str]
) -> None:
    """Refuse rows that hold a file byte-identical to a file in an evaluation project's scope."""
    wanted = {tuple(row_id.split("/", 1)): row_id for row_id in benchmark_row_ids}
    owners: dict[str, list[str]] = {}
    for source in benchmark["sources"]:
        project = wanted.get((source["lane"], source["repository"]))
        if project:
            owners.setdefault(source["sha256"], []).append(project)
    missing = set(wanted.values()) - {p for projects in owners.values() for p in projects}
    if missing:
        raise ValueError(
            f"Evaluation projects missing from the benchmark manifest: {sorted(missing)}"
        )
    shared = [
        f"{row['row_id']}: {file['path']} is in {sorted(set(owners[digest]))}"
        for row in rows
        for file in row["source_files"]
        if (digest := sha256_bytes(file["content"].encode())) in owners
    ]
    if shared:
        raise ValueError("Training rows share files with evaluation projects: " + "; ".join(shared))


def verify_targets(row: dict[str, Any], archive: dict[int, Any]) -> None:
    for t in row["scorer_metadata"]["targets"]:
        record = t["finding"]["archive_record"]
        if t["id"] != f"archive:{record['id']}" or archive.get(record["id"]) != record:
            raise ValueError("Private target does not equal its source archive record")
