"""Split isolation and exact archive joins for frozen training corpora."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from corpus_sources import SOLIDITY_TOKENS


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


def verify_targets(row: dict[str, Any], archive: dict[int, Any]) -> None:
    for t in row["scorer_metadata"]["targets"]:
        record = t["finding"]["archive_record"]
        if t["id"] != f"archive:{record['id']}" or archive.get(record["id"]) != record:
            raise ValueError("Private target does not equal its source archive record")
