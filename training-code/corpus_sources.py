"""Solidity import remappings and the source closure of a scope, read as data only."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import tomllib
from collections import deque
from pathlib import Path, PurePosixPath

# Comments, string literals, identifiers and single symbols.
SOLIDITY_TOKENS = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r"[A-Za-z_$][\w$]*|[^\s]"
)


def collect_sources(root, scope, mappings):
    root = Path(root).resolve()
    pending = deque(scope)
    hashes, edges, unresolved = {}, [], []
    prefixes = sorted(mappings, key=len, reverse=True)
    while pending:
        relative = posixpath.normpath(pending.popleft())
        if relative in hashes:
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source path escapes snapshot")
        raw = path.read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        tokens = [
            token
            for token in SOLIDITY_TOKENS.findall(raw.decode())
            if not token.startswith(("//", "/*"))
        ]
        index = 0
        while index < len(tokens):
            if tokens[index] != "import":
                index += 1
                continue
            end = index + 1
            while end < len(tokens) and tokens[end] != ";":
                end += 1
            literals = [token for token in tokens[index + 1 : end] if token.startswith(('"', "'"))]
            if len(literals) != 1 or end == len(tokens):
                raise ValueError(f"Unrecognized Solidity import in {relative}")
            imported = literals[0][1:-1]
            if "\\" in imported:
                raise ValueError("Escaped import literals require manual review")
            resolved = imported
            if imported.startswith("."):
                resolved = posixpath.join(posixpath.dirname(relative), imported)
            else:
                for prefix in prefixes:
                    if imported.startswith(prefix):
                        resolved = mappings[prefix] + imported[len(prefix) :]
                        break
            resolved = posixpath.normpath(resolved)
            dependency = (root / resolved).resolve()
            if not dependency.is_relative_to(root):
                raise ValueError("Import path escapes snapshot")
            edge = {"from": relative, "import": imported, "resolved": resolved}
            if dependency.is_file():
                edges.append(edge)
                pending.append(resolved)
            else:
                unresolved.append(edge)
            index = end + 1
    return {"source_hashes": hashes, "import_edges": edges, "unresolved_imports": unresolved}


def project_mappings(root):
    root = Path(root).resolve()
    mappings = {}
    for path in root.rglob("remappings.txt"):
        if len(path.relative_to(root).parts) > 3:
            continue
        prefix = path.parent.relative_to(root).as_posix()
        prefix = "" if prefix == "." else prefix + "/"
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                if ":" not in key:
                    mappings[key] = prefix + value
    config = root / "foundry.toml"
    if config.exists():
        foundry = tomllib.loads(config.read_text())
        for remapping in foundry.get("profile", {}).get("default", {}).get("remappings", []):
            key, value = remapping.split("=", 1)
            if ":" not in key:
                mappings[key] = value
    manifest = root / "package.json"
    if manifest.exists():
        workspaces = json.loads(manifest.read_text()).get("workspaces", [])
        if isinstance(workspaces, dict):
            workspaces = workspaces.get("packages", [])
        for pattern in workspaces:
            if PurePosixPath(pattern).is_absolute() or ".." in PurePosixPath(pattern).parts:
                raise ValueError("Workspace path escapes snapshot")
            for workspace in root.glob(pattern):
                if not workspace.resolve().is_relative_to(root):
                    raise ValueError("Workspace path escapes snapshot")
                package = workspace / "package.json"
                if package.exists():
                    name = json.loads(package.read_text()).get("name")
                    if name:
                        mappings[name + "/"] = str(workspace.relative_to(root)) + "/"
    return installed_mappings(root, mappings)


def installed_mappings(root, base):
    mappings = dict(base)
    for vendor in (Path(root) / "node_modules").glob("*"):
        mappings.setdefault(vendor.name + "/", "node_modules/" + vendor.name + "/")
    return mappings
