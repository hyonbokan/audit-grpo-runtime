"""Rebuild complete source scopes and export private immutable-reference candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import certifi
from corpus_dependencies import exact_package, extract_sources, locked_package, project_mappings
from corpus_sources import collect_sources


def visible_targets(candidate, records, paths):
    targets = []
    for identity in candidate["archive_ids"]:
        record = records[identity]
        if record["archive_record"]["impact"] not in ("HIGH", "MEDIUM"):
            continue
        refs = [
            r
            for r in record["code_references"]
            if r["repository"] == candidate["repository"]
            and r["revision"] == candidate["revision"]
            and r["path"].endswith(".sol")
            and r["path"] in paths
            and not r.get("vendor")
        ]
        if refs:
            targets.append(
                {
                    "archive_id": identity,
                    "archive_record": record["archive_record"],
                    "contracts": sorted({r["path"] for r in refs}),
                    "reference_revisions": [candidate["revision"]],
                }
            )
    return targets


def fetch(url):
    request = Request(url, headers={"User-Agent": "audit-grpo-corpus/1"})
    with urlopen(
        request, timeout=30, context=ssl.create_default_context(cafile=certifi.where())
    ) as response:
        raw = response.read(32 * 1024**2 + 1)
    if len(raw) > 32 * 1024**2:
        raise ValueError("Dependency download exceeds source-data limit")
    return raw


def recover_package(root, imported, cache):
    parts = imported.split("/")
    name = "/".join(parts[:2]) if imported.startswith("@") else parts[0]
    if not re.fullmatch(r"(?:@[\w.-]+/)?[\w.-]+", name):
        raise ValueError("Import is not an npm package")
    pins = []
    for manifest in sorted(root.rglob("package.json")):
        if (
            "node_modules" in manifest.relative_to(root).parts
            or "lib" in manifest.relative_to(root).parts
        ):
            continue
        data = json.loads(manifest.read_text())
        selector = {**data.get("devDependencies", {}), **data.get("dependencies", {})}.get(name)
        if selector is None:
            continue
        npm = manifest.parent / "package-lock.json"
        yarn = manifest.parent / "yarn.lock"
        try:
            pin = locked_package(
                name,
                selector,
                npm=json.loads(npm.read_text()) if npm.is_file() else None,
                yarn=yarn.read_text() if yarn.is_file() else None,
            )
        except ValueError:
            if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[\w.-]+)?", selector):
                continue
            metadata = json.loads(fetch(f"https://registry.npmjs.org/{name}/{selector}"))
            pin = exact_package(name, selector, metadata)
        pins.append(pin)
    unique = {json.dumps(pin, sort_keys=True) for pin in pins}
    if len(unique) != 1:
        raise ValueError("No unambiguous project-pinned package version")
    pin = json.loads(unique.pop())
    key = hashlib.sha256(json.dumps(pin, sort_keys=True).encode()).hexdigest()[:24]
    destination = cache / key
    receipt_path = cache / f"{key}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
    else:
        raw = fetch(pin["resolved"].split("#")[0])
        hashes = extract_sources(raw, pin["integrity"], destination)
        meta = json.loads((destination / "package.json").read_text())
        if meta.get("name") != name or meta.get("version") != pin["version"]:
            raise ValueError("Downloaded package identity mismatch")
        receipt = {
            "package": name,
            **pin,
            "files": hashes,
            "download_sha256": hashlib.sha256(raw).hexdigest(),
            "scripts_executed": False,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2))
    for p, digest in receipt["files"].items():
        source = destination / p
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Recovered dependency cache drift")
        target = root / "node_modules" / name / p
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("Recovery would overwrite different source bytes")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return receipt


def build(config, out):
    archive, previous = Path(config["archive"]), Path(config["previous"])
    records = {
        r["archive_id"]: r
        for r in map(
            json.loads, (archive / "index_v3/candidates.private.jsonl").read_text().splitlines()
        )
    }
    catalog = list(
        map(json.loads, (previous / "replacement-catalog.private.jsonl").read_text().splitlines())
    )
    reports = [
        json.loads(p.read_text())
        for p in (archive / "reports_verified_metadata/results").glob("*.json")
    ]
    already = set(config.get("already_reviewed", []))
    candidates, screening = [], []
    cache = out / "dependency-cache"
    cache.mkdir(parents=True, exist_ok=True)
    for c in sorted(catalog, key=lambda x: x["scope_bytes_before_dependencies"]):
        j = c["job_id"]
        if j in already or c["split_conflict"] or c["overlaps_original"] or len(c["splits"]) != 1:
            continue
        scope = c["scope_review_candidates"]
        original = archive / c["source_root"].split("full_archive_2026-10-06/")[-1]
        prior = previous / "recovered-scopes" / j
        selected = prior if prior.is_dir() else original
        root = out / "recovered-scopes" / j
        root.mkdir(parents=True, exist_ok=False)
        for source in selected.rglob("*"):
            if not source.is_file() or source.is_symlink():
                continue
            if not (
                source.suffix == ".sol"
                or source.name
                in (
                    "package.json",
                    "package-lock.json",
                    "yarn.lock",
                    "remappings.txt",
                    "foundry.toml",
                    "README.md",
                    "scope.txt",
                )
            ):
                continue
            path = source.relative_to(selected)
            if ".git" in path.parts:
                continue
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        # Some existing mirrors contain recovered source only; original manifests remain needed for immutable package pins.
        for source in original.rglob("*"):
            if (
                source.is_file()
                and not source.is_symlink()
                and source.name
                in (
                    "package.json",
                    "package-lock.json",
                    "yarn.lock",
                    "remappings.txt",
                    "foundry.toml",
                    "README.md",
                    "scope.txt",
                )
            ):
                target = root / source.relative_to(original)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        receipt = {
            "job_id": j,
            "repository": c["repository"],
            "packages": [],
            "recovery_errors": {},
        }
        try:
            closure = collect_sources(root, scope, project_mappings(root))
            attempted = set()
            for _ in range(3):
                imports = sorted(
                    {
                        e["import"]
                        for e in closure["unresolved_imports"]
                        if not e["import"].startswith(".")
                    }
                )
                fresh = [i for i in imports if i not in attempted]
                if not fresh:
                    break
                for imported in fresh:
                    attempted.add(imported)
                    try:
                        receipt["packages"].append(recover_package(root, imported, cache))
                    except Exception as exc:
                        receipt["recovery_errors"][imported] = f"{type(exc).__name__}: {exc}"
                closure = collect_sources(root, scope, project_mappings(root))
            files = [
                {"path": p, "content": (root / p).read_text(), "sha256": h}
                for p, h in sorted(closure["source_hashes"].items())
            ]
            targets = visible_targets(c, records, {f["path"] for f in files})
            receipt.update(
                unresolved_imports=closure["unresolved_imports"],
                visible_targets=len(targets),
                source_bytes=sum(len(f["content"].encode()) for f in files),
            )
            if (
                closure["unresolved_imports"]
                or not targets
                or receipt["source_bytes"] > config["maximum_source_bytes"]
            ):
                receipt["status"] = "withheld_mechanical_evidence"
            else:
                docs = [
                    {
                        "path": str(p.relative_to(root)),
                        "content": p.read_text(),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in [root / "README.md", root / "scope.txt"]
                    if p.is_file()
                ]
                texts = []
                ids = set(c["archive_ids"])
                for report in reports:
                    if (
                        report.get("status") != "recovered_report"
                        or not ids.intersection(report["archive_ids"])
                        or "/reports/" not in report["url"]
                    ):
                        continue
                    p = archive / report["text_path"].split("full_archive_2026-10-06/")[-1]
                    if hashlib.sha256(p.read_bytes()).hexdigest() != report["text_sha256"]:
                        raise ValueError("Published report text drift")
                    texts.append(
                        {
                            "url": report["url"],
                            "sha256": report["text_sha256"],
                            "content": p.read_text(),
                        }
                    )
                candidates.append(
                    {
                        "job_id": j,
                        "row_id": f"{j}/all-scope",
                        "repository": c["repository"],
                        "revision": c["revision"],
                        "family_ids": c["family_ids"],
                        "split": c["splits"][0],
                        "scope_paths": scope,
                        "unresolved_imports": [],
                        "source_files": files,
                        "targets": targets,
                        "scope_documents": docs,
                        "published_reports": texts,
                        "source_evidence": str(archive / f"source_final/results/{j}.json"),
                    }
                )
                receipt["status"] = "ready_for_semantic_review"
        except Exception as exc:
            receipt.update(
                status="withheld_mechanical_evidence", error=f"{type(exc).__name__}: {exc}"
            )
        screening.append(receipt)
        (out / "mechanical-screen.private.json").write_text(json.dumps(screening, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in receipt.items()
                    if k not in ("packages", "recovery_errors", "unresolved_imports")
                }
            ),
            flush=True,
        )
    # Select one immutable scope per repository before training; retain all recovery evidence.
    chosen = {}
    for c in candidates:
        if c["repository"] not in chosen or len(c["targets"]) > len(
            chosen[c["repository"]]["targets"]
        ):
            chosen[c["repository"]] = c
    (out / "candidates.private.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chosen.values())
    )
    print(json.dumps({"semantic_review_candidates": len(chosen)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
