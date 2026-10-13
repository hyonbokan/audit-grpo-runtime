"""Automated human-audit label admission and dependency-aware scope grouping."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from corpus_checks import check_isolation, verify_targets
from corpus_sources import collect_sources, project_mappings
from grpo.policy_prompt import policy_template_options, prompt_token_ids
from runtime_io import save_json as save

# The production audit prompt and answer schema; git-ignored, supplied separately.
POLICY_RENDERING = Path(__file__).resolve().parent / "private/policy_rendering.json"


def admit_human_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    files = candidate["source_files"]
    paths = {f["path"] for f in files}
    if not files or len(paths) != len(files):
        raise ValueError("Empty or duplicate source scope")
    if not re.fullmatch(r"[0-9a-f]{40}", candidate["revision"]):
        raise ValueError("Immutable audited revision required")
    for f in files:
        if hashlib.sha256(f["content"].encode()).hexdigest() != f["sha256"]:
            raise ValueError("Source hash differs")
    if candidate.get("unresolved_imports"):
        raise ValueError("Group has unresolved dependencies")
    targets = []
    for t in candidate["targets"]:
        record = t["archive_record"]
        if candidate["revision"] not in t["reference_revisions"]:
            raise ValueError("Finding revision does not match audited code")
        if not t["contracts"] or not set(t["contracts"]).issubset(paths):
            raise ValueError("Finding references code outside the group")
        if record["impact"] not in ("HIGH", "MEDIUM") or not record.get("source"):
            raise ValueError("Human audit severity and provenance required")
        targets.append(
            {
                "id": f"archive:{t['archive_id']}",
                "required_context": t.get("required_context", []),
                "finding": {
                    "issue": record["title"],
                    "description": record["content"],
                    "contracts": t["contracts"],
                    "archive_record": record,
                    "eligibility_review": {
                        "review_origin": "human_audit_metadata",
                        "status": "source_aligned_human_label",
                        "semantic_llm_review": False,
                        "context_applicability": "Unreviewed unless supplied by existing metadata",
                    },
                },
            }
        )
    if not targets:
        raise ValueError("Nonempty human-audit GT required")
    return {
        "row_id": candidate["row_id"],
        "repository": candidate["repository"],
        "revision": candidate["revision"],
        "group_id": candidate["row_id"].rsplit("/", 1)[-1],
        "source_files": files,
        "complementary_context": {},
        "scorer_metadata": {"targets": targets},
        "profile": "none",
    }


def group_candidate(
    candidate: dict[str, Any],
    token_count: Callable[[list[dict]], int],
    *,
    maximum_prompt_tokens: int,
) -> tuple[list[dict], dict]:
    """Partition core roots by measured prompt size, retaining their import closures."""
    files = {f["path"]: f for f in candidate["source_files"]}
    graph = defaultdict(set)
    for edge in candidate["import_edges"]:
        graph[edge["from"]].add(edge["resolved"])
    missing = {e["from"] for e in candidate.get("unresolved_imports", [])}
    receipt = {"blocked_core_paths": [], "unassigned_target_ids": []}
    if not missing and token_count(list(files.values())) <= maximum_prompt_tokens:
        return [candidate], receipt

    def closure(core):
        pending, found = [core], set()
        while pending:
            path = pending.pop()
            if path in found:
                continue
            if path not in files or path in missing:
                raise ValueError("Missing dependency")
            found.add(path)
            pending.extend(graph[path])
        return found

    bundles, current, roots, covered = [], set(), [], set()
    for core in sorted(candidate["core_paths"]):
        if core in covered:
            continue
        try:
            paths = closure(core)
            if token_count([files[p] for p in sorted(paths)]) > maximum_prompt_tokens:
                raise ValueError("Indivisible dependency closure exceeds prompt budget")
        except ValueError as exc:
            receipt["blocked_core_paths"].append({"path": core, "reason": str(exc)})
            continue
        union = current | paths
        if current and token_count([files[p] for p in sorted(union)]) > maximum_prompt_tokens:
            bundles.append((current, roots))
            current, roots = set(), []
        current |= paths
        roots.append(core)
        covered |= paths
    if current:
        bundles.append((current, roots))
    groups = []
    for paths, cores in bundles:
        group = "group-" + hashlib.sha256(json.dumps(cores).encode()).hexdigest()[:12]
        groups.append(
            {
                **candidate,
                "row_id": candidate["row_id"].rsplit("/", 1)[0] + "/" + group,
                "core_paths": cores,
                "source_files": [files[p] for p in sorted(paths)],
                "targets": [],
                "unresolved_imports": [],
            }
        )
    for target in candidate["targets"]:
        assigned = False
        for group in groups:
            paths = {f["path"] for f in group["source_files"]}
            if set(target["contracts"]).issubset(paths):
                group["targets"].append(target)
                assigned = True
                break
        if not assigned:
            receipt["unassigned_target_ids"].append(target["archive_id"])
    return [g for g in groups if g["targets"]], receipt


def core_path(path: str) -> bool:
    parts = Path(path).parts
    return (
        path.endswith(".sol")
        and not path.endswith(".t.sol")
        and not any(
            p.lower()
            in {"lib", "libs", "node_modules", "vendor", "test", "tests", "script", "scripts"}
            for p in parts
        )
    )


def source_mappings(root: Path) -> dict:
    mappings = project_mappings(root)
    for base in ("lib", "libs"):
        for package in (root / base).glob("*"):
            if not package.is_dir():
                continue
            directory = package / "src" if (package / "src").is_dir() else package
            mappings.setdefault(package.name + "/", directory.relative_to(root).as_posix() + "/")
    return mappings


def dependency_closure(root: Path, scope: list[str]) -> dict:
    """Resolve undeclared package aliases only when local source bytes are unambiguous."""
    mappings = source_mappings(root)
    available = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.sol")
        if p.is_file() and not p.is_symlink()
    )
    while True:
        closure = collect_sources(root, scope, mappings)
        added = False
        for missing in closure["unresolved_imports"]:
            imported = missing["import"]
            if imported.startswith(".") or imported in mappings:
                continue
            parts = imported.split("/")
            suffixes = [imported, "/".join(parts[1:])]
            if imported.startswith("@") and len(parts) > 2:
                suffixes.append("/".join(parts[2:]))
            for suffix in suffixes:
                if "/" not in suffix:
                    continue
                matches = [p for p in available if p == suffix or p.endswith("/" + suffix)]
                if not matches:
                    continue
                digests = {hashlib.sha256((root / p).read_bytes()).hexdigest() for p in matches}
                if len(digests) == 1:
                    mappings[imported] = matches[0]
                    added = True
                break
        if not added:
            return closure


def load_archive_candidate(receipt: dict, records: dict, root: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", receipt["revision"]):
        raise ValueError("Source revision is not immutable")
    paths = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.sol")
        if p.is_file() and not p.is_symlink()
    )
    scope = receipt.get("scope_candidates") or [p for p in paths if core_path(p)]
    scope = sorted(p for p in scope if p.endswith(".sol") and (root / p).is_file())
    if not scope:
        raise ValueError("No available Solidity application scope")
    closure = dependency_closure(root, scope)
    files = []
    original_hashes = {**receipt.get("file_hashes", {}), **receipt.get("source_hashes", {})}
    for path, digest in sorted(closure["source_hashes"].items()):
        if path in original_hashes and original_hashes[path] != digest:
            raise ValueError("Recovered source differs from immutable source receipt")
        files.append(
            {"path": path, "content": (root / path).read_bytes().decode(), "sha256": digest}
        )
    targets, roles, families = [], set(), set()
    for identity in receipt["archive_ids"]:
        record = records.get(identity)
        if (
            not record
            or record.get("duplicate_of")
            or record["archive_record"]["impact"] not in ("HIGH", "MEDIUM")
        ):
            continue
        refs = [
            r
            for r in record["code_references"]
            if r["repository"] == receipt["repository"]
            and r["revision"] == receipt["revision"]
            and r["path"] in closure["source_hashes"]
            and r["path"].endswith(".sol")
            and not r.get("vendor")
        ]
        if not refs:
            continue
        roles.add(record["split"])
        families.add(record["family_id"])
        targets.append(
            {
                "archive_id": identity,
                "archive_record": record["archive_record"],
                "contracts": sorted({r["path"] for r in refs}),
                "reference_revisions": [receipt["revision"]],
            }
        )
    if not targets or len(roles) != 1:
        raise ValueError("No nonempty source-aligned single-split human-audit targets")
    return {
        "row_id": receipt["job_id"] + "/all-scope",
        "repository": receipt["repository"],
        "revision": receipt["revision"],
        "split": roles.pop(),
        "family_ids": sorted(families),
        "core_paths": scope,
        "source_files": files,
        "targets": targets,
        "import_edges": closure["import_edges"],
        "unresolved_imports": closure["unresolved_imports"],
    }


def policy_rendering_bytes() -> bytes:
    if not POLICY_RENDERING.is_file():
        raise FileNotFoundError(
            f"{POLICY_RENDERING} is not in git; copy the frozen policy rendering there"
        )
    return POLICY_RENDERING.read_bytes()


def policy_messages(row: dict) -> list[dict]:
    rendering = json.loads(policy_rendering_bytes())
    contracts = "\n\n".join(f"// File: {f['path']}\n{f['content']}" for f in row["source_files"])
    messages = [
        {"role": "system", "content": rendering["system_prompt"]},
        {
            "role": "user",
            "content": rendering["user_prefix"] + contracts + rendering["user_suffix"],
        },
    ]
    messages[1]["content"] += (
        "\n\nFinal answer contract: return a JSON object matching the following schema. "
        "No Markdown fences. Use an empty findings list when no issue is found.\n"
        + json.dumps(rendering["output_schema"], ensure_ascii=False)
    )
    return messages


def build(config: dict, out: Path) -> None:
    from transformers import AutoTokenizer

    archive = Path(config["archive"])
    records = {
        r["archive_id"]: r
        for r in map(
            json.loads, (archive / "index_v3/candidates.private.jsonl").read_text().splitlines()
        )
    }
    staging_data = json.loads(Path(config["staging"]).read_text())
    if isinstance(staging_data, dict):
        staging_data = staging_data.get("records", staging_data.get("vulnerabilities"))
    staging = {r["id"]: r for r in staging_data}
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    plan = json.loads(Path(config["source_plan"]).read_text())
    options = policy_template_options(plan)
    if options["reasoning_effort"] != "xhigh" or options["enable_thinking"] is not True:
        raise ValueError("Automated corpus requires explicit xhigh thinking")
    roles = {s: [] for s in ("train", "calibration", "evaluation", "regression")}
    identities, decisions, used_repos = [], [], set()
    # Chosen cap, not the model limit. The maximum is maximum_context minus prompt length;
    # a lower value truncates answers.
    maximum_completion = config["maximum_completion_tokens"]
    out.mkdir(parents=True, exist_ok=True)

    def token_count(files):
        dummy = {
            "row_id": "scope/all-scope",
            "repository": "scope",
            "revision": "a" * 40,
            "group_id": "all-scope",
            "source_files": files,
            "scorer_metadata": {"targets": []},
            "profile": "none",
        }
        return len(prompt_token_ids(tokenizer, policy_messages(dummy), template_options=options))

    def append(candidate):
        split = candidate["split"]
        groups, receipt = group_candidate(
            candidate,
            token_count,
            maximum_prompt_tokens=(
                config["maximum_training_prompt_tokens"]
                if split == "train"
                else config["maximum_context"] - maximum_completion
            ),
        )
        admitted = 0
        for group in groups:
            if not any(not t.get("required_context", []) for t in group["targets"]):
                continue
            row = admit_human_audit(group)
            verify_targets(row, staging)
            identity = {
                **group,
                "targets": [{"archive_id": t["archive_id"]} for t in group["targets"]],
            }
            try:
                check_isolation(
                    [*identities, identity],
                    vendor_fingerprints=set(config.get("vendor_fingerprints", [])),
                )
            except ValueError as exc:
                decisions.append(
                    {
                        "row_id": row["row_id"],
                        "status": "withheld_split_overlap",
                        "reason": str(exc),
                    }
                )
                continue
            roles[split].append(row)
            identities.append(identity)
            admitted += 1
        decisions.append(
            {
                "row_id": candidate["row_id"],
                "repository": candidate["repository"],
                "split": split,
                "status": "admitted" if admitted else "withheld_no_usable_group",
                "groups": admitted,
                **receipt,
            }
        )
        if admitted:
            used_repos.add(candidate["repository"])

    # Preserve existing role assignments and source-aligned labels, including context annotations.
    seed = Path(config["seed_dataset"])
    for split in ("calibration", "evaluation", "regression", "train"):
        for row in map(json.loads, (seed / f"data/private/{split}.jsonl").read_text().splitlines()):
            root = out / "seed-source" / hashlib.sha256(row["row_id"].encode()).hexdigest()[:20]
            for f in row["source_files"]:
                path = root / f["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f["content"])
            cores = [f["path"] for f in row["source_files"] if core_path(f["path"])]
            closure = dependency_closure(root, cores)
            targets = [
                {
                    "archive_id": t["finding"]["archive_record"]["id"],
                    "archive_record": t["finding"]["archive_record"],
                    "contracts": t["finding"]["contracts"],
                    "required_context": t["required_context"],
                    "reference_revisions": [row["revision"]],
                }
                for t in row["scorer_metadata"]["targets"]
            ]
            append(
                {
                    "row_id": row["row_id"],
                    "repository": row["repository"],
                    "revision": row["revision"],
                    "split": split,
                    "family_ids": sorted({records[t["archive_id"]]["family_id"] for t in targets}),
                    "core_paths": cores,
                    "source_files": row["source_files"],
                    "targets": targets,
                    "import_edges": closure["import_edges"],
                    "unresolved_imports": closure["unresolved_imports"],
                }
            )

    jobs = [json.loads(p.read_text()) for p in (archive / "source_final/results").glob("*.json")]
    jobs.sort(key=lambda j: (-len(j.get("archive_ids", [])), j["job_id"]))
    excluded = config["excluded_repository_names"]
    for job in jobs:
        repo = job.get("repository", "")
        name = repo.rsplit("/", 1)[-1].lower()
        if (
            job.get("status") != "recovered_candidate"
            or job.get("splits") != ["train"]
            or repo in used_repos
            or any(re.search(r"(?:^|[-_])" + re.escape(x) + r"(?:$|[-_])", name) for x in excluded)
        ):
            continue
        try:
            relative = job["source_root"].split("full_archive_2026-10-06/", 1)[1]
            root = archive / relative
            for mirror in config["recovered_mirrors"]:
                alternative = Path(mirror) / job["job_id"]
                if alternative.is_dir():
                    root = alternative
                    break
            candidate = load_archive_candidate(job, records, root)
            if candidate["split"] == "train":
                append(candidate)
        except (ValueError, OSError, KeyError) as exc:
            decisions.append(
                {
                    "job_id": job["job_id"],
                    "repository": repo,
                    "status": "withheld_mechanical",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if len(decisions) % 25 == 0:
            save(out / "admission.private.json", decisions)
            print(
                json.dumps(
                    {
                        "processed_decisions": len(decisions),
                        "training_examples": len(roles["train"]),
                        "training_projects": len({r["repository"] for r in roles["train"]}),
                    }
                ),
                flush=True,
            )
        if len({r["repository"] for r in roles["train"]}) >= config.get(
            "target_training_projects", 100
        ):
            break

    requests = {}
    for split, rows in roles.items():
        private, policy = [], []
        for row in rows:
            messages = policy_messages(row)
            ids = prompt_token_ids(tokenizer, messages, template_options=options)
            rendered_prompt = tokenizer.decode(ids, skip_special_tokens=False)
            if (
                "Reasoning effort is set to xhigh." not in rendered_prompt
                or "Reasoning effort is set to low." in rendered_prompt
            ):
                raise ValueError("Actual rendered prompt does not contain xhigh instructions")
            context = max(36864, ((len(ids) + maximum_completion + 4095) // 4096) * 4096)
            if context > config["maximum_context"]:
                raise ValueError("Grouped prompt unexpectedly exceeds context")
            request = {
                "row_id": row["row_id"],
                "model": {
                    "id": config["model_id"],
                    "revision": config["model_revision"],
                    "precision": "bf16",
                },
                "prompt_ids": ids,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "full_completion_ceiling": maximum_completion,
                "max_model_len": context,
                "template_options": options,
                "chat_template_sha256": hashlib.sha256(
                    tokenizer.chat_template.encode()
                ).hexdigest(),
            }
            name = hashlib.sha256(row["row_id"].encode()).hexdigest()[:20] + ".json"
            save(out / "requests" / name, request)
            requests[row["row_id"]] = {
                "context": context,
                "prompt_tokens": len(ids),
                "container_path": config["dataset_container"] + "/requests/" + name,
                "request_sha256": hashlib.sha256(
                    (out / "requests" / name).read_bytes()
                ).hexdigest(),
            }
            private.append(row)
            policy.append({"row_id": row["row_id"], "messages": messages})
        for kind, values in (("private", private), ("policy", policy)):
            path = out / f"data/{kind}/{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
    save(out / "admission.private.json", decisions)
    manifest = {
        "version": "audited-xhigh-20261009-v1",
        "admission_policy": (
            "Human audit metadata; immutable code reference; automatic dependency grouping; "
            "no second LLM semantic review"
        ),
        "policy_template_options": options,
        "maximum_completion_tokens": maximum_completion,
        "policy_rendering_sha256": hashlib.sha256(policy_rendering_bytes()).hexdigest(),
        "roles": {s: [r["row_id"] for r in rows] for s, rows in roles.items()},
        "requests": requests,
        "counts": {
            s: {
                "rows": len(rows),
                "projects": len({r["repository"] for r in rows}),
                "targets": sum(len(r["scorer_metadata"]["targets"]) for r in rows),
                "code_only_targets": sum(
                    not t["required_context"] for r in rows for t in r["scorer_metadata"]["targets"]
                ),
            }
            for s, rows in roles.items()
        },
        "prior_optimizer_visits": {"2022-12-caviar/all-scope": 2},
        "limitations": [
            "New human labels are not re-reviewed for code-only applicability",
            "Known GT is sparse",
            "Import closure does not prove dynamic call coverage",
        ],
        "files": {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    }
    save(out / "manifest.json", manifest)
    print(json.dumps(manifest["counts"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
