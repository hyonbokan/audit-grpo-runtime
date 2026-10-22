"""Frozen benchmark assets: the free-text audit prompt, private truth and judge prompt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark_free_text import audit_prompt


def grading_input_hashes(samples: bytes, truth: bytes, template: bytes) -> dict[str, str]:
    """Bind grades to the exact files read, including any explicit normalization."""
    return {
        "samples": hashlib.sha256(samples).hexdigest(),
        "private_truth": hashlib.sha256(truth).hexdigest(),
        "judge_prompt": hashlib.sha256(template).hexdigest(),
    }


def load_assets(root: Path, lane: str, name: str) -> dict[str, Any]:
    if lane not in ("solidity", "solidity2"):
        raise ValueError("Only supplied Solidity evaluation lanes are admitted")
    manifest = json.loads((root / "BENCHMARK_MANIFEST.json").read_text())
    repos = [row for row in manifest["repos"] if (row["lane"], row["name"]) == (lane, name)]
    if len(repos) != 1 or manifest["role"] != "evaluation_test_only":
        raise ValueError("One evaluation-only repository required")
    repo = repos[0]
    records = [
        row for row in manifest["sources"] if (row["lane"], row["repository"]) == (lane, name)
    ]
    if len(records) != len(repo["contractFiles"]):
        raise ValueError("Incomplete or duplicate configured source scope")
    sources = []
    for scope_path in repo["contractFiles"]:
        matching = [row for row in records if row["scope_path"] == scope_path]
        if len(matching) != 1:
            raise ValueError("Each configured file must appear exactly once")
        record = matching[0]
        path = confined_path(root, record["path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Source identity differs: {scope_path}")
        sources.append(
            {"path": scope_path, "content": data.decode("utf-8"), "sha256": record["sha256"]}
        )
    separator = "=" * 80
    flat = "\n\n".join(
        f"{separator}\n// File: {row['path']}\n{separator}\n\n{row['content']}" for row in sources
    )
    prompt = audit_prompt(flat)
    truth_data = confined_path(root, repo["ground_truth_path"]).read_bytes()
    if (
        repo.get("ground_truth_sha256")
        and hashlib.sha256(truth_data).hexdigest() != repo["ground_truth_sha256"]
    ):
        raise ValueError("Private GT identity differs")
    truth = json.loads(truth_data)
    if not isinstance(truth, list) or len(truth) != repo["ground_truth_count"]:
        raise ValueError("Private GT denominator differs")
    normalized = []
    for item in truth:
        row = {
            key: item.get(key, item.get(alias))
            for key, alias in (
                ("issue", "Issue"),
                ("severity", "Severity"),
                ("description", "Description"),
                ("contracts", "Contract"),
            )
        }
        validate_finding(row)
        if repo.get("require_complete_gt_scope") and any(
            path not in repo["contractFiles"] for path in row["contracts"]
        ):
            raise ValueError("Private GT contract outside supplied scope")
        row["recommendation"] = item.get("recommendation", item.get("Recommendation", ""))
        for key, alias in (
            ("submitted", "Submitted"),
            ("link", "Link"),
            ("detector", "Detector"),
            ("mitigation", "Mitigation"),
        ):
            value = item.get(key, item.get(alias))
            if value is not None:
                row[key] = value
        normalized.append(row)
    return {
        "repo": repo,
        "sources": sources,
        "prompt": prompt,
        "ground_truth": normalized,
        "judge_prompt": read_template(
            root / "benchmark/src/data/calculate_prompt.ts", "CALCULATE_PROMPT"
        ),
    }


def read_template(path: Path, constant: str) -> str:
    text = path.read_text()
    prefix = f"export const {constant} = `"
    if text.count(prefix) != 1:
        raise ValueError("Native benchmark template declaration differs")
    template = text.split(prefix, 1)[1].split("`;", 1)[0]
    # Supplied templates only escape literal backticks; reject JS interpolation/other escapes.
    if "${" in template or "\\" in template.replace("\\`", ""):
        raise ValueError("Unsupported native template syntax")
    return template.replace("\\`", "`")


def confined_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes the frozen package")
    return path


def validate_finding(row: Any) -> None:
    if not isinstance(row, dict) or any(
        not isinstance(row.get(key), str) for key in ("issue", "description", "severity")
    ):
        raise ValueError("Native finding requires string issue, description, severity")
    if row["severity"] not in ("High", "Medium", "Low", "Info"):
        raise ValueError("Native finding severity differs")
    if not isinstance(row.get("contracts"), list) or any(
        not isinstance(path, str) for path in row["contracts"]
    ):
        raise ValueError("Native finding requires contract paths")
