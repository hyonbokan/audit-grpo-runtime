"""Frozen code-only benchmark rendering and native JSON validation, without GPU/API work."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def grading_input_hashes(samples: bytes, truth: bytes, template: bytes) -> dict[str, str]:
    """Bind grades to the exact files read, including any explicit normalization."""
    return {
        "samples": hashlib.sha256(samples).hexdigest(),
        "private_truth": hashlib.sha256(truth).hexdigest(),
        "judge_prompt": hashlib.sha256(template).hexdigest(),
    }


def load_assets(root: Path, lane: str, name: str, *, output_format: str = "json") -> dict[str, Any]:
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
    template = read_template(root / "benchmark/src/data/model_scan_prompt.ts", "MODEL_SCAN_PROMPT")
    prompt = template.replace("{ contracts }", flat, 1)
    if output_format == "markdown":
        prompt = markdown_prompt(flat)
    elif output_format != "json":
        raise ValueError("Unknown benchmark output format")
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


def parse_completion(
    text: str, finish_reason: str, *, output_format: str = "json"
) -> list[dict[str, Any]]:
    if finish_reason != "stop":
        raise ValueError(f"Incomplete benchmark response: {finish_reason}")
    if output_format == "markdown":
        return parse_markdown(text)
    if output_format != "json":
        raise ValueError("Unknown benchmark output format")
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"vulnerabilities"}:
        raise ValueError("Native vulnerabilities JSON object required")
    findings = value["vulnerabilities"]
    if not isinstance(findings, list):
        raise ValueError("Native vulnerabilities array required")
    for finding in findings:
        validate_finding(finding)
    return findings


def markdown_prompt(flat: str) -> str:
    return f"""You are a smart contracts security expert tasked with scanning the contracts
in the context for security issues. Analyse their logic and purpose and try to find
all security issues in the provided contracts.

## Contracts
```solidity
{flat}
```

## Output format
Return a Markdown audit report. For each distinct vulnerability use this format:

## Finding 1: Issue title
Severity: High | Medium | Low | Info (choose one)
Contracts: `relative/path.sol`, `another/path.sol`

Explain the root cause, relevant code and functions, conditions needed to trigger
the issue, and its impact. Include a concrete execution example where useful.
Use the exact contract paths from the provided file headers. Number findings in
order, one finding per section. Subheadings and code examples are welcome within
each finding. If there are no findings, return exactly: No vulnerabilities found.
"""


def parse_markdown(text: str) -> list[dict[str, Any]]:
    """Extract report sections without rewriting or supplementing their claims."""
    text = text.strip()
    if text == "No vulnerabilities found.":
        return []
    # Find section boundaries outside fenced examples; preserve original body bytes.
    lines = text.splitlines(keepends=True)
    masked, fence = [], None
    for line in lines:
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)[0]
            fence = None if fence == token else (token if fence is None else fence)
            masked.append(" " * len(line.rstrip("\n")) + ("\n" if line.endswith("\n") else ""))
        else:
            masked.append(
                line
                if fence is None
                else " " * len(line.rstrip("\n")) + ("\n" if line.endswith("\n") else "")
            )
    headers = list(re.finditer(r"^#{2,3}\s+Finding\s+\d+\s*:\s*(.+)$", "".join(masked), re.M))
    preamble = text[: headers[0].start()].strip() if headers else ""
    if not headers or any(
        line.strip() and not re.fullmatch(r"#\s+.+", line.strip()) for line in preamble.splitlines()
    ):
        raise ValueError("Markdown finding sections or explicit empty report required")
    findings = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[header.end() : end].strip()
        metadata = {}
        description_lines = []
        for line in body.splitlines():
            match = re.match(r"^(?:\*\*)?(Severity|Contracts):(?:\*\*)?\s*(.+)$", line)
            if match and len(metadata) < 2 and match.group(1) not in metadata:
                metadata[match.group(1)] = match.group(2).strip()
            else:
                description_lines.append(line)
        if set(metadata) != {"Severity", "Contracts"}:
            raise ValueError("Markdown finding lacks severity or contract paths")
        finding = {
            "issue": header.group(1).strip(),
            "severity": metadata["Severity"].strip("*` "),
            "contracts": [part.strip("` ") for part in metadata["Contracts"].split(",")],
            "description": "\n".join(description_lines).strip(),
        }
        validate_finding(finding)
        if not finding["issue"] or not finding["description"] or not all(finding["contracts"]):
            raise ValueError("Markdown finding is incomplete")
        findings.append(finding)
    return findings


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
