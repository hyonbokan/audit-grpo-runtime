"""Native benchmark inputs stay complete, ordered, and separate from private labels."""

import hashlib
import json

import pytest


def bundle(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    paths = ["B.sol", "A.sol"]
    records = []
    for name in paths:
        path = root / "sources/solidity/tiny" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"contract {name[0]} {{ string value = '$& $$'; }}")
        records.append(
            {
                "lane": "solidity",
                "repository": "tiny",
                "scope_path": name,
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    prompt = root / "benchmark/src/data/model_scan_prompt.ts"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("export const MODEL_SCAN_PROMPT = `HEADER\n{ contracts }\nEND`;")
    judge = prompt.with_name("calculate_prompt.ts")
    judge.write_text("export const CALCULATE_PROMPT = `{ junior_findings }\n{ truth_finding }`;")
    private = root / "private.json"
    private.write_text(
        json.dumps(
            [
                {
                    "Issue": "PRIVATE_SENTINEL",
                    "Description": "secret",
                    "Severity": "High",
                    "Contract": ["A.sol"],
                    "Submitted": ["auditor"],
                    "Link": "https://example/issue",
                }
            ]
        )
    )
    manifest = {
        "role": "evaluation_test_only",
        "sources": records,
        "repos": [
            {
                "lane": "solidity",
                "name": "tiny",
                "role": "evaluation_test_only",
                "contractFiles": paths,
                "ground_truth_path": "private.json",
                "ground_truth_count": 1,
                "repositoryURL": "https://github.com/example/tiny",
                "resolved_commit": "a" * 40,
            }
        ],
    }
    (root / "BENCHMARK_MANIFEST.json").write_text(json.dumps(manifest))
    return root


def test_native_prompt_keeps_source_order_and_private_labels_out(tmp_path):
    from benchmark_contract import load_assets

    assets = load_assets(bundle(tmp_path), "solidity", "tiny")
    assert assets["prompt"].index("// File: B.sol") < assets["prompt"].index("// File: A.sol")
    assert "$& $$" in assets["prompt"]
    assert "PRIVATE_SENTINEL" not in assets["prompt"]
    assert assets["ground_truth"][0]["issue"] == "PRIVATE_SENTINEL"
    assert assets["ground_truth"][0]["submitted"] == ["auditor"]
    assert assets["ground_truth"][0]["link"] == "https://example/issue"
    assert len(assets["sources"]) == 2


def test_reviewed_scope_rejects_ground_truth_for_omitted_contract(tmp_path):
    from benchmark_contract import load_assets

    root = bundle(tmp_path)
    manifest_path = root / "BENCHMARK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["repos"][0]["require_complete_gt_scope"] = True
    manifest_path.write_text(json.dumps(manifest))
    truth = json.loads((root / "private.json").read_text())
    truth[0]["Contract"] = ["Omitted.sol"]
    (root / "private.json").write_text(json.dumps(truth))
    with pytest.raises(ValueError, match="outside supplied scope"):
        load_assets(root, "solidity", "tiny")


def test_reviewed_truth_hash_rejects_label_tampering(tmp_path):
    from benchmark_contract import load_assets

    root = bundle(tmp_path)
    manifest_path = root / "BENCHMARK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["repos"][0]["ground_truth_sha256"] = hashlib.sha256(
        (root / "private.json").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    truth = json.loads((root / "private.json").read_text())
    truth[0]["Description"] = "Different target"
    (root / "private.json").write_text(json.dumps(truth))
    with pytest.raises(ValueError, match="Private GT identity"):
        load_assets(root, "solidity", "tiny")


@pytest.mark.parametrize("change", ["missing", "tampered", "duplicate", "escaped"])
def test_incomplete_or_corrupt_configured_scope_is_rejected(tmp_path, change):
    from benchmark_contract import load_assets

    root = bundle(tmp_path)
    manifest_path = root / "BENCHMARK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "missing":
        (root / manifest["sources"][0]["path"]).unlink()
    elif change == "tampered":
        (root / manifest["sources"][0]["path"]).write_text("other source")
    elif change == "duplicate":
        manifest["sources"].append(manifest["sources"][0])
    else:
        manifest["sources"][0]["path"] = "../../outside.sol"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_assets(root, "solidity", "tiny")


def test_empty_native_json_is_valid_zero_and_markdown_description_is_preserved():
    from benchmark_contract import parse_completion

    assert parse_completion('{"vulnerabilities":[]}', "stop") == []
    finding = {
        "issue": "x",
        "description": "### Impact\nFunds lost",
        "severity": "High",
        "contracts": ["A.sol"],
    }
    assert parse_completion(json.dumps({"vulnerabilities": [finding]}), "stop") == [finding]


@pytest.mark.parametrize(
    "text,reason",
    [
        ('{"findings":[]}', "stop"),
        ('{"vulnerabilities":[]}', "length"),
        ('```json\n{"vulnerabilities":[]}\n```', "stop"),
        ('{"vulnerabilities":[{"issue":"x"}]}', "stop"),
    ],
)
def test_invalid_or_truncated_completion_is_not_a_zero_grade(text, reason):
    from benchmark_contract import parse_completion

    with pytest.raises(ValueError):
        parse_completion(text, reason)


def test_markdown_findings_preserve_explanation_and_do_not_split_code_headings():
    from benchmark_contract import parse_completion

    text = """## Finding 1: Missing reset
**Severity:** High
**Contracts:** `A.sol`, `lib/B.sol`

### Root cause
The transfer leaves `dirty` set.
```solidity
## Finding 99: this is inside a code example
```
### Impact
Rewards are skipped.

## Finding 2: Unchecked deadline
Severity: Low
Contracts: A.sol

The deadline may be in the past.
"""
    findings = parse_completion(text, "stop", output_format="markdown")
    assert len(findings) == 2
    assert findings[0]["issue"] == "Missing reset"
    assert findings[0]["severity"] == "High"
    assert findings[0]["contracts"] == ["A.sol", "lib/B.sol"]
    assert "The transfer leaves `dirty` set." in findings[0]["description"]
    assert "## Finding 99: this is inside a code example" in findings[0]["description"]
    assert findings[1]["description"] == "The deadline may be in the past."


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "stop"),
        ("No vulnerabilities found.", "length"),
        ("## Finding 1: x\nSeverity: High\nExplanation", "stop"),
    ],
)
def test_markdown_missing_or_truncated_output_is_unknown_not_zero(text, reason):
    from benchmark_contract import parse_completion

    with pytest.raises(ValueError):
        parse_completion(text, reason, output_format="markdown")


def test_markdown_explicit_empty_report_is_valid():
    from benchmark_contract import parse_completion

    assert parse_completion("No vulnerabilities found.", "stop", output_format="markdown") == []


def test_markdown_policy_keeps_complete_code_and_excludes_json_and_truth(tmp_path):
    from benchmark_contract import load_assets

    root = bundle(tmp_path)
    assets = load_assets(root, "solidity", "tiny", output_format="markdown")
    assert assets["prompt"].index("// File: B.sol") < assets["prompt"].index("// File: A.sol")
    assert "$& $$" in assets["prompt"]
    assert "PRIVATE_SENTINEL" not in assets["prompt"]
    assert "JSON" not in assets["prompt"]
    assert "Markdown" in assets["prompt"]


def test_markdown_document_title_is_not_a_finding_and_preserves_body():
    from benchmark_contract import parse_completion

    report = (
        "# Security Audit Report: `A.sol`\n\n## Finding 1: Missing reset\n"
        "Severity: High\nContracts: `A.sol`\n\nThe dirty flag remains set."
    )
    rows = parse_completion(report, "stop", output_format="markdown")
    assert len(rows) == 1
    assert rows[0]["issue"] == "Missing reset"
    assert rows[0]["description"] == "The dirty flag remains set."


def test_unsectioned_findings_before_markdown_sections_are_not_silently_dropped():
    from benchmark_contract import parse_completion

    report = (
        "A vulnerability occurs in unsectioned text.\n\n"
        "## Finding 1: x\nSeverity: High\nContracts: A.sol\n\nbody"
    )
    with pytest.raises(ValueError):
        parse_completion(report, "stop", output_format="markdown")
