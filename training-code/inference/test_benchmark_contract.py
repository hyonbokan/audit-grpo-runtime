"""Benchmark inputs stay complete, ordered, and separate from private labels."""

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
    judge = root / "benchmark/src/data/calculate_prompt.ts"
    judge.parent.mkdir(parents=True)
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


def test_audit_prompt_keeps_source_order_and_private_labels_out(tmp_path):
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
