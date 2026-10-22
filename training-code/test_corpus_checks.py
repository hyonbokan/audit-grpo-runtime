import hashlib

import pytest

import corpus_checks


def candidate():
    text = "contract Vault {}"
    return {
        "row_id": "vault/all-scope",
        "repository": "https://github.com/acme/vault",
        "revision": "a" * 40,
        "family_ids": ["vault"],
        "split": "train",
        "scope_paths": ["Vault.sol"],
        "unresolved_imports": [],
        "source_files": [
            {
                "path": "Vault.sol",
                "content": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "targets": [
            {
                "archive_id": 1,
                "archive_record": {
                    "id": 1,
                    "title": "Loss",
                    "content": "Visible loss",
                    "impact": "HIGH",
                },
                "contracts": ["Vault.sol"],
                "reference_revisions": ["a" * 40],
            }
        ],
    }


def test_split_isolation_rejects_shared_application_files_and_families():
    a, b = candidate(), candidate()
    b.update(
        row_id="another/all-scope",
        repository="https://github.com/other/repo",
        family_ids=["another"],
        split="evaluation",
    )
    with pytest.raises(ValueError, match="application overlap"):
        corpus_checks.check_isolation([a, b])
    b["source_files"][0] = {
        "path": "Other.sol",
        "content": "other",
        "sha256": hashlib.sha256(b"other").hexdigest(),
    }
    b["targets"][0]["archive_id"] = 2
    corpus_checks.check_isolation([a, b])
    b["family_ids"] = a["family_ids"]
    with pytest.raises(ValueError, match="family overlap"):
        corpus_checks.check_isolation([a, b])


def test_formatting_and_comments_do_not_hide_duplicate_application_code_across_splits():
    a, b = candidate(), candidate()
    b.update(
        row_id="fork/all-scope",
        repository="https://github.com/other/fork",
        family_ids=["fork"],
        split="evaluation",
    )
    b["targets"][0]["archive_id"] = 2
    text = "// reformatted fork\ncontract  Vault{\n}\n"
    b["source_files"][0].update(content=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError):
        corpus_checks.check_isolation([a, b])


def test_reviewed_shared_library_fingerprint_does_not_join_unrelated_families():
    a, b = candidate(), candidate()
    b.update(
        row_id="other/all-scope",
        repository="https://github.com/other/project",
        family_ids=["other"],
        split="calibration",
    )
    b["targets"][0]["archive_id"] = 2
    with pytest.raises(ValueError):
        corpus_checks.check_isolation([a, b])
    corpus_checks.check_isolation(
        [a, b],
        vendor_fingerprints={
            corpus_checks.application_fingerprint(a["source_files"][0]["content"])
        },
    )


def test_archive_identity_mismatch_is_rejected_before_policy_export():
    row = {
        "scorer_metadata": {
            "targets": [
                {"id": "archive:1", "finding": {"archive_record": {"id": 1, "content": "original"}}}
            ]
        }
    }
    corpus_checks.verify_targets(row, {1: {"id": 1, "content": "original"}})
    with pytest.raises(ValueError):
        corpus_checks.verify_targets(row, {1: {"id": 1, "content": "changed"}})
