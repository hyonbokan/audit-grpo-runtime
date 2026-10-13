import hashlib
import pytest
import corpus


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


def review():
    return {
        "scope_verified": True,
        "scope_rationale": "Complete published scope",
        "targets": [
            {
                "archive_id": 1,
                "accepted": True,
                "final_verdict": "HIGH",
                "required_context": [],
                "rationale": "Visible transfer is unchecked",
            }
        ],
    }


def test_export_requires_review_and_keeps_ground_truth_private():
    assert hasattr(corpus, "admit"), "Reviewed corpus admission is missing"
    row = corpus.admit(candidate(), review())
    assert row["group_id"] == "all-scope"
    assert row["scorer_metadata"]["targets"][0]["finding"]["archive_record"]["id"] == 1
    assert row["scorer_metadata"]["targets"][0]["required_context"] == []


@pytest.mark.parametrize("change", ["hash", "revision", "imports", "scope", "verdict", "context"])
def test_bad_evidence_cannot_be_admitted(change):
    assert hasattr(corpus, "admit"), "Reviewed corpus admission is missing"
    c, r = candidate(), review()
    if change == "hash":
        c["source_files"][0]["content"] += "changed"
    if change == "revision":
        c["targets"][0]["reference_revisions"] = ["b" * 40]
    if change == "imports":
        c["unresolved_imports"] = ["Missing.sol"]
    if change == "scope":
        c["scope_paths"].append("Missing.sol")
    if change == "verdict":
        r["targets"][0]["final_verdict"] = "INVALID"
    if change == "context":
        r["targets"][0]["required_context"] = ["documentation"]
    with pytest.raises(ValueError):
        corpus.admit(c, r)


def test_split_isolation_rejects_family_and_application_duplicates_but_not_vendor():
    assert hasattr(corpus, "check_isolation"), "Corpus isolation check is missing"
    a, b = candidate(), candidate()
    b.update(
        row_id="another/all-scope",
        repository="https://github.com/other/repo",
        family_ids=["another"],
        split="evaluation",
    )
    with pytest.raises(ValueError):
        corpus.check_isolation([a, b])
    b["source_files"][0] = {
        "path": "Other.sol",
        "content": "other",
        "sha256": hashlib.sha256(b"other").hexdigest(),
    }
    b["targets"][0]["archive_id"] = 2
    corpus.check_isolation([a, b])
    b["family_ids"] = a["family_ids"]
    with pytest.raises(ValueError):
        corpus.check_isolation([a, b])


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
        corpus.check_isolation([a, b])


def test_reviewed_shared_library_fingerprint_does_not_join_unrelated_families():
    a, b = candidate(), candidate()
    b.update(
        row_id="other/all-scope",
        repository="https://github.com/other/project",
        family_ids=["other"],
        split="calibration",
    )
    b["targets"][0]["archive_id"] = 2
    assert hasattr(corpus, "application_fingerprint"), (
        "Normalized application fingerprint is missing"
    )
    with pytest.raises(ValueError):
        corpus.check_isolation([a, b])
    corpus.check_isolation(
        [a, b],
        vendor_fingerprints={corpus.application_fingerprint(a["source_files"][0]["content"])},
    )
