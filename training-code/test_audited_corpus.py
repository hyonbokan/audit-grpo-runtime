import hashlib
import json

import pytest

import audited_corpus


def candidate():
    contents = {
        "A.sol": "contract A {}",
        "B.sol": "contract B {}",
        "lib/Shared.sol": "library S {}",
    }
    return {
        "row_id": "audit/all-scope",
        "repository": "https://github.com/code-423n4/example",
        "revision": "a" * 40,
        "split": "train",
        "family_ids": ["example"],
        "core_paths": ["A.sol", "B.sol"],
        "source_files": [
            {"path": p, "content": c, "sha256": hashlib.sha256(c.encode()).hexdigest()}
            for p, c in contents.items()
        ],
        "import_edges": [
            {"from": "A.sol", "resolved": "lib/Shared.sol"},
            {"from": "B.sol", "resolved": "lib/Shared.sol"},
        ],
        "unresolved_imports": [],
        "targets": [
            {
                "archive_id": i,
                "archive_record": {
                    "id": i,
                    "title": "Known issue",
                    "content": "Human audit",
                    "impact": "MEDIUM",
                    "source": "Code4rena",
                    "source_link": "https://code4rena.com/reports/example",
                },
                "contracts": [p],
                "reference_revisions": ["a" * 40],
            }
            for i, p in [(1, "A.sol"), (2, "B.sol")]
        ],
    }


def characters(files):
    return sum(len(f["content"]) for f in files)


def test_human_audit_targets_are_admitted_without_a_second_semantic_review():
    row = audited_corpus.admit_human_audit(candidate())
    assert [t["id"] for t in row["scorer_metadata"]["targets"]] == ["archive:1", "archive:2"]
    c = candidate()
    c["targets"][0]["reference_revisions"] = ["b" * 40]
    with pytest.raises(ValueError, match="revision"):
        audited_corpus.admit_human_audit(c)


def test_oversized_scope_separates_core_contracts_and_reuses_dependency_closure():
    groups, receipt = audited_corpus.group_candidate(
        candidate(), characters, maximum_prompt_tokens=25
    )
    assert len(groups) == 2
    assert [{f["path"] for f in g["source_files"]} for g in groups] == [
        {"A.sol", "lib/Shared.sol"},
        {"B.sol", "lib/Shared.sol"},
    ]
    assert [t["archive_id"] for g in groups for t in g["targets"]] == [1, 2]
    assert receipt["unassigned_target_ids"] == []


def test_grouping_does_not_silently_drop_an_indivisible_dependency_or_cross_group_target():
    c = candidate()
    c["import_edges"].append({"from": "A.sol", "resolved": "B.sol"})
    groups, receipt = audited_corpus.group_candidate(c, characters, maximum_prompt_tokens=25)
    assert len(groups) == 1
    assert [t["archive_id"] for t in groups[0]["targets"]] == [2]
    assert receipt["unassigned_target_ids"] == [1]
    c = candidate()
    c["targets"][0]["contracts"] = ["A.sol", "B.sol"]
    groups, receipt = audited_corpus.group_candidate(c, characters, maximum_prompt_tokens=25)
    assert receipt["unassigned_target_ids"] == [1]


def test_whole_scope_is_retained_when_it_fits_and_missing_dependencies_are_recorded():
    c = candidate()
    groups, receipt = audited_corpus.group_candidate(c, characters, maximum_prompt_tokens=100)
    assert len(groups) == 1 and len(groups[0]["source_files"]) == 3
    c["unresolved_imports"] = [{"from": "A.sol", "import": "Missing.sol"}]
    groups, receipt = audited_corpus.group_candidate(c, characters, maximum_prompt_tokens=25)
    assert [t["archive_id"] for g in groups for t in g["targets"]] == [2]
    assert receipt["unassigned_target_ids"] == [1]


def test_archive_loader_uses_all_application_sources_and_resolves_shared_dependencies(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "lib").mkdir()
    (root / "A.sol").write_text('import "./lib/Shared.sol"; contract A {}')
    (root / "B.sol").write_text('import "./lib/Shared.sol"; contract B {}')
    (root / "lib/Shared.sol").write_text("library S {}")
    c = candidate()
    receipt = {
        "job_id": "audit",
        "repository": c["repository"],
        "revision": c["revision"],
        "archive_ids": [1, 2],
        "scope_candidates": [],
    }
    records = {
        t["archive_id"]: {
            "archive_id": t["archive_id"],
            "archive_record": t["archive_record"],
            "split": "train",
            "family_id": "example",
            "duplicate_of": None,
            "code_references": [
                {
                    "repository": c["repository"],
                    "revision": c["revision"],
                    "path": t["contracts"][0],
                    "vendor": False,
                }
            ],
        }
        for t in c["targets"]
    }
    loaded = audited_corpus.load_archive_candidate(receipt, records, root)
    assert loaded["core_paths"] == ["A.sol", "B.sol"]
    assert {f["path"] for f in loaded["source_files"]} == {"A.sol", "B.sol", "lib/Shared.sol"}
    assert len(loaded["targets"]) == 2 and loaded["unresolved_imports"] == []


def test_policy_prompt_uses_the_reward_schema_without_exposing_human_gt(tmp_path, monkeypatch):
    rendering = {
        "origin": "synthetic",
        "system_prompt": "Audit the code.",
        "user_prefix": "Code:\n",
        "user_suffix": "\nEnd.",
        "output_schema": {"type": "object", "properties": {"findings": {"type": "array"}}},
    }
    path = tmp_path / "policy_rendering.json"
    path.write_text(json.dumps(rendering))
    monkeypatch.setattr(audited_corpus, "POLICY_RENDERING", path)
    row = audited_corpus.admit_human_audit(candidate())
    messages = audited_corpus.policy_messages(row)
    assert "archive:1" not in str(messages) and "Known issue" not in str(messages)
    assert '"findings"' in messages[-1]["content"]
    assert messages[0] == {"role": "system", "content": "Audit the code."}


def test_missing_policy_rendering_fails_with_where_to_put_it(tmp_path, monkeypatch):
    monkeypatch.setattr(audited_corpus, "POLICY_RENDERING", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="not in git"):
        audited_corpus.policy_messages(audited_corpus.admit_human_audit(candidate()))


def test_available_dependency_alias_is_resolved_without_fetching_or_mixing_versions(tmp_path):
    root = tmp_path / "source"
    (root / "lib/vendor/token").mkdir(parents=True)
    (root / "A.sol").write_text('import "@vendor/token/Token.sol"; contract A {}')
    (root / "B.sol").write_text("contract B {}")
    (root / "lib/vendor/token/Token.sol").write_text("contract Token {}")
    c = candidate()
    receipt = {
        "job_id": "audit",
        "repository": c["repository"],
        "revision": c["revision"],
        "archive_ids": [1, 2],
        "scope_candidates": ["A.sol", "B.sol"],
    }
    records = {
        t["archive_id"]: {
            "archive_id": t["archive_id"],
            "archive_record": t["archive_record"],
            "split": "train",
            "family_id": "example",
            "duplicate_of": None,
            "code_references": [
                {
                    "repository": c["repository"],
                    "revision": c["revision"],
                    "path": t["contracts"][0],
                    "vendor": False,
                }
            ],
        }
        for t in c["targets"]
    }
    loaded = audited_corpus.load_archive_candidate(receipt, records, root)
    assert loaded["unresolved_imports"] == []
    assert "lib/vendor/token/Token.sol" in {f["path"] for f in loaded["source_files"]}
    (root / "lib/another/token").mkdir(parents=True)
    (root / "lib/another/token/Token.sol").write_text("contract DifferentToken {}")
    loaded = audited_corpus.load_archive_candidate(receipt, records, root)
    assert loaded["unresolved_imports"], "Ambiguous source versions must not be guessed"
