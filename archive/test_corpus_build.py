import corpus_build


def test_target_join_excludes_mutable_other_revision_missing_and_non_solidity_references():
    assert hasattr(corpus_build, "visible_targets"), "Immutable visible target join is missing"
    c = {"repository": "repo", "revision": "a" * 40, "archive_ids": [1, 2, 3]}
    records = {}
    for identity, path, rev in [
        (1, "A.sol", "a" * 40),
        (2, "A.sol", "main"),
        (3, "B.rs", "a" * 40),
    ]:
        records[identity] = {
            "archive_record": {"id": identity, "impact": "HIGH"},
            "code_references": [{"repository": "repo", "revision": rev, "path": path}],
        }
    targets = corpus_build.visible_targets(c, records, {"A.sol", "B.rs"})
    assert [t["archive_id"] for t in targets] == [1]
    assert targets[0]["archive_record"] == records[1]["archive_record"]
    assert corpus_build.visible_targets(c, records, set()) == []
