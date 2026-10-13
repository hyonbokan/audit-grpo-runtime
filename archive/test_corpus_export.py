import pytest
import corpus_export


def test_archive_identity_mismatch_is_rejected_before_policy_export():
    assert hasattr(corpus_export, "verify_targets"), "Exact archive join is missing"
    row = {
        "scorer_metadata": {
            "targets": [
                {"id": "archive:1", "finding": {"archive_record": {"id": 1, "content": "original"}}}
            ]
        }
    }
    corpus_export.verify_targets(row, {1: {"id": 1, "content": "original"}})
    with pytest.raises(ValueError):
        corpus_export.verify_targets(row, {1: {"id": 1, "content": "changed"}})
