"""The free-text corpus: the evaluation's prompt over the same rows, splits and labels."""

import json

import pytest

import benchmark_free_text as free_text
import free_text_corpus
from runtime_io import read_json, sha256_bytes, sha256_file

ROW = "abc/group-1"
SOURCE_FILES = [
    {"path": "src/A.sol", "content": "contract A {}", "sha256": "0" * 64},
    {"path": "src/B.sol", "content": "contract B {}", "sha256": "1" * 64},
]


def target(identity, impact, required_context=()):
    return {
        "id": identity,
        "required_context": list(required_context),
        "finding": {
            "issue": f"issue {identity}",
            "description": f"description {identity}",
            "contracts": ["src/A.sol"],
            "archive_record": {"impact": impact, "id": identity},
            "eligibility_review": {"status": "ok"},
        },
    }


class FakeTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize, return_dict, **options):
        assert tokenize and not return_dict
        return [ord(c) for c in "".join(m["content"] for m in messages)][:50]


def private_row(targets):
    return {
        "row_id": ROW,
        "repository": "abc",
        "revision": "f" * 40,
        "group_id": "group-1",
        "source_files": SOURCE_FILES,
        "complementary_context": None,
        "scorer_metadata": {"targets": targets},
        "profile": "none",
    }


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "data/private").mkdir(parents=True)
    (root / "data/policy").mkdir()
    (root / "requests").mkdir()
    rows = {"train": [private_row([target("t1", "HIGH"), target("t2", "MEDIUM", ["docs"])])]}
    for split in free_text_corpus.SPLITS:
        lines = [json.dumps(row) for row in rows.get(split, [])]
        (root / f"data/private/{split}.jsonl").write_text("\n".join(lines))
        (root / f"data/policy/{split}.jsonl").write_text(
            "\n".join(
                json.dumps({"row_id": r["row_id"], "messages": []}) for r in rows.get(split, [])
            )
        )
    request = {
        "row_id": ROW,
        "prompt_ids": [1],
        "prompt_ids_sha256": "x",
        "full_completion_ceiling": 5,
        "max_model_len": 57344,
        "template_options": {"reasoning_effort": "xhigh"},
        "chat_template_sha256": sha256_bytes(b"template"),
    }
    (root / "requests/r.json").write_text(json.dumps(request))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "old",
                "policy_rendering_sha256": "p" * 64,
                "roles": {"train": [ROW], "calibration": [], "evaluation": [], "regression": []},
                "requests": {ROW: {"container_path": "/results/old/requests/r.json"}},
                "files": {},
            }
        )
    )
    return root


def test_policy_message_is_the_evaluation_prompt_over_the_same_files():
    row = private_row([])
    assert free_text_corpus.policy_messages(row) == [
        {
            "role": "user",
            "content": free_text.audit_prompt(free_text.render_sources(SOURCE_FILES)),
        }
    ]


def test_truth_uses_the_evaluation_shape_and_drops_targets_needing_unshown_context():
    row = private_row([target("t1", "HIGH"), target("t2", "MEDIUM", ["docs"])])
    assert free_text_corpus.truth_findings(row) == [
        {
            "issue": "issue t1",
            "severity": "High",
            "description": "description t1",
            "contracts": ["src/A.sol"],
            "recommendation": "",
        }
    ]


def test_an_unknown_impact_is_refused_with_the_allowed_values():
    row = private_row([target("t1", "CRITICAL")])
    with pytest.raises(ValueError, match=r"CRITICAL.*HIGH"):
        free_text_corpus.truth_findings(row)


def test_derive_writes_prompts_truth_and_requests_that_match_them(source, tmp_path):
    out = tmp_path / "derived"
    free_text_corpus.derive(source, out, FakeTokenizer(), "template")
    policy = [
        json.loads(line) for line in (out / "data/policy/train.jsonl").read_text().splitlines()
    ]
    assert policy == [
        {"row_id": ROW, "messages": free_text_corpus.policy_messages(private_row([]))}
    ]
    private = [
        json.loads(line) for line in (out / "data/private/train.jsonl").read_text().splitlines()
    ]
    assert private[0]["truth"] == free_text_corpus.truth_findings(
        private_row([target("t1", "HIGH")])
    )
    assert private[0]["truth_ids"] == ["t1"]
    request = read_json(out / "requests/r.json")
    expected = FakeTokenizer().apply_chat_template(policy[0]["messages"], True, False)
    assert request["prompt_ids"] == expected
    assert request["prompt_ids_sha256"] == sha256_bytes(json.dumps(expected).encode())
    assert request["full_completion_ceiling"] == 57344 - len(expected)


def test_the_judge_template_is_frozen_into_the_corpus(source, tmp_path):
    out = tmp_path / "derived"
    free_text_corpus.derive(source, out, FakeTokenizer(), "{ truth_finding }")
    assert (out / "judge-prompt.txt").read_text() == "{ truth_finding }"
    assert "judge-prompt.txt" in read_json(out / "manifest.json")["files"]


def test_derived_manifest_names_its_source_and_hashes_every_file(source, tmp_path):
    out = tmp_path / "derived"
    free_text_corpus.derive(source, out, FakeTokenizer(), "template")
    manifest = read_json(out / "manifest.json")
    assert "policy_rendering_sha256" not in manifest
    assert manifest["derived_from_manifest_sha256"] == sha256_file(source / "manifest.json")
    assert manifest["roles"]["train"] == [ROW]
    assert manifest["requests"] == {ROW: {"container_path": "/results/old/requests/r.json"}}
    assert set(manifest["files"]) == {
        f"data/{kind}/{split}.jsonl"
        for kind in ("policy", "private")
        for split in free_text_corpus.SPLITS
    } | {"requests/r.json", "judge-prompt.txt"}
    for relative, digest in manifest["files"].items():
        assert sha256_file(out / relative) == digest


def test_a_request_from_a_different_chat_template_is_refused(source, tmp_path):
    path = source / "requests/r.json"
    request = read_json(path)
    request["chat_template_sha256"] = "0" * 64
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="chat template"):
        free_text_corpus.derive(source, tmp_path / "derived", FakeTokenizer(), "template")


def test_derive_refuses_an_existing_output(source, tmp_path):
    out = tmp_path / "derived"
    out.mkdir()
    with pytest.raises(FileExistsError):
        free_text_corpus.derive(source, out, FakeTokenizer(), "template")


def test_a_train_row_without_a_shown_label_is_refused(source, tmp_path):
    row = private_row([target("t2", "MEDIUM", ["docs"])])
    (source / "data/private/train.jsonl").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="no label"):
        free_text_corpus.derive(source, tmp_path / "derived", FakeTokenizer(), "template")
