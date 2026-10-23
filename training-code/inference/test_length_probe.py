"""Length probe: row choice, per-row saving and resume, and the cap analysis."""

import json
from types import SimpleNamespace

import pytest

import length_probe
from runtime_io import read_json, sha256_bytes, sha256_file
from testing.judge import ROW_ID, SOURCE_FILES, private_row


class FakeTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize, return_dict, **options):
        marker = ord("x") if options["reasoning_effort"] == "xhigh" else ord("m")
        return [marker, *(ord(c) for c in messages[0]["content"][:7])]

    def decode(self, ids, skip_special_tokens):
        return ""


def corpus_with(tmp_path, count):
    """A derived corpus of `count` train rows with growing source sizes, one request each."""
    root = tmp_path / "corpus"
    for kind in ("private", "policy"):
        (root / "data" / kind).mkdir(parents=True)
    (root / "requests").mkdir()
    ids, manifest_requests = [], {}
    private, policy = [], []
    for index in range(count):
        row = dict(private_row(), row_id=f"audit/group-{index:02d}")
        row["source_files"] = [dict(SOURCE_FILES[0], content="a" * (10 + index))]
        ids.append(row["row_id"])
        private.append(row)
        from free_text_corpus import policy_messages

        policy.append({"row_id": row["row_id"], "messages": policy_messages(row)})
        original = FakeTokenizer().apply_chat_template(
            policy[-1]["messages"], True, False, reasoning_effort="xhigh"
        )
        request = {
            "row_id": row["row_id"],
            "prompt_ids": original,
            "chat_template_sha256": sha256_bytes(b"template"),
            "template_options": {"reasoning_effort": "xhigh"},
            "max_model_len": 1000,
            "full_completion_ceiling": 1000 - len(original),
        }
        (root / "requests" / f"{index}.json").write_text(json.dumps(request))
        manifest_requests[row["row_id"]] = {"container_path": f"/results/c/requests/{index}.json"}
    for name, rows in (("private", private), ("policy", policy)):
        (root / f"data/{name}/train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {"files": {}, "requests": manifest_requests, "roles": {"train": ids}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def config_for(corpus, **changes):
    return {
        "corpus": str(corpus),
        "corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
        "model_path": "/model",
        "row_count": 2,
        "cap_total_tokens": 98304,
        "seconds_limit": 3600,
        "tensor_parallel_size": 2,
        **changes,
    }


def test_rows_are_spread_evenly_over_the_prompt_length_order():
    tokens = {f"r{i}": 100 * (i + 1) for i in range(10)}
    assert length_probe.select_rows(tokens, 1) == ["r5"]
    assert length_probe.select_rows(tokens, 2) == ["r2", "r7"]
    assert length_probe.select_rows(tokens, 10) == [f"r{i}" for i in range(10)]
    with pytest.raises(ValueError, match="11 rows"):
        length_probe.select_rows(tokens, 11)


def test_prepare_retokenizes_every_row_at_medium_and_saves_the_chosen_requests(tmp_path):
    corpus = corpus_with(tmp_path, 6)
    out = tmp_path / "probe"
    length_probe.prepare(config_for(corpus), out, FakeTokenizer())
    prompts = read_json(out / "prompts.json")
    assert sorted(prompts) == [f"audit/group-{i:02d}" for i in range(6)]
    assert list(prompts.values()) == sorted(prompts.values())
    rows = read_json(out / "manifest.json")["rows"]
    assert [row["row_id"] for row in rows] == ["audit/group-01", "audit/group-04"]
    for row in rows:
        request = read_json(row["request"])
        assert request["prompt_ids"][0] == ord("m")
        assert request["template_options"]["reasoning_effort"] == "medium"
        assert row["prompt_tokens"] == len(request["prompt_ids"]) == prompts[row["row_id"]]
        assert row["max_tokens"] == 98304 - row["prompt_tokens"]


def test_prepare_refuses_a_changed_corpus_a_bad_config_and_a_row_with_no_room(tmp_path):
    corpus = corpus_with(tmp_path, 3)
    with pytest.raises(ValueError, match="keys"):
        length_probe.prepare({**config_for(corpus), "extra": 1}, tmp_path / "a", FakeTokenizer())
    with pytest.raises(ValueError, match="manifest"):
        length_probe.prepare(
            config_for(corpus, corpus_manifest_sha256="0" * 64), tmp_path / "b", FakeTokenizer()
        )
    with pytest.raises(ValueError, match="no room"):
        length_probe.prepare(
            config_for(corpus, cap_total_tokens=3), tmp_path / "c", FakeTokenizer()
        )
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        length_probe.prepare(
            config_for(corpus, tensor_parallel_size=3), tmp_path / "d", FakeTokenizer()
        )


class FakeEngine:
    def __init__(self, lengths):
        self.lengths, self.calls = lengths, []

    def generate(self, prompts, params, use_tqdm):
        self.calls.append((len(prompts[0]["prompt_token_ids"]), params))
        draws = [
            SimpleNamespace(
                token_ids=list(range(length)),
                finish_reason="length" if length >= params["max_tokens"] else "stop",
                text="t",
            )
            for length in self.lengths
        ]
        return [SimpleNamespace(outputs=draws)]


def patch_engine(monkeypatch, engine):
    loads = []

    def load(arm, out, receipt):
        loads.append(arm)
        return engine, None, lambda **settings: settings

    monkeypatch.setattr(length_probe, "load_engine", load)
    monkeypatch.setattr(length_probe, "think_end_id", lambda _path: 3)
    return loads


def test_the_probe_saves_each_row_as_it_finishes_and_a_rerun_generates_only_the_rest(
    tmp_path, monkeypatch
):
    corpus = corpus_with(tmp_path, 6)
    out = tmp_path / "probe"
    config = config_for(corpus)
    length_probe.prepare(config, out, FakeTokenizer())
    engine = FakeEngine([5, 6, 7, 98304])
    loads = patch_engine(monkeypatch, engine)
    length_probe.run_probe(config, out)
    first = read_json(out / "rows/00.json")
    assert [d["completion_tokens"] for d in first["draws"]] == [5, 6, 7, 98304]
    assert [d["finish_reason"] for d in first["draws"]] == ["stop", "stop", "stop", "length"]
    assert first["draws"][0]["closed_reasoning"] is True and first["draws"][2]["answer_tokens"] == 3
    assert (loads[0]["config"]["engine"]["tensor_parallel_size"], len(engine.calls)) == (2, 2)
    assert engine.calls[0][1]["max_tokens"] == first["max_tokens"]
    assert engine.calls[0][1]["n"] == 4 and "logprobs" not in engine.calls[0][1]
    (out / "rows/01.json").unlink()
    again = FakeEngine([1, 1, 1, 1])
    patch_engine(monkeypatch, again)
    length_probe.run_probe(config, out)
    assert len(again.calls) == 1
    assert [d["completion_tokens"] for d in read_json(out / "rows/00.json")["draws"]] == [
        5,
        6,
        7,
        98304,
    ]
    done = FakeEngine([])
    loads = patch_engine(monkeypatch, done)
    length_probe.run_probe(config, out)
    assert loads == []


def row(row_id, prompt, lengths, cut=()):
    return {
        "row_id": row_id,
        "prompt_tokens": prompt,
        "seconds": 10.0,
        "draws": [
            {
                "completion_tokens": n,
                "finish_reason": "length" if i in cut else "stop",
                "reasoning_tokens": n - 1,
            }
            for i, n in enumerate(lengths)
        ],
    }


def test_the_summary_counts_cut_draws_and_what_each_total_cap_would_cost():
    rows = [
        row("a", 20000, [10000, 20000, 30000, 40000]),
        row("b", 60000, [5000, 6000, 7000, 38304], cut=[3]),
    ]
    prompts = {"a": 20000, "b": 60000, "c": 30000, "d": 90000}
    result = length_probe.summarize_probe(rows, prompts, 98304)
    assert result["draws"] == 8 and result["cut_at_probe_cap"] == 1
    tokens = result["completion_tokens"]
    assert (tokens["p50"], tokens["p90"], tokens["p95"], tokens["max"]) == (
        10000,
        40000,
        40000,
        40000,
    )
    assert tokens["censored_at_cap"] is True
    low = result["candidate_totals"]["65536"]
    # Only row b reaches 65536 in total: 66000, 67000 and the 98304 draw cut at the cap.
    assert low["draws_at_or_over_total"] == 3
    assert low["prompt_limit_for_p95_completion"] == 65536 - 40000
    assert low["rows_kept"] == 1
    top = result["candidate_totals"]["98304"]
    assert top["draws_at_or_over_total"] == 1 and top["rows_kept"] == 2


def test_a_total_above_the_probe_cap_is_not_estimated_and_an_empty_probe_is_refused():
    rows = [row("a", 1000, [10, 20, 30, 40])]
    result = length_probe.summarize_probe(rows, {"a": 1000}, 70000)
    assert list(result["candidate_totals"]) == ["65536"]
    with pytest.raises(ValueError, match="No probed rows"):
        length_probe.summarize_probe([], {}, 98304)


def test_percentiles_use_the_nearest_rank():
    values = list(range(1, 21))
    assert [length_probe.percentile(values, f) for f in (0.5, 0.9, 0.95)] == [10, 18, 19]
    assert length_probe.percentile([7], 0.95) == 7


def test_the_report_names_the_totals_rows_kept_and_every_probed_row(tmp_path, monkeypatch):
    corpus = corpus_with(tmp_path, 4)
    out = tmp_path / "probe"
    config = config_for(corpus, row_count=1)
    length_probe.prepare(config, out, FakeTokenizer())
    patch_engine(monkeypatch, FakeEngine([10, 20, 30, 40]))
    length_probe.run_probe(config, out)
    length_probe.write_report(config, out)
    markdown = (out / "REPORT.md").read_text()
    assert (
        "1 rows x 4 draws" in markdown
        and "| 65536 |" in markdown
        and ROW_ID.split("/")[0] in markdown
    )
    assert read_json(out / "report.json")["probed_rows"] == 1
