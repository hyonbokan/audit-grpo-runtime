import json
from pathlib import Path

import pytest

import published_results
from testing.benchmark import project

THINK_END = 99


def write_run(
    runs: Path, name: str, known: int, counts, *, arm="base", effort="xhigh", targets=None, **draw
):
    """A finished project run on disk whose `arm` has four graded draws.

    `counts` is (exact, partial, findings) for every draw, matching the same known vulnerabilities
    each time; `targets` instead gives each draw's (exact indices, partial indices, findings).
    """
    root = runs / f"run-{name}"
    (root / arm).mkdir(parents=True)
    if targets is None:
        exact, partial, findings = counts
        targets = [(range(exact), range(exact, exact + partial), findings)] * 4
    comparison = project(name, known, (0, 0, 0), (0, 0, 0))["comparison"]
    comparison["draws"] = {arm: [graded(known, *target) for target in targets]}
    for item in comparison["draws"][arm]:
        item.update(draw)
    (root / "comparison.json").write_text(json.dumps(comparison))
    prepared = {
        "row_id": f"solidity/{name}",
        "model_id": "Qwen/Qwen3.8-27B",
        "model_revision": "1d4bf0f2",
        "base_precision": "bf16",
        "benchmark_suite": "solidity_free_text",
        "template_options": {"reasoning_effort": effort},
        "sampling": {"n": 4, "temperature": 1.0, "max_tokens": 262144 - 1000 * known},
    }
    (root / "prepared.json").write_text(json.dumps(prepared))
    receipt = {"weight_version": "base:1d4bf0f2", "weight_representation": "base_bf16"}
    (root / arm / "receipt.json").write_text(json.dumps(receipt))
    samples = [
        {"completion_ids": [1, 2, 3, THINK_END, 4], "finish_reason": "stop"} for _ in range(4)
    ]
    (root / arm / "samples.json").write_text(json.dumps(samples))
    return root


def graded(known, exact, partial, findings):
    verdicts = [
        {"target_index": i, "is_match": i in exact, "is_partial_match": i in partial}
        for i in range(known)
    ]
    for verdict in verdicts:
        verdict.update(severity_from_truth="High", severity_from_junior_auditor="High")
    return {
        "status": "graded",
        "known_gt_count": known,
        "exact_matches": len(exact),
        "partial_matches": len(partial),
        "findings_count": findings,
        "unmatched_findings": findings - len(exact) - len(partial),
        "verdicts": verdicts,
    }


def publish(runs, out, roots, name="qwen-base-xhigh", arm="base", when="2026-10-22T00:00:00+00:00"):
    return published_results.publish(name, arm, roots, out, THINK_END, when)


def test_a_model_is_published_with_unique_matches_over_every_finding_reported(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    roots = [write_run(runs, "alpha", 10, (3, 1, 12)), write_run(runs, "beta", 20, (5, 2, 20))]
    publish(runs, out, roots)
    saved = json.loads((out / "qwen-base-xhigh/results.json").read_text())
    total = saved["total"]
    # The same vulnerabilities found in all four draws count once; findings count every draw.
    assert (total["exact"], total["partial"], total["findings"], total["known"]) == (8, 3, 128, 30)
    assert total["precision"] == 8 / 128
    assert total["recall"] == 8 / 30
    assert total["f1"] == pytest.approx(2 * (8 / 128) * (8 / 30) / (8 / 128 + 8 / 30))
    alpha = saved["projects"][0]
    assert (alpha["name"], alpha["exact"], alpha["findings"], alpha["precision"]) == (
        "alpha",
        3,
        48,
        3 / 48,
    )
    assert (alpha["mean_reasoning_tokens"], alpha["mean_answer_tokens"]) == (3, 1)
    assert saved["reasoning_effort"] == "xhigh"
    assert saved["weight_version"] == "base:1d4bf0f2"
    index = (out / "README.md").read_text()
    assert "| qwen-base-xhigh | xhigh | 8 | 3 | 128 | 0.062 | 0.267 | 0.101 |" in index


def test_a_vulnerability_counts_once_and_as_exact_if_any_draw_matched_it_exactly(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    targets = [([0], [], 5), ([1], [], 5), ([0], [1], 5), ([], [2], 5)]
    publish(runs, out, [write_run(runs, "alpha", 10, None, targets=targets)])
    alpha = json.loads((out / "qwen-base-xhigh/results.json").read_text())["projects"][0]
    assert (alpha["exact"], alpha["partial"], alpha["findings"]) == (2, 1, 20)


def test_the_index_lists_every_published_model_in_publication_order(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    base = [write_run(runs, "alpha", 10, (3, 1, 12))]
    trained = [write_run(runs / "t", "alpha", 10, (6, 0, 10), arm="checkpoint-2")]
    publish(runs, out, base, name="z-base", when="2026-10-22T00:00:00+00:00")
    publish(
        runs,
        out,
        trained,
        name="a-checkpoint-2",
        arm="checkpoint-2",
        when="2026-10-23T00:00:00+00:00",
    )
    index = (out / "README.md").read_text()
    assert index.index("| z-base |") < index.index("| a-checkpoint-2 |")


def test_a_draw_that_never_closed_its_reasoning_counts_every_token_as_reasoning(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    root = write_run(runs, "alpha", 10, (3, 1, 12))
    samples = [{"completion_ids": [1, 2, 3], "finish_reason": "length"} for _ in range(4)]
    (root / "base/samples.json").write_text(json.dumps(samples))
    publish(runs, out, [root])
    alpha = json.loads((out / "qwen-base-xhigh/results.json").read_text())["projects"][0]
    assert (alpha["mean_reasoning_tokens"], alpha["mean_answer_tokens"]) == (3, 0)
    assert alpha["draws"][0]["closed_reasoning"] is False


def test_an_ungraded_draw_is_never_published_as_a_score(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    root = write_run(runs, "alpha", 10, (3, 1, 12), status="truncated")
    with pytest.raises(ValueError, match="alpha"):
        publish(runs, out, [root])
    assert not out.exists()


@pytest.mark.parametrize("change", [{"effort": "medium"}, {"arm": "checkpoint-2"}])
def test_projects_from_different_settings_or_arms_are_not_pooled(tmp_path, change):
    runs, out = tmp_path / "runs", tmp_path / "results"
    roots = [
        write_run(runs, "alpha", 10, (3, 1, 12)),
        write_run(runs, "beta", 20, (5, 2, 20), **change),
    ]
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        publish(runs, out, roots)
    assert not out.exists()


def test_a_published_model_is_never_overwritten(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    roots = [write_run(runs, "alpha", 10, (3, 1, 12))]
    publish(runs, out, roots)
    with pytest.raises(FileExistsError):
        publish(runs, out, roots)


def test_the_reasoning_end_token_comes_from_the_pinned_tokenizer(tmp_path):
    added = {"7": {"content": "<think>"}, "8": {"content": "</think>"}}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}))
    assert published_results.think_end_id(tmp_path) == 8
    del added["8"]
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}))
    with pytest.raises(ValueError, match="</think>"):
        published_results.think_end_id(tmp_path)


def test_projects_differ_only_in_their_output_allowance(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    roots = [write_run(runs, "alpha", 10, (3, 1, 12)), write_run(runs, "beta", 20, (5, 2, 20))]
    publish(runs, out, roots)
    prepared = json.loads((roots[1] / "prepared.json").read_text())
    prepared["sampling"]["temperature"] = 0.7
    (roots[1] / "prepared.json").write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="sampling"):
        publish(runs, out, roots, name="other")
