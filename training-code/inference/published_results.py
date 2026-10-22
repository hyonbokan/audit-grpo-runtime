"""Publish one model's finished free-text runs as results/<name>/results.json and the index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_matching import native_metrics
from benchmark_suite import arm_summary
from grpo.samples import write_new
from runtime_io import read_json, sha256_file, utc_now

# Every project of one published model must share these prepared settings.
SHARED_KEYS = ("model_id", "model_revision", "base_precision", "benchmark_suite")
INDEX_COLUMNS = ("exact", "partial", "findings", "precision", "recall", "f1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--name", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    PHASES[args.phase](args)
    return 0


def phase_publish(args: argparse.Namespace) -> None:
    model_path = Path(read_json(args.runs[0] / "prepared.json")["model_path"])
    think_end = think_end_id(model_path)
    publish(args.name, args.arm, args.runs, args.out, think_end, utc_now().isoformat())


PHASES = {"publish": phase_publish}


# --- Publish


def publish(
    name: str, arm: str, roots: list[Path], out: Path, think_end: int, published_utc: str
) -> dict[str, Any]:
    """Write one model's results.json, then rebuild the index; a published model is final."""
    result = model_results(name, arm, roots, think_end, published_utc)
    folder = out / name
    if folder.exists():
        raise FileExistsError(f"{folder} is already published")
    folder.mkdir(parents=True)
    write_new(folder / "results.json", result)
    (out / "README.md").write_text(render_index(published(out)))
    return result


def model_results(
    name: str, arm: str, roots: list[Path], think_end: int, published_utc: str
) -> dict[str, Any]:
    prepared = [read_json(root / "prepared.json") for root in roots]
    receipts = [read_json(root / arm / "receipt.json") for root in roots]
    for key, values in (
        *((key, [p[key] for p in prepared]) for key in SHARED_KEYS),
        ("reasoning_effort", [p["template_options"]["reasoning_effort"] for p in prepared]),
        # Each project's output allowance is the context left after its own prompt.
        (
            "sampling",
            [{k: v for k, v in p["sampling"].items() if k != "max_tokens"} for p in prepared],
        ),
        ("weight_version", [r["weight_version"] for r in receipts]),
    ):
        if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
            raise ValueError(f"Projects of one published model differ in {key}")
    projects = [
        project_results(root, saved, arm, think_end)
        for root, saved in zip(roots, prepared, strict=True)
    ]
    total = {
        key: sum(p[key] for p in projects) for key in ("exact", "partial", "findings", "known")
    }
    return {
        "name": name,
        "arm": arm,
        "model_id": prepared[0]["model_id"],
        "model_revision": prepared[0]["model_revision"],
        "base_precision": prepared[0]["base_precision"],
        "weight_version": receipts[0]["weight_version"],
        "weight_representation": receipts[0]["weight_representation"],
        "reasoning_effort": prepared[0]["template_options"]["reasoning_effort"],
        "draws_per_project": prepared[0]["sampling"]["n"],
        "benchmark_suite": prepared[0]["benchmark_suite"],
        "published_utc": published_utc,
        "definitions": {
            "exact": "known vulnerabilities matched exactly in at least one draw",
            "partial": "known vulnerabilities matched partially, and never exactly, in any draw",
            "findings": "every finding reported, summed over all draws",
            "precision": "exact / findings",
            "recall": "exact / known vulnerabilities",
            "f1": "harmonic mean of precision and recall",
        },
        "total": {
            **total,
            **metrics(total),
            "mean_reasoning_tokens": mean(p["mean_reasoning_tokens"] for p in projects),
            "mean_answer_tokens": mean(p["mean_answer_tokens"] for p in projects),
        },
        "projects": projects,
    }


def project_results(root: Path, prepared: dict, arm: str, think_end: int) -> dict[str, Any]:
    name = prepared["row_id"].split("/", 1)[1]
    draws = read_json(root / "comparison.json")["draws"][arm]
    statuses = [draw["status"] for draw in draws]
    if any(status != "graded" for status in statuses):
        raise ValueError(f"{name}: every draw must be graded to publish; got {statuses}")
    samples = read_json(root / arm / "samples.json")
    if len(samples) != len(draws):
        raise ValueError(f"{name}: {len(samples)} samples for {len(draws)} graded draws")
    summary = arm_summary(draws)
    counts = {
        "exact": summary["unique_exact_gt"],
        "partial": summary["unique_exact_or_partial_gt"] - summary["unique_exact_gt"],
        "findings": summary["findings_total"],
        "known": draws[0]["known_gt_count"],
    }
    tokens = [split_tokens(sample["completion_ids"], think_end) for sample in samples]
    return {
        "name": name,
        "row_id": prepared["row_id"],
        "run_root": str(root),
        "comparison_sha256": sha256_file(root / "comparison.json"),
        **counts,
        **metrics(counts),
        "mean_reasoning_tokens": mean(t["reasoning_tokens"] for t in tokens),
        "mean_answer_tokens": mean(t["answer_tokens"] for t in tokens),
        "draws": [
            {
                "exact": draw["exact_matches"],
                "partial": draw["partial_matches"],
                "findings": draw["findings_count"],
                "finish_reason": sample["finish_reason"],
                **split,
            }
            for draw, sample, split in zip(draws, samples, tokens, strict=True)
        ],
    }


def think_end_id(model_path: Path) -> int:
    """The token id that closes reasoning, from the pinned tokenizer's added tokens."""
    added = read_json(model_path / "tokenizer_config.json")["added_tokens_decoder"]
    ids = [int(key) for key, token in added.items() if token["content"] == "</think>"]
    if len(ids) != 1:
        raise ValueError(f"Expected one </think> token in {model_path}; found {ids}")
    return ids[0]


def split_tokens(ids: list[int], think_end: int) -> dict[str, Any]:
    """Reasoning runs up to `</think>`; a draw that never closed it is all reasoning."""
    if think_end not in ids:
        return {"reasoning_tokens": len(ids), "answer_tokens": 0, "closed_reasoning": False}
    end = ids.index(think_end)
    return {
        "reasoning_tokens": end,
        "answer_tokens": len(ids) - end - 1,
        "closed_reasoning": True,
    }


def metrics(counts: dict[str, float]) -> dict[str, float]:
    values = native_metrics(counts["known"], counts["findings"], counts["exact"], counts["partial"])
    return {key: values[key] for key in ("precision", "recall", "f1")}


# --- Index


def published(out: Path) -> list[dict[str, Any]]:
    entries = [read_json(path) for path in out.glob("*/results.json")]
    return sorted(entries, key=lambda entry: entry["published_utc"])


def render_index(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Results",
        "",
        "Free-text Solidity benchmark: each model audits every project's sources with no output",
        "format; a judge extracts the findings and matches them one-to-one against the known",
        "vulnerabilities. Generated by `training-code/publish-results.sh`; do not edit by hand.",
        "",
        "Exact and Partial count known vulnerabilities once, however many of a model's draws",
        "found them; Partial means matched partially and never exactly. Total is every finding",
        "all draws reported. Precision = exact / total; recall = exact / known vulnerabilities;",
        "F1 is their harmonic mean. Partial matches are not counted in precision, recall or F1.",
        "",
        "| Model | Effort | Exact | Partial | Total | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(row(entry["name"], entry["reasoning_effort"], entry["total"]))
    for entry in entries:
        lines += [
            "",
            f"## {entry['name']}",
            "",
            f"{entry['model_id']} `{entry['weight_version']}`, {entry['base_precision']}, "
            f"{entry['reasoning_effort']} reasoning, {entry['draws_per_project']} draws per "
            f"project, {entry['total']['known']:g} known vulnerabilities. Mean tokens per draw: "
            f"{entry['total']['mean_reasoning_tokens']:,.0f} reasoning, "
            f"{entry['total']['mean_answer_tokens']:,.0f} answer. "
            f"Details: [`{entry['name']}/results.json`]({entry['name']}/results.json).",
            "",
            "| Project | Known | Exact | Partial | Total | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for project in entry["projects"]:
            lines.append(row(project["name"], f"{project['known']:g}", project))
        lines.append(row("**All**", f"{entry['total']['known']:g}", entry["total"]))
    return "\n".join(lines) + "\n"


def row(label: str, second: str, values: dict[str, float]) -> str:
    counts = [f"{values[key]:,}" for key in INDEX_COLUMNS[:3]]
    rates = [f"{values[key]:.3f}" for key in INDEX_COLUMNS[3:]]
    return f"| {label} | {second} | " + " | ".join(counts + rates) + " |"


if __name__ == "__main__":
    raise SystemExit(main())
