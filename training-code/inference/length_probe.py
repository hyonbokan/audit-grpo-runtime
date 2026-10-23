"""Medium-effort length probe: how long the base model's free-text audits run on training rows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import benchmark_free_text as free_text
from frozen_requests import rerender_request
from grpo.placement import rollout_gpus
from progress import verify_dataset
from published_results import split_tokens, think_end_id
from rollout import load_engine
from runtime_io import read_json, save_json, sha256_file

EFFORT = "medium"
# Totals (prompt plus completion) the update could be capped at, as multiples of 16k tokens.
CANDIDATE_TOTALS = (65536, 81920, 98304)
CONFIG_KEYS = {
    "corpus",
    "corpus_manifest_sha256",
    "model_path",
    "row_count",
    "cap_total_tokens",
    "seconds_limit",
    "tensor_parallel_size",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    PHASES[args.phase](read_json(args.config), args.out)
    return 0


def phase_settings(config: dict[str, Any], _out: Path) -> None:
    validate_config(config)
    scope = read_json(free_text.SCOPE_PATH)
    print(scope["native_image"], scope["rollout_image"], config["seconds_limit"], sep="\n")


def phase_prepare(config: dict[str, Any], out: Path) -> None:
    # The tokenizer comes from the GPU images; CPU tooling imports this module without it.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    prepare(config, out, tokenizer)


def phase_run(config: dict[str, Any], out: Path) -> None:
    run_probe(config, out)


def phase_report(config: dict[str, Any], out: Path) -> None:
    write_report(config, out)


PHASES = {
    "settings": phase_settings,
    "prepare": phase_prepare,
    "run": phase_run,
    "report": phase_report,
}


def validate_config(config: dict[str, Any]) -> None:
    if set(config) != CONFIG_KEYS:
        raise ValueError(f"Probe config keys must be {sorted(CONFIG_KEYS)}")
    for key in ("row_count", "cap_total_tokens", "seconds_limit"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    rollout_gpus({"tensor_parallel_size": config["tensor_parallel_size"]})


# --- Prepare: every row's medium prompt length, and the rows to sample


def prepare(config: dict[str, Any], out: Path, tokenizer: Any) -> None:
    """Re-tokenize every training row at medium effort and choose the rows to generate."""
    validate_config(config)
    corpus = Path(config["corpus"])
    verify_dataset(
        {"dataset_host": str(corpus), "dataset_manifest_sha256": config["corpus_manifest_sha256"]}
    )
    scope = read_json(free_text.SCOPE_PATH)
    manifest = read_json(corpus / "manifest.json")
    train = manifest["roles"]["train"]
    messages = {
        row["row_id"]: row["messages"]
        for row in map(json.loads, (corpus / "data/policy/train.jsonl").read_text().splitlines())
    }
    requests, prompt_tokens = {}, {}
    for row_id in train:
        original = corpus / "requests" / Path(manifest["requests"][row_id]["container_path"]).name
        requests[row_id] = rerender_request(
            read_json(original),
            messages[row_id],
            tokenizer,
            EFFORT,
            sha256_file(original),
            maximum_context=scope["engine"]["max_model_len"],
        )
        prompt_tokens[row_id] = len(requests[row_id]["prompt_ids"])
    chosen = select_rows(prompt_tokens, config["row_count"])
    rows = []
    for index, row_id in enumerate(chosen):
        path = out / "requests" / f"{index:02d}.json"
        save_json(path, requests[row_id])
        rows.append(
            {
                "index": index,
                "row_id": row_id,
                "request": str(path),
                "prompt_tokens": prompt_tokens[row_id],
                "max_tokens": config["cap_total_tokens"] - prompt_tokens[row_id],
            }
        )
    if any(row["max_tokens"] < 1 for row in rows):
        raise ValueError("A chosen row's prompt leaves no room under the total cap")
    save_json(out / "prompts.json", prompt_tokens)
    save_json(out / "manifest.json", {"rows": rows})


def select_rows(prompt_tokens: dict[str, int], count: int) -> list[str]:
    """Rows spread evenly over the prompt-length order, so short and long prompts are both seen."""
    ordered = sorted(prompt_tokens, key=lambda row_id: (prompt_tokens[row_id], row_id))
    if count > len(ordered):
        raise ValueError(f"{count} rows asked for; the corpus has {len(ordered)}")
    return [ordered[int((position + 0.5) * len(ordered) / count)] for position in range(count)]


# --- Run: one engine, one row at a time, each row saved as it finishes


def run_probe(config: dict[str, Any], out: Path) -> None:
    scope = read_json(free_text.SCOPE_PATH)
    manifest = read_json(out / "manifest.json")
    engine_options = dict(scope["engine"], tensor_parallel_size=config["tensor_parallel_size"])
    arm = {
        "config": {"engine": engine_options, "model_path": config["model_path"]},
        "checkpoint": None,
        "step": 0,
        "digest": None,
        "merged": False,
        "runtime_adapter": False,
    }
    (out / "rows").mkdir(exist_ok=True)
    pending = [
        row for row in manifest["rows"] if not (out / "rows" / f"{row['index']:02d}.json").exists()
    ]
    if not pending:
        return
    started = time.monotonic()
    engine, _, sampling_params = load_engine(arm, out, {})
    think_end = think_end_id(Path(config["model_path"]))
    for row in pending:
        params = sampling_params(
            **{k: v for k, v in scope["sampling"].items() if k != "logprobs"},
            max_tokens=row["max_tokens"],
        )
        began = time.monotonic()
        output = engine.generate(
            [{"prompt_token_ids": read_json(row["request"])["prompt_ids"]}],
            params,
            use_tqdm=False,
        )[0]
        draws = [draw_record(draw, think_end) for draw in output.outputs]
        save_json(
            out / "rows" / f"{row['index']:02d}.json",
            {**row, "seconds": time.monotonic() - began, "draws": draws},
        )
        print(
            f"row {row['index']} {row['row_id']}: {[d['completion_tokens'] for d in draws]}",
            flush=True,
        )
    save_json(
        out / "run-receipt.json", {"seconds": time.monotonic() - started, "rows": len(pending)}
    )


def draw_record(draw: Any, think_end: int) -> dict[str, Any]:
    ids = list(draw.token_ids)
    return {
        "completion_tokens": len(ids),
        "finish_reason": draw.finish_reason,
        **split_tokens(ids, think_end),
        "text": draw.text,
    }


# --- Report


def write_report(config: dict[str, Any], out: Path) -> None:
    rows = [read_json(path) for path in sorted((out / "rows").glob("*.json"))]
    result = summarize_probe(rows, read_json(out / "prompts.json"), config["cap_total_tokens"])
    save_json(out / "report.json", result)
    (out / "REPORT.md").write_text(render_report(result))


def summarize_probe(
    rows: list[dict[str, Any]], prompt_tokens: dict[str, int], cap_total: int
) -> dict[str, Any]:
    """Completion lengths, how many draws a total cap would cut, and the rows each cap keeps."""
    if not rows:
        raise ValueError("No probed rows to summarize")
    draws = [(row["prompt_tokens"], draw) for row in rows for draw in row["draws"]]
    lengths = sorted(draw["completion_tokens"] for _, draw in draws)
    p95 = percentile(lengths, 0.95)
    cut_at_cap = sum(draw["finish_reason"] == "length" for _, draw in draws)
    totals = {}
    for total in CANDIDATE_TOTALS:
        if total > cap_total:
            continue
        # A draw cut at the probe's cap is at least that long, so it exceeds every smaller total.
        exceeding = sum(prompt + draw["completion_tokens"] >= total for prompt, draw in draws)
        limit = total - p95
        totals[str(total)] = {
            "draws_at_or_over_total": exceeding,
            "fraction_at_or_over_total": exceeding / len(draws),
            "prompt_limit_for_p95_completion": limit,
            "rows_kept": sum(tokens <= limit for tokens in prompt_tokens.values()),
        }
    return {
        "cap_total_tokens": cap_total,
        "probed_rows": len(rows),
        "draws": len(draws),
        "cut_at_probe_cap": cut_at_cap,
        "completion_tokens": {
            "p50": percentile(lengths, 0.5),
            "p90": percentile(lengths, 0.9),
            "p95": p95,
            "max": lengths[-1],
            "mean": sum(lengths) / len(lengths),
            "censored_at_cap": cut_at_cap > 0,
        },
        "mean_reasoning_tokens": mean(draw["reasoning_tokens"] for _, draw in draws),
        "train_rows": len(prompt_tokens),
        "candidate_totals": totals,
        "rows": [
            {
                "row_id": row["row_id"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": [d["completion_tokens"] for d in row["draws"]],
                "finish_reasons": [d["finish_reason"] for d in row["draws"]],
                "seconds": row["seconds"],
            }
            for row in rows
        ],
    }


def percentile(sorted_values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of an ascending list."""
    rank = max(1, -(-int(fraction * 1000) * len(sorted_values) // 1000))
    return sorted_values[rank - 1]


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values)


def render_report(result: dict[str, Any]) -> str:
    tokens = result["completion_tokens"]
    lines = [
        "# Medium-effort length probe (base model, free-text prompt)",
        "",
        f"{result['probed_rows']} rows x 4 draws, generation cap {result['cap_total_tokens']} "
        f"tokens including the prompt; {result['cut_at_probe_cap']} of {result['draws']} draws "
        "hit it.",
        "",
        f"Completion tokens: p50 {tokens['p50']}, p90 {tokens['p90']}, p95 {tokens['p95']}, "
        f"max {tokens['max']}, mean {tokens['mean']:.0f}"
        + (" (draws cut at the cap are counted at the cap)." if tokens["censored_at_cap"] else "."),
        "",
        "| Total cap | Draws at or over it | Prompt limit for a p95 completion | "
        f"Train rows kept of {result['train_rows']} |",
        "|---:|---:|---:|---:|",
    ]
    for total, row in result["candidate_totals"].items():
        lines.append(
            f"| {total} | {row['draws_at_or_over_total']} of {result['draws']} "
            f"({row['fraction_at_or_over_total']:.0%}) | {row['prompt_limit_for_p95_completion']} "
            f"| {row['rows_kept']} |"
        )
    lines += ["", "| Row | Prompt | Completion tokens (4 draws) | Finish |", "|---|---:|---|---|"]
    for row in result["rows"]:
        lines.append(
            f"| {row['row_id']} | {row['prompt_tokens']} | {row['completion_tokens']} "
            f"| {row['finish_reasons']} |"
        )
    return "\n".join([*lines, ""])


if __name__ == "__main__":
    raise SystemExit(main())
