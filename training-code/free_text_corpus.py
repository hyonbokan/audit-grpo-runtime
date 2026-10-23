"""Derive the free-text training corpus: the evaluation's prompt over the same rows and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_contract import read_template
from benchmark_free_text import audit_prompt, render_sources
from runtime_io import read_json, save_json, sha256_bytes, sha256_file

SPLITS = ("train", "calibration", "evaluation", "regression")
JUDGE_TEMPLATE = "judge-prompt.txt"
IMPACT_SEVERITY = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low", "INFO": "Info"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark-assets", type=Path, required=True)
    args = parser.parse_args()
    # The tokenizer comes from the GPU images; CPU tooling imports this module without it.
    from transformers import AutoTokenizer

    template = read_template(
        args.benchmark_assets / "benchmark/src/data/calculate_prompt.ts", "CALCULATE_PROMPT"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    derive(args.source, args.out, tokenizer, template)
    return 0


def derive(source: Path, out: Path, tokenizer: Any, judge_template: str) -> None:
    """Write prompts, truth, re-tokenized requests, the judge template and a hashed manifest."""
    if out.exists():
        raise FileExistsError(f"Derived corpus output must be fresh: {out}")
    manifest = read_json(source / "manifest.json")
    messages_by_row = {}
    for split in SPLITS:
        rows = read_jsonl(source / f"data/private/{split}.jsonl")
        write_jsonl(out / f"data/private/{split}.jsonl", [with_truth(row) for row in rows])
        policy = [{"row_id": r["row_id"], "messages": policy_messages(r)} for r in rows]
        write_jsonl(out / f"data/policy/{split}.jsonl", policy)
        messages_by_row.update((row["row_id"], row["messages"]) for row in policy)
    (out / JUDGE_TEMPLATE).write_text(judge_template)
    for row_id, info in manifest["requests"].items():
        name = Path(info["container_path"]).name
        request = read_json(source / "requests" / name)
        if request["row_id"] != row_id or row_id not in messages_by_row:
            raise ValueError(f"Request {name} does not belong to a corpus row: {row_id}")
        save_json(out / "requests" / name, retokenize(request, messages_by_row[row_id], tokenizer))
    derived = {
        key: value for key, value in manifest.items() if key not in ("policy_rendering_sha256",)
    }
    derived.update(
        version=f"free-text-{manifest['version']}",
        prompt="benchmark_free_text.audit_prompt over benchmark_free_text.render_sources",
        derived_from_manifest_sha256=sha256_file(source / "manifest.json"),
        files=file_hashes(out),
    )
    save_json(out / "manifest.json", derived)


def policy_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """One user message, exactly the evaluation's prompt: no system prompt, no answer format."""
    return [{"role": "user", "content": audit_prompt(render_sources(row["source_files"]))}]


def truth_findings(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Labels the prompt can support, in the evaluation's truth shape."""
    return [truth_finding(target) for target in shown_targets(row)]


def shown_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    # The prompt carries source files only, so a label that needs other context is unreachable.
    return [t for t in row["scorer_metadata"]["targets"] if not t["required_context"]]


def truth_finding(target: dict[str, Any]) -> dict[str, Any]:
    finding = target["finding"]
    impact = finding["archive_record"]["impact"]
    if impact not in IMPACT_SEVERITY:
        raise ValueError(f"Label impact {impact!r}; known: {sorted(IMPACT_SEVERITY)}")
    return {
        "issue": finding["issue"],
        "severity": IMPACT_SEVERITY[impact],
        "description": finding["description"],
        "contracts": finding["contracts"],
        "recommendation": "",
    }


def with_truth(row: dict[str, Any]) -> dict[str, Any]:
    truth = truth_findings(row)
    if not truth:
        raise ValueError(f"Row {row['row_id']} has no label its prompt can support")
    return {**row, "truth": truth, "truth_ids": [t["id"] for t in shown_targets(row)]}


def retokenize(request: dict[str, Any], messages: list[dict], tokenizer: Any) -> dict[str, Any]:
    if sha256_bytes(tokenizer.chat_template.encode()) != request["chat_template_sha256"]:
        raise ValueError("Pinned chat template differs from the request's")
    ids = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, **request["template_options"]
        )
    )
    return {
        **request,
        "prompt_ids": ids,
        "prompt_ids_sha256": sha256_bytes(json.dumps(ids).encode()),
        "full_completion_ceiling": request["max_model_len"] - len(ids),
    }


# --- Files


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


if __name__ == "__main__":
    raise SystemExit(main())
