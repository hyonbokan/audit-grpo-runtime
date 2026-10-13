"""Freeze reviewed private rows, policy-only prompts and full-context tokenizer requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from corpus import check_isolation
from experiment import save


def verify_targets(row, archive):
    for t in row["scorer_metadata"]["targets"]:
        record = t["finding"]["archive_record"]
        if t["id"] != f"archive:{record['id']}" or archive.get(record["id"]) != record:
            raise ValueError("Private target does not equal its source archive record")


def export(config, out):
    from scripts.finetuning.tools.training_rows import render_training_row

    staging = json.loads(Path(config["staging"]).read_text())
    if isinstance(staging, dict):
        staging = staging.get("records", staging.get("vulnerabilities"))
    archive = {r["id"]: r for r in staging}
    index = {
        r["archive_id"]: r
        for r in map(json.loads, Path(config["archive_index"]).read_text().splitlines())
    }
    roles = {s: [] for s in ("train", "calibration", "evaluation", "regression")}
    identities = []
    for split in ("train", "calibration", "evaluation"):
        for row in map(
            json.loads,
            (Path(config["original"]) / f"data/private/{split}.jsonl").read_text().splitlines(),
        ):
            role = "regression" if row["row_id"] in config["regression_row_ids"] else split
            roles[role].append(row)
    for folder in map(Path, config["reviews"]):
        for path in sorted(folder.glob("*/accepted-row.private.json")):
            row = json.loads(path.read_text())
            request = json.loads((path.parent / "request.json").read_text())
            role = request["payload"]["split"]
            roles[role].append(row)
    for split, rows in roles.items():
        for row in rows:
            verify_targets(row, archive)
            targets = [
                {"archive_id": t["finding"]["archive_record"]["id"]}
                for t in row["scorer_metadata"]["targets"]
            ]
            families = sorted({index[t["archive_id"]]["family_id"] for t in targets})
            identities.append({**row, "split": split, "family_ids": families, "targets": targets})
    check_isolation(identities, vendor_fingerprints=set(config.get("vendor_fingerprints", [])))
    for split, rows in roles.items():
        private = out / f"data/private/{split}.jsonl"
        policy = out / f"data/policy/{split}.jsonl"
        private.parent.mkdir(parents=True, exist_ok=True)
        policy.parent.mkdir(parents=True, exist_ok=True)
        rendered = []
        for row in rows:
            value = render_training_row(row, include_context=False)
            if not value["scorer_metadata"]["eligible_targets"]:
                raise ValueError("Empty code-only target set")
            rendered.append({"row_id": row["row_id"], "messages": value["messages"]})
        private.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        policy.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rendered))
    save(
        out / "semantic-roles.private.json", {k: [r["row_id"] for r in v] for k, v in roles.items()}
    )
    print(
        json.dumps(
            {
                k: {
                    "rows": len(v),
                    "targets": sum(len(r["scorer_metadata"]["targets"]) for r in v),
                    "code_only_targets": sum(
                        not t["required_context"]
                        for r in v
                        for t in r["scorer_metadata"]["targets"]
                    ),
                }
                for k, v in roles.items()
            }
        )
    )


def requests(config, out):
    from transformers import AutoTokenizer
    from grpo.policy_prompt import policy_template_options, prompt_token_ids
    from inference.benchmark import validate_request

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    options = policy_template_options(json.loads(Path(config["source_plan"]).read_text()))
    accepted, rejected, info = {}, [], {}
    for split in ("train", "calibration", "evaluation", "regression"):
        pp = out / f"data/policy/{split}.jsonl"
        vp = out / f"data/private/{split}.jsonl"
        policy = list(map(json.loads, pp.read_text().splitlines()))
        private = list(map(json.loads, vp.read_text().splitlines()))
        selected = set()
        for row in policy:
            ids = prompt_token_ids(tokenizer, row["messages"], template_options=options)
            context = max(36864, ((len(ids) + 16384 + 4095) // 4096) * 4096)
            if context > 98304:
                rejected.append(
                    {
                        "row_id": row["row_id"],
                        "split": split,
                        "prompt_tokens": len(ids),
                        "reason": "complete prompt plus fixed 16384 output exceeds reviewed 98304 context",
                    }
                )
                continue
            request = {
                "row_id": row["row_id"],
                "model": {
                    "id": config["model_id"],
                    "revision": config["model_revision"],
                    "precision": "bf16",
                },
                "prompt_ids": ids,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "full_completion_ceiling": 16384,
                "max_model_len": context,
                "template_options": options,
                "chat_template_sha256": hashlib.sha256(
                    tokenizer.chat_template.encode()
                ).hexdigest(),
            }
            validate_request(request)
            filename = hashlib.sha256(row["row_id"].encode()).hexdigest()[:20] + ".json"
            save(out / f"requests/{filename}", request)
            info[row["row_id"]] = {
                "context": context,
                "prompt_tokens": len(ids),
                "container_path": config["dataset_container"] + f"/requests/{filename}",
                "request_sha256": hashlib.sha256(
                    (out / f"requests/{filename}").read_bytes()
                ).hexdigest(),
            }
            selected.add(row["row_id"])
        # Preserve full semantic export separately before applying the honest current-engine fit gate.
        for p in (pp, vp):
            snapshot = out / "semantic-export" / p.relative_to(out)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(p.read_bytes())
        pp.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in policy if r["row_id"] in selected
            )
        )
        vp.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in private if r["row_id"] in selected
            )
        )
        accepted[split] = [r["row_id"] for r in private if r["row_id"] in selected]
    save(out / "context-withheld.json", rejected)
    counts = {}
    for split in accepted:
        rows = list(map(json.loads, (out / f"data/private/{split}.jsonl").read_text().splitlines()))
        counts[split] = {
            "rows": len(rows),
            "targets": sum(len(r["scorer_metadata"]["targets"]) for r in rows),
            "code_only_targets": sum(
                not t["required_context"] for r in rows for t in r["scorer_metadata"]["targets"]
            ),
        }
    files = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    save(
        out / "manifest.json",
        {
            "version": "full-training-20261009-v1",
            "roles": accepted,
            "requests": info,
            "counts": counts,
            "files": files,
            "source_archive_records": 7195,
            "label_coverage": "sparse reviewed known findings; not exhaustive report GT",
            "prior_optimizer_visits": {"2022-12-caviar/all-scope": 2},
            "context_withheld": rejected,
        },
    )
    print(
        json.dumps(
            {
                "counts": counts,
                "context_withheld": rejected,
                "prompt_tokens": {k: v["prompt_tokens"] for k, v in info.items()},
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("export", "requests"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (export if args.phase == "export" else requests)(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
