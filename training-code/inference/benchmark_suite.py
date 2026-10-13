"""Bound a matched Solidity suite and report complete or explicitly partial panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_matching import native_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("next", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.phase == "next":
        print(next_project(config))
    else:
        records = []
        for project in config["projects"]:
            path = Path(
                project.get("reuse_comparison") or Path(project["run_root"]) / "comparison.json"
            )
            comparison = None
            if path.is_file():
                data = path.read_bytes()
                if (
                    project.get("comparison_sha256")
                    and hashlib.sha256(data).hexdigest() != project["comparison_sha256"]
                ):
                    raise ValueError("Saved comparison identity differs")
                comparison = json.loads(data)
            execution = {}
            for label in ("base", config.get("trained_label", "checkpoint-2")):
                receipt = Path(project["run_root"]) / label / "receipt.json"
                if receipt.is_file():
                    saved = json.loads(receipt.read_text())
                    execution[label] = {
                        key: saved.get(key)
                        for key in (
                            "generation_seconds",
                            "total_seconds",
                            "completion_tokens",
                            "tokens_per_second",
                        )
                    }
            records.append({**project, "comparison": comparison, "execution": execution})
        result = summarize_suite(
            records,
            sample_count=config.get("sampling", {}).get("n", 4),
            output_format=config.get("output_format", "json"),
            checkpoint_label=config.get("trained_label", "checkpoint-2"),
        )
        generation_seconds = sum(
            arm.get("generation_seconds") or 0
            for project in records
            if not project.get("reuse_comparison")
            for arm in project["execution"].values()
        )
        ledger = (
            json.loads(Path(config["judge_ledger"]).read_text())
            if config.get("judge_ledger")
            else {}
        )
        operations = {p.get("run_id") for p in records if not p.get("reuse_comparison")}
        calls = [call for call in ledger.get("calls", []) if call.get("operation_id") in operations]
        result["cost"] = {
            "judge_known_usage_estimate_usd": sum(call.get("estimated_usd") or 0 for call in calls),
            "judge_conservative_charge_usd": sum(call["budget_charge_usd"] for call in calls),
            "unsettled_judge_reservations": sum(call["status"] == "reserved" for call in calls),
            "shared_ledger_charge_usd": ledger.get("budget_charge_usd"),
            "generation_compute_estimate_usd": generation_seconds
            / 3600
            * config.get("rate_usd_per_hour", 0),
            "scope": (
                "New suite generation and judging; excludes reused results, model startup, "
                "idle and other VM time."
            ),
        }
        result.update(
            run_id=config["run_id"],
            generated_at=datetime.now(UTC).isoformat(),
            configuration=config,
            reasoning_effort=config.get("template_options", {}).get("reasoning_effort", "low"),
        )
        args.out.mkdir(parents=True, exist_ok=True)
        temporary = args.out / "report.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.out / "report.json")
        (args.out / "REPORT.md").write_text(render_report(result))
    return 0


def next_project(config: dict[str, Any], *, now: datetime | None = None) -> str:
    if config.get("stop_on_project_failure"):
        for project in config["projects"]:
            exit_path = Path(project["run_root"]) / "launcher-exit.txt"
            if exit_path.is_file() and exit_path.read_text().strip() != "0":
                return "complete"
    now = now or datetime.now(UTC)
    remaining = (
        int((datetime.fromisoformat(config["absolute_deadline"]) - now).total_seconds()) - 60
    )
    if remaining < 600:
        return "complete"
    for project in config["projects"]:
        if project.get("reuse_comparison"):
            continue
        root = Path(project["run_root"])
        if (root / "launcher-exit.txt").exists() or (root / "comparison.json").exists():
            continue
        if (root / "prepared.json").exists():
            raise ValueError("An unfinished prepared run must not be restarted")
        path = root / "config.json"
        child = json.loads(path.read_text())
        child["seconds_limit"] = min(child["seconds_limit"], remaining)
        path.write_text(json.dumps(child, indent=2) + "\n")
        return str(Path(config["runtime_host"]) / "results" / root.name)
    return "complete"


def summarize_suite(
    projects: list[dict[str, Any]],
    *,
    sample_count: int = 4,
    output_format: str = "json",
    checkpoint_label: str = "checkpoint-2",
) -> dict[str, Any]:
    if sample_count not in (1, 4):
        raise ValueError("Unsupported suite sample count")
    summaries, completed = [], []
    for project in projects:
        comparison = project["comparison"]
        summary = {
            "name": project["name"],
            "known_gt_count": project["known_gt_count"],
            "status": "pending",
            "arms": {},
            "execution": project.get("execution", {}),
            "run_root": project.get("run_root"),
        }
        if comparison is not None:
            if comparison.get("output_format", "json") != output_format:
                raise ValueError("Comparison output format differs from the suite")
            arms = comparison["arms"]
            arms = list(arms.values()) if isinstance(arms, dict) else arms
            if isinstance(comparison["arms"], dict):
                arms = [dict(arm, label=label) for label, arm in comparison["arms"].items()]
            if {arm["label"] for arm in arms} != {"base", checkpoint_label}:
                raise ValueError("Matched base and checkpoint2 arms required")
            for arm in arms:
                draws = arm["draws"]
                if len(draws) != sample_count or any(
                    draw["known_gt_count"] != project["known_gt_count"] for draw in draws
                ):
                    raise ValueError("Declared draws with the fixed GT denominator required")
                known = all(draw["status"] == "graded" for draw in draws)
                summary["arms"][arm["label"]] = {
                    "status": "graded" if known else "incomplete",
                    "draw_statuses": [draw["status"] for draw in draws],
                    "strict_json_validity": comparison.get("strict_json_validity", {}).get(
                        arm["label"]
                    ),
                    **(arm_summary(draws) if known else {}),
                }
            summary["status"] = (
                "complete"
                if all(a["status"] == "graded" for a in summary["arms"].values())
                else "incomplete"
            )
            if summary["status"] == "complete":
                completed.append(summary)
        summaries.append(summary)
    aggregates = aggregate_projects(completed, checkpoint_label=checkpoint_label)
    complete = len(completed) == len(projects) and bool(projects)
    comparison = None
    if complete:
        base, trained = aggregates["base"], aggregates[checkpoint_label]
        comparison = {
            "macro_deltas": {
                key: trained["macro"][key] - base["macro"][key] for key in base["macro"]
            },
            "micro_deltas": {
                key: trained["micro"][key] - base["micro"][key] for key in base["micro"]
            },
            "positive_detection_signal": trained["macro"]["recall"] > base["macro"]["recall"]
            and trained["macro"]["recall_with_partial"] >= base["macro"]["recall_with_partial"],
        }
    return {
        "status": "matched_suite_completed" if complete else "partial_suite",
        "projects": summaries,
        "aggregates": aggregates if complete else None,
        "comparison": comparison,
        "completed_subset": {
            "project_names": [p["name"] for p in completed],
            "aggregates": aggregates,
        },
        "quality_gain_established": False,
        "sample_count": sample_count,
        "output_format": output_format,
        "checkpoint_label": checkpoint_label,
        "limitations": [
            "Established regression benchmarks have historical inspection exposure.",
            "Unmatched predictions are not confirmed false positives.",
            "Detection scores alone do not establish general reasoning or retention gains.",
        ],
    }


def arm_summary(draws: list[dict[str, Any]]) -> dict[str, Any]:
    known = sum(draw["known_gt_count"] for draw in draws)
    exact = sum(draw["exact_matches"] for draw in draws)
    partial = sum(draw["partial_matches"] for draw in draws)
    findings = sum(draw["findings_count"] for draw in draws)
    values = [
        native_metrics(
            d["known_gt_count"], d["findings_count"], d["exact_matches"], d["partial_matches"]
        )
        for d in draws
    ]
    severities = {}
    unique_exact, unique_partial = set(), set()
    for draw in draws:
        for verdict in draw.get("verdicts", []):
            row = severities.setdefault(
                verdict["severity_from_truth"],
                {
                    "gt_opportunities": 0,
                    "exact_matches": 0,
                    "partial_matches": 0,
                    "matched_severity_correct": 0,
                },
            )
            row["gt_opportunities"] += 1
            is_exact = bool(verdict["is_match"])
            is_partial = bool(verdict["is_partial_match"] and not is_exact)
            row["exact_matches"] += is_exact
            row["partial_matches"] += is_partial
            row["matched_severity_correct"] += (is_exact or is_partial) and verdict[
                "severity_from_junior_auditor"
            ] == verdict["severity_from_truth"]
            if is_exact:
                unique_exact.add(verdict["target_index"])
            elif is_partial:
                unique_partial.add(verdict["target_index"])
    return {
        "exact_matches": exact,
        "partial_matches": partial,
        "findings_total": findings,
        "mean_findings": findings / len(draws),
        "known_gt_opportunities": known,
        "unmatched_findings": sum(d["unmatched_findings"] for d in draws),
        "micro": native_metrics(known, findings, exact, partial),
        "mean_draw_metrics": {key: mean(v[key] for v in values) for key in values[0]},
        "unique_exact_gt": len(unique_exact),
        "unique_exact_or_partial_gt": len(unique_exact | unique_partial),
        "severity": severities,
    }


def aggregate_projects(
    projects: list[dict[str, Any]], *, checkpoint_label: str = "checkpoint-2"
) -> dict[str, Any] | None:
    if not projects:
        return None
    result = {}
    for label in ("base", checkpoint_label):
        arms = [project["arms"][label] for project in projects]
        totals = {
            key: sum(arm[key] for arm in arms)
            for key in (
                "exact_matches",
                "partial_matches",
                "findings_total",
                "known_gt_opportunities",
                "unmatched_findings",
                "unique_exact_gt",
                "unique_exact_or_partial_gt",
            )
        }
        result[label] = {
            **totals,
            "macro": {
                key: mean(arm["mean_draw_metrics"][key] for arm in arms)
                for key in arms[0]["mean_draw_metrics"]
            },
            "micro": native_metrics(
                totals["known_gt_opportunities"],
                totals["findings_total"],
                totals["exact_matches"],
                totals["partial_matches"],
            ),
        }
    return result


def render_report(result: dict[str, Any]) -> str:
    trained = result.get("checkpoint_label", "checkpoint-2")
    effort = result.get("reasoning_effort", "low")
    lines = [
        f"# Solidity 1: BF16 base versus GRPO {trained}",
        "",
        f"Status: **{result['status']}**. {result['sample_count']} {effort} audit(s) per "
        f"project and model; {result['output_format']} output.",
        "",
        f"| Project | GT | Base exact recall | {trained} exact recall "
        f"| Base with partial | {trained} with partial |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for project in result["projects"]:
        values = []
        for metric in ("recall", "recall_with_partial"):
            for label in ("base", trained):
                arm = project["arms"].get(label, {})
                value = arm.get("mean_draw_metrics", {}).get(metric)
                values.append(f"{value:.2%}" if value is not None else "pending/unknown")
        lines.append(f"| {project['name']} | {project['known_gt_count']} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "Full precision/recall/F1, partial metrics, severity, strict validity and counts "
            "are in `report.json`.",
            "Missing/unknown projects do not receive zero scores; full-panel aggregates are "
            "withheld until complete.",
            "These observations do not establish general reasoning-quality or "
            "capability-retention gains.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
