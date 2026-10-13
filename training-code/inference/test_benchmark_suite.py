"""Suite scores keep project weighting and missing grades explicit."""

import json
from datetime import UTC, datetime

import pytest

from testing.benchmark import project


def test_one_shot_suite_counts_each_gt_once_and_rejects_old_json_panel():
    from benchmark_suite import summarize_suite

    row = project("tiny", 4, (1, 1, 3), (2, 1, 4))
    row["comparison"]["output_format"] = "markdown"
    for arm in row["comparison"]["arms"]:
        arm["draws"] = arm["draws"][:1]
    result = summarize_suite([row], sample_count=1, output_format="markdown")
    assert result["aggregates"]["checkpoint-2"]["known_gt_opportunities"] == 4
    assert result["aggregates"]["checkpoint-2"]["micro"]["recall"] == 0.5
    assert result["projects"][0]["arms"]["base"]["mean_findings"] == 3
    del row["comparison"]["output_format"]
    with pytest.raises(ValueError):
        summarize_suite([row], sample_count=1, output_format="markdown")


def test_suite_reports_macro_micro_and_partial_metrics_without_reusing_denominators():
    from benchmark_suite import summarize_suite

    result = summarize_suite(
        [
            project("small", 2, (1, 1, 3), (2, 0, 3)),
            project("large", 4, (1, 0, 2), (2, 1, 4)),
        ]
    )
    assert result["status"] == "matched_suite_completed"
    assert result["aggregates"]["base"]["macro"]["recall"] == 0.375
    assert result["aggregates"]["base"]["micro"]["recall"] == pytest.approx(1 / 3)
    trained = result["aggregates"]["checkpoint-2"]
    assert trained["macro"]["recall"] == 0.75
    assert trained["micro"]["recall"] == pytest.approx(2 / 3)
    assert trained["micro"]["recall_with_partial"] == pytest.approx(5 / 6)
    assert trained["micro"]["precision"] == pytest.approx(4 / 7)
    assert trained["exact_matches"] == 16
    assert trained["partial_matches"] == 4
    assert trained["known_gt_opportunities"] == 24
    assert result["quality_gain_established"] is False
    assert result["comparison"]["macro_deltas"]["recall"] == 0.375
    assert result["comparison"]["positive_detection_signal"] is True


def test_unknown_or_missing_project_withholds_full_panel_metrics():
    from benchmark_suite import summarize_suite

    good = project("small", 2, (1, 1, 3), (2, 0, 3))
    unknown = project("large", 4, (1, 0, 2), (2, 1, 4))
    unknown["comparison"]["arms"][1]["draws"][0]["status"] = "unknown"
    result = summarize_suite(
        [good, unknown, {"name": "pending", "known_gt_count": 3, "comparison": None}]
    )
    assert result["status"] == "partial_suite"
    assert result["aggregates"] is None
    assert result["comparison"] is None
    assert result["completed_subset"]["project_names"] == ["small"]
    assert result["completed_subset"]["aggregates"]["base"]["micro"]["recall"] == 0.5


def test_exact_gain_with_lower_partial_coverage_does_not_pass_detection_criterion():
    from benchmark_suite import summarize_suite

    result = summarize_suite([project("one", 4, (1, 3, 4), (2, 0, 4))])
    assert result["comparison"]["macro_deltas"]["recall"] == 0.25
    assert result["comparison"]["macro_deltas"]["recall_with_partial"] == -0.5
    assert result["comparison"]["positive_detection_signal"] is False


def test_suite_rejects_wrong_gt_denominator():
    from benchmark_suite import summarize_suite

    wrong = project("wrong", 2, (1, 1, 3), (2, 0, 3))
    wrong["comparison"]["arms"][0]["draws"][0]["known_gt_count"] = 99
    with pytest.raises(ValueError, match="denominator"):
        summarize_suite([wrong])


def test_suite_aggregates_the_declared_final_arm_under_its_own_label():
    from benchmark_suite import summarize_suite

    item = project("tiny", 4, (1, 0, 1), (3, 0, 3))
    item["comparison"]["arms"][1]["label"] = "checkpoint-5"
    result = summarize_suite([item], checkpoint_label="checkpoint-5")
    assert set(result["aggregates"]) == {"base", "checkpoint-5"}
    assert result["aggregates"]["checkpoint-5"]["macro"]["recall"] == 0.75
    assert result["aggregates"]["base"]["macro"]["recall"] == 0.25


def test_next_project_bounds_runtime_and_never_restarts_finished_runs(tmp_path):
    from benchmark_suite import next_project

    child = tmp_path / "child"
    child.mkdir()
    (child / "config.json").write_text(json.dumps({"seconds_limit": 5400}))
    config = {
        "absolute_deadline": "2026-10-09T11:23:06+00:00",
        "runtime_host": "/srv/audit-grpo-runtime",
        "projects": [{"name": "one", "run_root": str(child)}],
    }
    now = datetime(2026, 10, 9, 11, 0, tzinfo=UTC)
    selected = next_project(config, now=now)
    assert selected.endswith("/results/child")
    assert json.loads((child / "config.json").read_text())["seconds_limit"] == 1326
    (child / "launcher-exit.txt").write_text("124")
    assert next_project(config, now=now) == "complete"
    (child / "launcher-exit.txt").unlink()
    assert next_project(config, now=datetime(2026, 10, 9, 11, 23, tzinfo=UTC)) == "complete"


def test_fail_closed_suite_does_not_launch_more_projects_after_failure(tmp_path):
    from benchmark_suite import next_project

    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "launcher-exit.txt").write_text("57\n")
    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "config.json").write_text(json.dumps({"seconds_limit": 5400}))
    config = {
        "absolute_deadline": "2026-10-10T12:30:00+00:00",
        "runtime_host": "/srv/audit-grpo-runtime",
        "stop_on_project_failure": True,
        "projects": [{"run_root": str(failed)}, {"run_root": str(pending)}],
    }
    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    assert next_project(config, now=now) == "complete"
    assert json.loads((pending / "config.json").read_text())["seconds_limit"] == 5400
    config["stop_on_project_failure"] = False
    assert next_project(config, now=now).endswith("/results/pending")


def test_report_cli_renders_saved_scores_and_rejects_changed_reference(tmp_path):
    import hashlib

    from benchmark_suite import main

    record = project("small", 2, (1, 1, 3), (2, 0, 3))
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(record["comparison"]))
    config = {
        "run_id": "test",
        "projects": [
            {
                "name": "small",
                "known_gt_count": 2,
                "reuse_comparison": str(path),
                "run_root": str(tmp_path),
                "comparison_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    out = tmp_path / "report"
    assert main(["report", "--config", str(config_path), "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["status"] == "matched_suite_completed"
    assert "| small | 2 | 50.00% | 100.00% |" in (out / "REPORT.md").read_text()
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        main(["report", "--config", str(config_path), "--out", str(out)])


def test_suite_report_accounts_for_new_generation_and_judge_reservations(tmp_path):
    from benchmark_suite import main

    record = project("small", 2, (1, 1, 3), (2, 0, 3))
    child = tmp_path / "child"
    child.mkdir()
    (child / "comparison.json").write_text(json.dumps(record["comparison"]))
    for label, seconds in (("base", 60), ("checkpoint-2", 120)):
        folder = child / label
        folder.mkdir()
        (folder / "receipt.json").write_text(
            json.dumps(
                {
                    "generation_seconds": seconds,
                    "completion_tokens": 100,
                    "total_seconds": seconds + 10,
                    "tokens_per_second": 1,
                }
            )
        )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "budget_charge_usd": 10.7,
                "calls": [
                    {
                        "operation_id": "new",
                        "estimated_usd": 0.1,
                        "budget_charge_usd": 0.2,
                        "status": "settled",
                    },
                    {
                        "operation_id": "unrelated",
                        "estimated_usd": 10,
                        "budget_charge_usd": 10,
                        "status": "settled",
                    },
                    {
                        "operation_id": "new",
                        "estimated_usd": None,
                        "budget_charge_usd": 0.5,
                        "status": "reserved",
                    },
                ],
            }
        )
    )
    config = {
        "run_id": "test",
        "judge_ledger": str(ledger),
        "rate_usd_per_hour": 2.51,
        "projects": [
            {"name": "small", "known_gt_count": 2, "run_id": "new", "run_root": str(child)}
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    main(["report", "--config", str(config_path), "--out", str(tmp_path / "report")])
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert result["cost"]["judge_known_usage_estimate_usd"] == 0.1
    assert result["cost"]["judge_conservative_charge_usd"] == 0.7
    assert result["cost"]["unsettled_judge_reservations"] == 1
    assert result["cost"]["generation_compute_estimate_usd"] == pytest.approx(0.1255)
    assert result["projects"][0]["execution"]["checkpoint-2"]["generation_seconds"] == 120
