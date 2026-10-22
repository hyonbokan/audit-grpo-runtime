"""Suite scores keep project weighting and missing grades explicit."""

import json
from pathlib import Path

import pytest

from testing.benchmark import project


def summarize(rows, label="checkpoint-2"):
    from benchmark_suite import summarize_suite

    return summarize_suite(rows, sample_count=4, checkpoint_label=label)


def test_suite_counts_each_gt_once_and_rejects_a_comparison_that_is_not_free_text():
    row = project("tiny", 4, (1, 1, 3), (2, 1, 4))
    result = summarize([row])
    assert result["aggregates"]["checkpoint-2"]["known_gt_opportunities"] == 16
    assert result["aggregates"]["checkpoint-2"]["micro"]["recall"] == 0.5
    assert result["projects"][0]["arms"]["base"]["mean_findings"] == 3
    del row["comparison"]["output_format"]
    with pytest.raises(ValueError, match="free-text"):
        summarize([row])


def test_suite_reports_macro_micro_and_partial_metrics_without_reusing_denominators():
    result = summarize(
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
    good = project("small", 2, (1, 1, 3), (2, 0, 3))
    unknown = project("large", 4, (1, 0, 2), (2, 1, 4))
    unknown["comparison"]["draws"]["checkpoint-2"][0]["status"] = "unknown"
    result = summarize(
        [good, unknown, {"name": "pending", "known_gt_count": 3, "comparison": None}]
    )
    assert result["status"] == "partial_suite"
    assert result["aggregates"] is None
    assert result["comparison"] is None
    assert result["completed_subset"]["project_names"] == ["small"]
    assert result["completed_subset"]["aggregates"]["base"]["micro"]["recall"] == 0.5


def test_exact_gain_with_lower_partial_coverage_does_not_pass_detection_criterion():
    result = summarize([project("one", 4, (1, 3, 4), (2, 0, 4))])
    assert result["comparison"]["macro_deltas"]["recall"] == 0.25
    assert result["comparison"]["macro_deltas"]["recall_with_partial"] == -0.5
    assert result["comparison"]["positive_detection_signal"] is False


def test_suite_rejects_wrong_gt_denominator():
    wrong = project("wrong", 2, (1, 1, 3), (2, 0, 3))
    wrong["comparison"]["draws"]["base"][0]["known_gt_count"] = 99
    with pytest.raises(ValueError, match="denominator"):
        summarize([wrong])


def test_suite_aggregates_the_declared_final_arm_under_its_own_label():
    item = project("tiny", 4, (1, 0, 1), (3, 0, 3), label="checkpoint-5")
    result = summarize([item], label="checkpoint-5")
    assert set(result["aggregates"]) == {"base", "checkpoint-5"}
    assert result["aggregates"]["checkpoint-5"]["macro"]["recall"] == 0.75
    assert result["aggregates"]["base"]["macro"]["recall"] == 0.25


def trained_suite(tmp_path, names=("munchables", "vultisig", "phi")):
    """A suite config whose projects each hold a prepared-to-run trained-arm config."""
    from testing.benchmark import trained_free_text_configuration

    projects = []
    for name in names:
        root = tmp_path / name
        root.mkdir()
        child = trained_free_text_configuration(f"solidity/{name}")
        (root / "config.json").write_text(json.dumps(child))
        projects.append(
            {
                "name": name,
                "known_gt_count": 3,
                "run_id": f"suite-{name}",
                "run_root": str(root),
                "base_run_root": str(tmp_path / f"base-{name}"),
            }
        )
    return suite_config(projects, absolute_deadline="2026-10-23T00:00:00+00:00")


def suite_config(projects, **settings):
    """The suite config panels.evaluation_configs writes, around the given projects."""
    return {
        "run_id": "suite",
        "trained_label": "checkpoint-7",
        "sampling": {"n": 4},
        "template_options": {"reasoning_effort": "xhigh"},
        "output_format": "free_text",
        "rate_usd_per_hour": 3.6,
        "judge_ledger": "/results/ledger.json",
        "projects": projects,
        **settings,
    }


def run_report(tmp_path, config):
    from benchmark_suite import main

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    out = tmp_path / "report"
    assert main(["report", "--config", str(config_path), "--out", str(out)]) == 0
    return json.loads((out / "report.json").read_text()), (out / "REPORT.md").read_text()


def finished_project(tmp_path, name, base=(1, 1, 3), trained=(2, 0, 3)):
    """A project whose comparison is saved, with the frozen base run beside it."""
    record = project(name, 2, base, trained, label="checkpoint-7")
    child, frozen = tmp_path / name, tmp_path / f"base-{name}"
    child.mkdir()
    frozen.mkdir()
    (child / "comparison.json").write_text(json.dumps(record["comparison"]))
    return {
        "name": name,
        "known_gt_count": 2,
        "run_id": name,
        "run_root": str(child),
        "base_run_root": str(frozen),
    }


def test_suite_settings_run_every_unfinished_project_within_the_deadline(tmp_path, monkeypatch):
    import runtime_io
    from benchmark_suite import suite_settings
    from testing.clock import freeze

    config = trained_suite(tmp_path)
    (tmp_path / "vultisig/comparison.json").write_text("{}")
    freeze(monkeypatch, runtime_io, "2026-10-22T00:00:00+00:00")
    child = json.loads((tmp_path / "munchables/config.json").read_text())
    assert suite_settings(config) == [
        child["native_image"],
        child["rollout_image"],
        str(child["seconds_limit"]),
        str(child["judge_seconds_limit"]),
        "checkpoint-7",
        "munchables phi",
    ]
    freeze(monkeypatch, runtime_io, "2026-10-22T23:00:00+00:00")
    assert suite_settings(config)[2] == str(3600 - 60)


@pytest.mark.parametrize(
    "change",
    [
        lambda root: (root / "prepared.json").write_text("{}"),
        lambda root: (root / "config.json").write_text(
            json.dumps(
                {
                    **json.loads((root / "config.json").read_text()),
                    "merged_model_path": "/workspace/cache/other",
                }
            )
        ),
        lambda root: (root / "config.json").write_text(
            json.dumps(
                {
                    **json.loads((root / "config.json").read_text()),
                    "generation_order": ["checkpoint-8"],
                }
            )
        ),
    ],
)
def test_suite_refuses_unfinished_restarts_and_projects_that_cannot_share_an_engine(
    tmp_path, change
):
    from benchmark_suite import suite_settings

    config = trained_suite(tmp_path)
    change(tmp_path / "phi")
    with pytest.raises(ValueError):
        suite_settings(config)


def test_rollout_manifest_names_each_unfinished_projects_arm_and_one_receipt(tmp_path):
    from benchmark_suite import write_rollout_manifest

    config = trained_suite(tmp_path)
    (tmp_path / "vultisig/comparison.json").write_text("{}")
    out = tmp_path / "suite"
    out.mkdir()
    path = write_rollout_manifest(config, out)
    assert json.loads(path.read_text()) == {
        "arms": [
            {
                "config": str(tmp_path / f"{name}/checkpoint-7-config.json"),
                "out": str(tmp_path / f"{name}/checkpoint-7"),
            }
            for name in ("munchables", "phi")
        ],
        "receipt": str(out / "rollout-receipt.json"),
    }


def empty_ledger(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    return str(ledger)


def test_report_cli_renders_saved_scores(tmp_path):
    projects = [finished_project(tmp_path, "small")]
    result, markdown = run_report(
        tmp_path, suite_config(projects, judge_ledger=empty_ledger(tmp_path))
    )
    assert result["status"] == "matched_suite_completed"
    assert "| small | 2 | 50.00% | 100.00% |" in markdown


def test_suite_report_counts_only_this_suites_judge_calls_and_reservations(tmp_path):
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
    projects = [dict(finished_project(tmp_path, "small"), run_id="new")]
    result, _ = run_report(tmp_path, suite_config(projects, judge_ledger=str(ledger)))
    assert result["cost"]["judge_known_usage_estimate_usd"] == 0.1
    assert result["cost"]["judge_conservative_charge_usd"] == 0.7
    assert result["cost"]["unsettled_judge_reservations"] == 1
    assert result["cost"]["shared_ledger_charge_usd"] == 10.7


def test_free_text_suite_compares_the_trained_arm_with_the_frozen_base_draws():
    rows = [
        project("a", 4, (1, 0, 2), (2, 0, 2), label="checkpoint-7"),
        project("b", 2, (0, 1, 1), (1, 1, 2), label="checkpoint-7"),
    ]
    result = summarize(rows, label="checkpoint-7")
    assert result["status"] == "matched_suite_completed"
    assert result["aggregates"]["base"]["exact_matches"] == 4 * 1 + 4 * 0
    assert result["aggregates"]["checkpoint-7"]["exact_matches"] == 4 * 2 + 4 * 1
    assert result["comparison"]["macro_deltas"]["recall"] == pytest.approx((0.25 + 0.5) / 2)


def test_suite_report_reads_the_frozen_base_receipt_but_charges_only_new_generation(tmp_path):
    project_record = finished_project(tmp_path, "small")
    frozen = tmp_path / "base-small"
    (frozen / "base").mkdir()
    receipt = {"generation_seconds": 600, "completion_tokens": 9, "total_seconds": 600}
    (frozen / "base/receipt.json").write_text(json.dumps(receipt))
    config = suite_config([project_record], judge_ledger=empty_ledger(tmp_path))
    result, _ = run_report(tmp_path, config)
    assert result["projects"][0]["execution"]["base"]["generation_seconds"] == 600
    assert result["cost"]["generation_compute_estimate_usd"] == 0


def test_suite_report_charges_one_shared_generation_call_once(tmp_path):
    projects = []
    for name in ("a", "b"):
        record = finished_project(tmp_path, name)
        shared = {"completion_tokens": 9, "shared_generation_receipt": "rollout-receipt.json"}
        (Path(record["run_root"]) / "checkpoint-7").mkdir()
        (Path(record["run_root"]) / "checkpoint-7/receipt.json").write_text(json.dumps(shared))
        projects.append(record)
    out = tmp_path / "report"
    out.mkdir()
    (out / "rollout-receipt.json").write_text(
        json.dumps({"status": "versioned_rollout_saved", "generation_seconds": 240})
    )
    result, _ = run_report(tmp_path, suite_config(projects, judge_ledger=empty_ledger(tmp_path)))
    assert result["cost"]["generation_compute_estimate_usd"] == pytest.approx(0.24)
    assert result["shared_generation"]["generation_seconds"] == 240
    assert result["projects"][0]["execution"]["checkpoint-7"]["completion_tokens"] == 9
