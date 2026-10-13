import json

import pytest

from merge_comparison import (
    compare,
    fixed_sequence_logprobs,
    numerical_agreement,
    phase_timings,
)


def test_decode_excludes_prefill_first_token_and_queued_time():
    result = phase_timings(
        {
            "queued_ts": 9.0,
            "scheduled_ts": 10.0,
            "first_token_ts": 12.0,
            "last_token_ts": 15.0,
        },
        7,
    )
    assert result["queue_seconds"] == 1.0
    assert result["prefill_to_first_token_seconds"] == 2.0
    assert result["decode_seconds"] == 3.0
    assert result["decode_tokens_per_second"] == 2.0


def test_chosen_prompt_logprobs_score_fixed_suffix_without_selected_token_limit():
    from vllm.logprobs import Logprob

    rows = [None, {90: Logprob(-3.0)}, {1: Logprob(-1.0)}, {2: Logprob(-2.0)}]
    assert fixed_sequence_logprobs(rows, 2, [1, 2]) == [-1.0, -2.0]
    with pytest.raises(ValueError, match="aligned"):
        fixed_sequence_logprobs(rows, 2, [1, 3])


def test_missing_or_invalid_timestamps_are_not_accepted():
    with pytest.raises(ValueError):
        phase_timings(
            {
                "queued_ts": 0.0,
                "scheduled_ts": 0.0,
                "first_token_ts": 0.0,
                "last_token_ts": 0.0,
            },
            128,
        )


def test_numerical_gate_retains_failed_positions_and_requires_both_limits():
    result = numerical_agreement([-1.0, -2.0], [-1.01, -2.08], 0.05, 0.01)
    assert not result["passed"]
    assert result["positions_exceeding_atol"] == [1]
    assert result["absolute_differences"][1] == pytest.approx(0.08)
    # Max error passes here but RMS does not: neither gate may be omitted.
    assert not numerical_agreement([-1.0, -2.0], [-1.02, -2.02], 0.05, 0.01)["passed"]
    assert numerical_agreement([-1.0, -2.0], [-1.001, -2.001], 0.05, 0.01)["passed"]
    with pytest.raises(ValueError):
        numerical_agreement([-1.0], [], 0.05, 0.01)


def test_incomplete_saved_arms_cannot_pass_comparison(tmp_path):
    for arm in ("adapter", "merged"):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "receipt.json").write_text(
            json.dumps(
                {"status": "completed", "fixed_sequences": [], "measurements": []}
            )
        )
    with pytest.raises(ValueError, match="Incomplete"):
        compare(
            {"prompts": [{"name": "case", "sha256": "a"}], "repetitions": 2}, tmp_path
        )


def test_report_requires_stable_repeats_and_retains_numerical_failure(tmp_path):
    config = {
        "prompts": [{"name": "case", "sha256": "a"}],
        "repetitions": 2,
        "logprob_atol": 0.05,
        "logprob_rms_limit": 0.01,
        "max_relative_decode_rate_spread": 0.1,
    }
    for arm, rates, score in [
        ("adapter", [7.0, 7.1], -1.0),
        ("merged", [30.0, 40.0], -1.08),
    ]:
        (tmp_path / arm).mkdir()
        receipt = {
            "status": "completed",
            "fixed_sequences": [
                {"prompt": "case", "token_ids": [1], "chosen_raw_logprobs": [score]}
            ],
            "measurements": [
                {
                    "prompt": "case",
                    "prompt_sha256": "a",
                    "repeat": i,
                    "decode_tokens_per_second": rate,
                    "prefill_to_first_token_seconds": 0.5,
                }
                for i, rate in enumerate(rates)
            ],
        }
        (tmp_path / arm / "receipt.json").write_text(json.dumps(receipt))
    result = compare(config, tmp_path)
    assert not result["numerical_agreement_passed"]
    assert not result["repeat_stability_passed"]
    assert not result["acceptance_passed"]
    assert result["repeat_stability"][1]["relative_spread"] == pytest.approx(2 / 7)


@pytest.mark.parametrize(
    "preservation_status, accepted",
    [("preserved", True), ("preservation_failed", False)],
)
def test_complete_acceptance_requires_source_preservation(
    tmp_path, preservation_status, accepted
):
    config = {
        "prompts": [{"name": "case", "sha256": "a"}],
        "repetitions": 2,
        "logprob_atol": 0.05,
        "logprob_rms_limit": 0.01,
        "max_relative_decode_rate_spread": 0.1,
    }
    for arm in ("adapter", "merged"):
        (tmp_path / arm).mkdir()
        receipt = {
            "status": "completed",
            "fixed_sequences": [
                {"prompt": "case", "token_ids": [1], "chosen_raw_logprobs": [-1.0]}
            ],
            "measurements": [
                {
                    "prompt": "case",
                    "prompt_sha256": "a",
                    "repeat": i,
                    "decode_tokens_per_second": 10.0,
                    "prefill_to_first_token_seconds": 0.5,
                }
                for i in range(2)
            ],
        }
        (tmp_path / arm / "receipt.json").write_text(json.dumps(receipt))
    (tmp_path / "preservation-after.json").write_text(
        json.dumps({"status": preservation_status})
    )
    result = compare(config, tmp_path)
    assert result["acceptance_passed"] is accepted
