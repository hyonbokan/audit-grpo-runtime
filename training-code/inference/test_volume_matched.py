"""Volume-matched recall: the base is cut to the trained report length before comparing."""

import pytest

import volume_matched
from testing.benchmark import verdict_draw as draw


def test_cutting_a_draw_keeps_only_the_matches_inside_the_first_findings():
    full = draw(4, [0, 3, 9], findings=12)
    assert volume_matched.recall_at_length(full, 1) == pytest.approx(1 / 4)
    assert volume_matched.recall_at_length(full, 4) == pytest.approx(2 / 4)
    assert volume_matched.recall_at_length(full, 12) == pytest.approx(3 / 4)
    assert volume_matched.recall_at_length(full, 99) == pytest.approx(3 / 4)
    assert volume_matched.recall_at_length(full, 0) == 0.0


def test_a_fractional_length_interpolates_between_the_two_nearest_whole_cuts():
    full = draw(4, [0, 3], findings=6)
    # Cuts at 3 and 4 findings give 1/4 and 2/4; 3.5 findings is halfway.
    assert volume_matched.recall_at_length(full, 3.5) == pytest.approx(1.5 / 4)


def test_gain_is_trained_recall_minus_base_recall_cut_to_the_trained_mean_length():
    base = [draw(4, [0, 10], 20), draw(4, [2, 12], 20)]
    trained = [draw(4, [0, 1], 2), draw(4, [0], 4)]
    result = volume_matched.volume_matched_gain(base, trained)
    assert result["trained_mean_findings"] == 3.0
    assert result["trained_recall"] == pytest.approx((2 / 4 + 1 / 4) / 2)
    # Cut to 3 findings: base draw one keeps its match at 0; draw two keeps its match at 2.
    assert result["base_recall_at_trained_length"] == pytest.approx((1 / 4 + 1 / 4) / 2)
    assert result["base_recall_uncut"] == pytest.approx(2 / 4)
    assert result["gain"] == pytest.approx(0.375 - 0.25)
    assert (result["base_graded_draws"], result["trained_graded_draws"]) == (2, 2)


def test_only_graded_draws_count_and_a_missing_arm_gives_no_gain():
    base = [draw(4, [0], 5), draw(4, [], 0, status="truncated")]
    trained = [draw(4, [0], 5), draw(4, [], 0, status="unknown")]
    result = volume_matched.volume_matched_gain(base, trained)
    assert (result["base_graded_draws"], result["trained_graded_draws"]) == (1, 1)
    assert volume_matched.volume_matched_gain([draw(4, [], 0, "truncated")], trained) is None
    assert volume_matched.volume_matched_gain(base, []) is None


def test_an_empty_trained_report_cuts_the_base_to_nothing():
    base = [draw(2, [0], 3)]
    trained = [draw(2, [], 0)]
    result = volume_matched.volume_matched_gain(base, trained)
    assert result["base_recall_at_trained_length"] == 0.0
    assert result["gain"] == 0.0
