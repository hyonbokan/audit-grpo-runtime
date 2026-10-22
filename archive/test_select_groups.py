"""Seeded random project schedules must be reproducible, diverse and leak-free."""

import pytest

from select_groups import select_groups


def tokens(rows):
    return {row: {"prompt_tokens": size} for row, size in rows.items()}


ROWS = {
    "a/group-1": 1000,
    "a/group-2": 2000,
    "b/all-scope": 3000,
    "c/group-1": 4000,
    "c/group-2": 5000,
    "d/group-1": 90000,
    "e/all-scope": 6000,
    "heldout/all-scope": 7000,
    "used/all-scope": 8000,
}


def choose(**changes):
    arguments = {
        "train_rows": list(ROWS),
        "non_train_rows": ["heldout/group-9"],
        "requests": tokens(ROWS),
        "exclude_projects": {"used"},
        "seed": 11,
        "count": 4,
        "max_prompt_tokens": 10000,
    }
    arguments.update(changes)
    return select_groups(**arguments)


def test_selection_is_reproducible_and_changes_with_the_seed():
    assert choose()["order"] == choose()["order"]
    assert any(choose(seed=seed)["order"] != choose()["order"] for seed in range(12, 40))


def test_each_project_appears_once_before_any_project_repeats():
    order = choose(count=4)["order"]
    projects = [row.split("/")[0] for row in order]
    assert len(set(projects)) == len(projects) == 4


def test_excluded_held_out_oversized_and_used_projects_are_never_selected():
    for seed in range(50):
        order = choose(seed=seed, count=4)["order"]
        assert not {"d/group-1", "heldout/all-scope", "used/all-scope"} & set(order)


def test_record_keeps_the_audit_trail_without_scores():
    record = choose()
    assert record["seed"] == 11
    assert record["eligible_projects"] == 4
    assert record["max_prompt_tokens"] == 10000
    assert record["prompt_tokens"] == {row: ROWS[row] for row in record["order"]}
    assert not [key for key in record if "score" in key or "reward" in key]


def test_asking_for_more_projects_than_are_eligible_fails_instead_of_repeating():
    with pytest.raises(ValueError, match="only 4 eligible projects"):
        choose(count=5)


def test_a_row_without_a_verified_request_is_rejected():
    with pytest.raises(ValueError, match="verified request"):
        choose(requests=tokens({"a/group-1": 1000}))
