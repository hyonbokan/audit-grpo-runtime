from testing.judge import write_workspace


def test_reward_worker_accepts_a_dependency_group_and_preserves_private_gt(tmp_path):
    from grpo.reward import load_split

    _, private = load_split(write_workspace(tmp_path), "train")
    assert private["audit/group-123"]["scorer_metadata"]["eligible_targets"][0]["id"] == "archive:1"
