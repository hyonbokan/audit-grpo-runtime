"""Synthetic judge, samples and reward workspace shared by the judge-path tests."""

import copy
import hashlib
import json
from pathlib import Path

# Matches the pinned values `reward.validate_contract` accepts; no private data.
REWARD_CONTRACT = {
    "judge": {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "high",
        "systems": {"sgr_match": "Match findings.", "sgr_reason": "Reason about the match."},
    },
    "reward": {"beta": 0.25, "volume_cost": 0.05, "max_findings": 32, "format_floor": -2},
}


def contract() -> dict:
    return copy.deepcopy(REWARD_CONTRACT)


def write_workspace(root: Path) -> Path:
    """A /workspace stand-in: the reward contract and one private/policy training row."""
    from scripts.finetuning.tools.training_rows import render_training_row

    (root / "configs").mkdir(parents=True)
    (root / "configs/reward_contract.json").write_text(json.dumps(REWARD_CONTRACT))
    row = private_training_row()
    for kind in ("private", "policy"):
        (root / "data" / kind).mkdir(parents=True)
    (root / "data/private/train.jsonl").write_text(json.dumps(row) + "\n")
    policy = {
        "row_id": row["row_id"],
        "messages": render_training_row(row, include_context=False)["messages"],
    }
    (root / "data/policy/train.jsonl").write_text(json.dumps(policy) + "\n")
    return root


def private_training_row() -> dict:
    content = "contract Vault {}"
    return {
        "row_id": "audit/group-123",
        "repository": "https://github.com/code-423n4/audit",
        "revision": "a" * 40,
        "group_id": "group-123",
        "profile": "none",
        "source_files": [
            {
                "path": "Vault.sol",
                "content": content,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        ],
        "scorer_metadata": {
            "targets": [
                {
                    "id": "archive:1",
                    "required_context": [],
                    "finding": {"contracts": ["Vault.sol"], "issue": "Known issue"},
                }
            ]
        },
    }


def private_rows():
    return {
        "row": {
            "source_files": [{"path": "A.sol", "content": "function withdraw() {}"}],
            "scorer_metadata": {
                "eligible_targets": [
                    {
                        "id": "g0",
                        "finding": {
                            "issue": "Missing guard",
                            "description": "Caller drains balance",
                            "contracts": ["A.sol"],
                        },
                    }
                ]
            },
        }
    }


def finding():
    return {
        "issue": "Missing guard",
        "severity": "high",
        "contracts": ["A.sol"],
        "description": {
            "context": "Withdrawal",
            "root_cause": "No caller check",
            "impact": "Caller drains balance",
            "recommendation": "Add guard",
            "code_references": [],
        },
    }


def sample(findings=None, **changes):
    final = json.dumps({"findings": [finding()] if findings is None else findings})
    return {
        "row_id": "row",
        "prompt_ids": [1],
        "completion_ids": [7, 8, 9],
        "logprobs": [-0.1, -0.2, -0.3],
        "raw_text": "<think>Guard absent.</think>" + final,
        "reasoning": "Guard absent.",
        "final_text": final,
        "finish_reason": "stop",
        **changes,
    }


class Judge:
    def __init__(
        self, *, fail=0, review=False, identity_only=False, contradiction=False, effort="high"
    ):
        self.fail = fail
        self.review = review
        self.identity_only = identity_only
        self.contradiction = contradiction
        self.effort = effort

    async def __call__(self, request):
        assert request["model"] == "gpt-6.1-sol"
        assert request["reasoning_effort"] == self.effort
        if self.fail:
            self.fail -= 1
            raise TimeoutError("judge unavailable")
        payload = request["payload"]
        if request["method"] == "sgr_match":
            refs = payload["candidate_findings"][0]["evidence_ids"]
            return {
                "assessments": [
                    {
                        "gt_id": "g0",
                        "candidate_id": "c0",
                        "candidate_evidence_ids": refs,
                        "identity_analysis": "Same absent guard",
                        "same_defect_operation": True,
                        "harm_analysis": "Caller drains balance",
                        "harm_established": not self.identity_only,
                    }
                ]
            }
        trace = payload["trace_evidence_ids"]
        source = payload["source_files"][0]["evidence_ids"]
        check = {
            "trace_evidence_ids": trace,
            "source_evidence_ids": source,
            "analysis": "Guard absence establishes the unauthorized withdrawal",
            "verdict": "supported",
        }
        return {
            "assessments": [
                {
                    "gt_id": "g0",
                    "candidate_id": "c0",
                    "reference_review": {"analysis": "Source checked", "needs_review": self.review},
                    "code_grounding": check,
                    "causal_mechanism": check,
                    "reachability": check,
                    "consequence": check,
                    "correction_analysis": {
                        "trace_evidence_ids": trace,
                        "analysis": "Settled conclusion",
                        "unresolved_critical_contradiction": self.contradiction,
                    },
                }
            ]
        }
