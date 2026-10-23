"""Synthetic free-text judges, samples and corpus rows shared by the reward tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import free_text_corpus

ROW_ID = "audit/group-123"
SOURCE_FILES = [{"path": "Vault.sol", "content": "contract Vault {}", "sha256": "0" * 64}]
TRUTH = [
    {
        "issue": "Missing guard",
        "severity": "High",
        "description": "Caller drains balance",
        "contracts": ["Vault.sol"],
        "recommendation": "",
    },
    {
        "issue": "Stale price",
        "severity": "Medium",
        "description": "Oracle is read without a freshness check",
        "contracts": ["Vault.sol"],
        "recommendation": "",
    },
]
TRUTH_IDS = ["archive:1", "archive:2"]
TEMPLATE = "Truth:\n{ truth_finding }\nFindings:\n{ junior_findings }"


def private_row() -> dict:
    """A derived-corpus private row: source files, scorer metadata, truth and its ids."""
    return {
        "row_id": ROW_ID,
        "repository": "audit",
        "revision": "a" * 40,
        "group_id": "group-123",
        "profile": "none",
        "source_files": SOURCE_FILES,
        "complementary_context": None,
        "scorer_metadata": {"targets": []},
        "truth": TRUTH,
        "truth_ids": TRUTH_IDS,
    }


def write_corpus(root: Path) -> Path:
    """A derived corpus holding one train row, with prompts as `free_text_corpus` renders them."""
    row = private_row()
    for kind in ("private", "policy"):
        (root / "data" / kind).mkdir(parents=True)
    (root / "data/private/train.jsonl").write_text(json.dumps(row) + "\n")
    policy = {"row_id": ROW_ID, "messages": free_text_corpus.policy_messages(row)}
    (root / "data/policy/train.jsonl").write_text(json.dumps(policy) + "\n")
    (root / free_text_corpus.JUDGE_TEMPLATE).write_text(TEMPLATE)
    return root


def report_finding(title: str) -> dict:
    return {"issue": title, "description": f"{title} in detail", "severity": "N/A", "contracts": []}


def matching(title: str) -> dict:
    """The extracted finding the fake matcher pairs with the truth finding of that issue."""
    return report_finding(f"match:{title}")


def sample(report: str = "report", finish_reason: str = "stop", **changes) -> dict:
    return {
        "row_id": ROW_ID,
        "prompt_ids": [1],
        "completion_ids": [7, 8, 9],
        "logprobs": [-0.1, -0.2, -0.3],
        "raw_text": f"<think>thoughts</think>{report}",
        "reasoning": "thoughts",
        "final_text": report,
        "finish_reason": finish_reason,
        **changes,
    }


class Judges:
    """Extraction by report text and matching by `match:<issue>` markers; records every call."""

    def __init__(self, reports: dict[str, list[dict]], *, failures: int = 0):
        self.reports, self.failures = reports, failures
        self.extract_prompts: list[str] = []
        self.match_prompts: list[str] = []
        self.attempts = 0

    def new_attempt(self) -> None:
        self.attempts += 1

    async def extract(self, prompt: str) -> dict:
        self.extract_prompts.append(prompt)
        if self.failures:
            self.failures -= 1
            raise ValueError("extraction unavailable")
        report = prompt.split("## Report\n", 1)[1].strip()
        return {"findings": self.reports[report]}

    async def match(self, prompt: str) -> dict:
        self.match_prompts.append(prompt)
        batch = json.loads(prompt.split("Findings:\n", 1)[1])
        target = json.loads(prompt.split("Truth:\n", 1)[1].split("\nFindings:", 1)[0])
        for entry in batch:
            if entry["issue"] == f"match:{target['issue']}":
                return verdict(True, entry["index"], target["severity"])
        return verdict(False, -1, target["severity"])


def verdict(is_match: bool, index: int, severity: str) -> dict:
    return {
        "is_match": is_match,
        "is_partial_match": False,
        "explanation": "fake",
        "severity_from_junior_auditor": "N/A",
        "severity_from_truth": severity,
        "index_of_finding_from_junior_auditor": index,
    }


class FakeRouter(Judges):
    """The `llm_router` surface `BenchmarkJudge` calls, answered by the marker judges."""

    def __init__(self, reports: dict[str, list[dict]], *, failures: int = 0):
        super().__init__(reports, failures=failures)
        self.event_dir = None
        self.closed = False

    async def complete(self, model, messages, *, response_model, **options):
        assert model == "gpt-6-luna" and options["max_retries"] == 0
        prompt = messages[0]["content"]
        answer = await (
            self.extract if response_model.__name__ == "ExtractedReport" else self.match
        )(prompt)
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        return SimpleNamespace(
            content=response_model(**answer),
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=0,
            raw_response=SimpleNamespace(
                model=model,
                id="r0",
                status="completed",
                usage=usage,
                model_dump_json=lambda **_: '{"id":"r0","status":"completed"}',
            ),
        )

    async def close(self):
        self.closed = True
