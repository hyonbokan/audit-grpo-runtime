"""Healing re-formats what a draw already wrote; it never audits or touches strict results."""

import asyncio
import hashlib
import json
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 14), reason="Existing judge router is 3.14")

RAW = (
    "Let me look. The `withdraw` function in Vault.sol sends ETH before updating the balance, "
    "so a reentrant call drains funds. Also fee rounding in `Pool.sol` loses dust."
)


def report(*findings):
    return "\n\n".join(
        f"## Finding {index}: {title}\nSeverity: {severity}\nContracts: `{path}`\n\n"
        f'{body}\nSource quote: "{quote}"'
        for index, (title, severity, path, body, quote) in enumerate(findings, 1)
    )


GOOD = report(
    (
        "Reentrancy in withdraw",
        "High",
        "Vault.sol",
        "Funds leave before the balance update.",
        "sends ETH before updating the balance",
    )
)


def sample(text=RAW, finish="stop"):
    return {
        "row_id": "p/all",
        "raw_text": text,
        "final_text": text,
        "reasoning": "",
        "finish_reason": finish,
        "completion_ids": [1, 2, 3],
    }


class FakeHealer:
    def __init__(self, markdown):
        self.markdown, self.prompts = markdown, []

    async def __call__(self, prompt):
        self.prompts.append(prompt)
        return {"report_markdown": self.markdown}


def test_prompt_contains_only_the_model_output_and_forbids_new_findings():
    from heal_samples import healing_prompt

    prompt = healing_prompt(RAW)
    assert RAW in prompt
    assert "Do not add" in prompt and "Source quote" in prompt
    assert "No vulnerabilities found." in prompt


def test_healed_sample_passes_the_strict_parser_and_keeps_the_original_identity():
    from benchmark_contract import parse_markdown
    from heal_samples import heal_sample

    original = sample()
    healed = asyncio.run(heal_sample(original, FakeHealer(GOOD)))
    findings = parse_markdown(healed["final_text"])
    assert [row["issue"] for row in findings] == ["Reentrancy in withdraw"]
    assert "Source quote" not in healed["final_text"]
    assert healed["finish_reason"] == "stop"
    assert healed["completion_ids"] == original["completion_ids"]
    assert healed["raw_text"] == original["raw_text"]
    assert original["final_text"] == RAW
    meta = healed["healing"]
    assert meta["untraceable_findings"] == []
    assert meta["findings"] == 1
    assert meta["source_sha256"] == hashlib.sha256(RAW.encode()).hexdigest()


def test_a_finding_whose_quote_is_not_in_the_output_is_flagged_not_dropped():
    from benchmark_contract import parse_markdown
    from heal_samples import heal_sample

    invented = report(
        ("Real", "High", "Vault.sol", "x", "sends ETH   before updating  the balance"),
        ("Invented", "High", "Vault.sol", "y", "an oracle can be manipulated"),
    )
    healed = asyncio.run(heal_sample(sample(), FakeHealer(invented)))
    assert len(parse_markdown(healed["final_text"])) == 2
    assert healed["healing"]["untraceable_findings"] == ["Invented"]


def test_empty_report_is_allowed_and_an_unparseable_reply_is_an_error():
    from benchmark_contract import parse_markdown
    from heal_samples import heal_sample

    healed = asyncio.run(heal_sample(sample(), FakeHealer("No vulnerabilities found.")))
    assert parse_markdown(healed["final_text"]) == []
    with pytest.raises(ValueError, match="Markdown finding"):
        asyncio.run(heal_sample(sample(), FakeHealer("Here are some thoughts")))


def test_every_draw_is_healed_with_the_same_prompt_rule_and_strict_inputs_are_untouched(tmp_path):
    from heal_samples import heal_samples

    source = tmp_path / "samples.json"
    draws = [sample(), sample(text="Cut off mid-sent")]
    source.write_text(json.dumps(draws))
    before = source.read_bytes()
    healer = FakeHealer(GOOD)
    out = tmp_path / "healed"
    healed = asyncio.run(heal_samples(source, out, healer))
    assert len(healer.prompts) == 2
    assert [row["healing"]["untraceable_findings"] for row in healed] == [
        [],
        ["Reentrancy in withdraw"],
    ]
    assert source.read_bytes() == before
    assert json.loads((out / "samples.json").read_text()) == healed
    with pytest.raises(FileExistsError):
        asyncio.run(heal_samples(source, out, healer))
