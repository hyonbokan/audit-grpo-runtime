"""Exercise installed vLLM processed probabilities against Transformers warpers."""

import json
from pathlib import Path

import torch
from transformers import TopKLogitsWarper, TopPLogitsWarper
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler


def main():
    torch.manual_seed(3407)
    results = []
    for size in (128, 248320):
        logits = torch.randn(1, size, dtype=torch.float32)
        ids = torch.zeros(1, 1, dtype=torch.long)
        expected = TopKLogitsWarper(20)(ids, logits.clone())
        expected = TopPLogitsWarper(0.95)(ids, expected)
        expected = expected.log_softmax(-1)
        sampler = TopKTopPSampler(logprobs_mode="processed_logprobs")
        sampled, actual = sampler.forward_native(
            logits.clone(), {0: torch.Generator().manual_seed(3407)},
            torch.tensor([20]), torch.tensor([0.95]),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(actual.exp().sum(-1), torch.ones(1))
        assert torch.isfinite(actual[0, sampled[0]])
        raw = logits.log_softmax(-1)
        assert not torch.allclose(raw, actual)
        results.append({"vocabulary": size, "sampled_token": sampled.item(),
                        "processed_logprob": actual[0, sampled[0]].item(),
                        "raw_logprob": raw[0, sampled[0]].item(),
                        "finite_support": torch.isfinite(actual).sum().item()})
    Path("/checks/sampling-check.json").write_text(json.dumps({
        "status": "installed_sampler_matches_transformers_topk_topp",
        "temperature": 1, "top_k": 20, "top_p": 0.95, "checks": results,
        "full_model_distribution_verified": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
