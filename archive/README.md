# Archived training code

Code from finished experiments, moved out of `training-code/` on 2026-10-13. Nothing in
`training-code/` imports it, the launchers do not mount it, and its tests are not run.
Imports here still name the old layout (`scripts.finetuning.training`, `experiment`,
`benchmark`), so files do not run as-is. The local git baseline commit of
`audit-grpo-runtime/` has the runnable originals.

| Path | Was used for |
|---|---|
| `grpo/trainer.py`, `grpo/pilot_runner.py`, `inference/integration.py` | TRL GRPO pilot and its first integration check (superseded by the corrected cycle). |
| `experiment.py`, `launch-experiment.sh` | Expanded-experiment controller (`full-training-20261009`, `audited-xhigh-20261009`). |
| `continuation.py` | Multi-step `schedule` list inside one cycle (`continuation-expanded-20261009`). |
| `test_cycle_recovery.py`, `grpo/judge_recovery.py` | One-off saved-rollout and saved-grade recovery for the first reduced-run visit. |
| `corpus.py`, `corpus_build.py`, `corpus_export.py`, `corpus_dependencies.py` | LLM-reviewed corpus pipeline and npm/git dependency recovery. |
| `inference/benchmark.py` | Standalone short-generation benchmark CLI. |
| `inference/sampling_check.py`, `inference/check_sampling.py`, `inference/acceptance.py` | One-time sampler acceptance probes. |
| `provenance/` | Initial extraction map, its hashes and the implementation diff against it (local only, git-ignored: it holds rental paths). |
