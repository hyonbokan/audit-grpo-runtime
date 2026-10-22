# Archived training code

Code from finished experiments, moved out of `training-code/` on 2026-10-13. Nothing in
`training-code/` imports it, the launchers do not mount it, and its tests are not run.
Imports here still name the old layout (`scripts.finetuning.training`, `experiment`,
`benchmark`), so files do not run as-is. The local git baseline commit of
`audit-grpo-runtime/` has the runnable originals. Files archived on 2026-10-22 keep that day's
imports; code removed from a module that stays maintained is named below with the last commit
that has it (`git show 82d7ad1:training-code/<path>`).

| Path | Was used for |
|---|---|
| `grpo/trainer.py`, `grpo/pilot_runner.py`, `inference/integration.py` | TRL GRPO pilot and its first integration check (superseded by the corrected cycle). |
| `experiment.py`, `launch-experiment.sh` | Expanded-experiment controller (`full-training-20261009`, `audited-xhigh-20261009`). |
| `continuation.py` | Multi-step `schedule` list inside one cycle (`continuation-expanded-20261009`). |
| `test_cycle_recovery.py`, `grpo/judge_recovery.py` | One-off saved-rollout and saved-grade recovery for the first reduced-run visit. |
| `corpus.py`, `corpus_build.py`, `corpus_export.py`, `corpus_dependencies.py` | LLM-reviewed corpus pipeline and npm/git dependency recovery. |
| `inference/benchmark.py` | Standalone short-generation benchmark CLI. |
| `inference/sampling_check.py`, `inference/check_sampling.py`, `inference/acceptance.py` | One-time sampler acceptance probes. |
| `inference/evaluation.py`, `evaluation_scope.json`, `fixtures/evaluation.json` | Kindless matched held-out comparison of two checkpoints on a policy row, graded by the reward worker. |
| `repeat_evaluation.py`, `inference/heal_samples.py` | Fresh-seed repeats of the Markdown evaluation; symmetric parser healing of JSON/Markdown answers. |
| `inference/test_weekend_evaluation.py`, `inference/test_continuation_benchmark.py` | Tests of the full-native-context ("weekend") and continuation-reference evaluations. |
| Removed at `82d7ad1` from `inference/benchmark_evaluation.py`, `benchmark_contract.py`, `benchmark_judge.py`, `benchmark_suite.py`, `launch-evaluation.sh`, `native_context_scope.json` | Solidity 1 evaluations with JSON or Markdown answers, the full-native-context profile, continuation references to a saved comparison, conditional (fence-removed) grading, Markdown repeats and reused suite comparisons. The free-text suite replaced them on 2026-10-21. |
| `inference/panel_evaluation.py` | Validation panels during training (`grpo_held_out_panel`); validation was dropped on 2026-10-22, so a run evaluates only its final checkpoint. |
| `provenance/` | Initial extraction map, its hashes and the implementation diff against it (local only, git-ignored: it holds rental paths). |
