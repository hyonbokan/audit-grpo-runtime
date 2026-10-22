# Checkpoint inference diagnostic

These tools diagnose inference and do not update training weights or rollout paths.
They reuse the frozen training implementation's adapter mapping/worker verification.
The active training-code source manifest remains unchanged.

`merge_checkpoint.py` creates a fresh inference-only safetensors model from a pinned
base and adapter. It validates the supported LoRA configuration, casts adapter
operands to BF16, accumulates each256-row block in FP32, saves BF16 weights, verifies
all untouched tensors and records source/output hashes. Original checkpoint state
is never written. Unsupported scaling/adapter features fail explicitly.

`merge_comparison.py` runs separate runtime-LoRA and normal merged-model vLLM
workers, then compares recorded timing and fixed-sequence raw log-probabilities.
Merged loading uses `enable_lora=False` and no LoRARequest. Prompts, sampling,
precision and concurrency come from the same run-local configuration. Engine
timestamps separate queueing, prefill-to-first-token and subsequent decoding;
prefill timing includes engine overhead. Empty/misaligned evidence cannot pass.
Numerical and repeated-rate gates preserve their failures without retries.

Run settings, bounded GPU scheduling, evidence and CPU verification:
`runs/checkpoint2-merge-diagnostic-20261009`. The launcher coordinates with the
current evaluation after its GPU work ends naturally; CPU grading can continue.
No paid APIs or network access
are used by either GPU worker. Tests use the pinned Python3.12 Docker environment
and the existing vendored test tools. The original checkpoint remains the sole
resumable training artifact; merged weights have no optimizer/scheduler/RNG state.
