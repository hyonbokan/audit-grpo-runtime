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

## Long-sequence fit inputs

`long_fit_inputs.py` seeds a training cycle that measures how long a sequence the update fits,
without a judge call. `build` takes a visit's `config.json` as template, recorded draws
(`samples.json` with `prompt_ids` and `completion_ids`, e.g. a frozen evaluation's base draws) and
the fit schedule's settings; it cuts each draw to `--length` tokens (prompt + completion), writes
the run's `config.json`, and stages rollout and grades (rewards [1, 0, 0, 0]) in
`<run>.fit-seed/`. After `cycle_contract prepare`, `seed` moves them into the run, and
`launch-cycle.sh --resume` runs the update. Draws recorded without sampling support get a
synthetic two-token support and zero sampled log-probs: the fit measures memory and time, not
correction health. Procedure: `HOST_SETUP.md` §7. Tests: `uvx --with pytest pytest -q
test_long_fit_inputs.py` (stdlib only).

```bash
python3 diagnostic-tools/long_fit_inputs.py build --template runs/<visit>/config.json \
  --samples runs/<eval>/base/samples.json --run runs/<fit-run> --length 96000 --vocab 248320 \
  --schedule-id <fit-schedule> --training-cutoff <its cutoff> --seconds-limit 14400 \
  --offload-layers 64 --tiled-mlp true \
  --update-placement '{"gpus": [0, 1], "embedding": "gpu", "max_memory_gib": 30}'
```
