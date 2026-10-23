# Compute planning: memory, multiple GPUs and larger models

How much GPU and host memory LoRA GRPO with long reasoning traces needs, what multiple GPUs do and
do not buy, what we measured on our own host, and hardware tiers for larger open-weights models.
The estimator and its registries are in [`planning-tools/`](../planning-tools/README.md); the
measured numbers below are its calibration. Estimates are planning numbers (roughly ±2×), not
quotes: a fit test on the target hardware decides. Budgets in money are kept in `private/`.

Written 2026-10-23. Target candidate: GLM-5.3-Flash in BF16 (no quantization).

## Summary

1. **Two memory budgets.** Rollout needs every weight on the GPUs plus the KV cache. The update
   needs the weights (resident, or streamed from host RAM) plus *sequence memory*: the working
   tensors of the whole sequence inside the layer being computed.
2. **More GPUs split weights, not sequences.** Layer split, DDP and FSDP/ZeRO leave the whole
   sequence on one GPU. Only context parallelism (CP) divides it; tensor parallelism divides part.
   Measured sequence memory: 0.52 GiB per 1k tokens per GPU at hidden size 5120, 0.38 with tiled
   MLP, whatever the GPU count.
3. **Our host (2 × RTX A6000 48 GB, 251 GB RAM) trains Qwen3.8-27B up to 96k tokens** (all
   layers streamed + tiled MLP); 128k fails. GLM-4.7-Flash (30B, 3B active) would fit up to 128k.
4. **GLM-5.3-Flash in BF16 (596 GiB of weights) needs 8 × H200/B200/B300 or 16 × H100.** It does
   not fit one 8 × H100 node in BF16 (the FP8 release would). GLM-5.3 (1.4 TiB) needs 16 × B200,
   16 × H200, 8 × B300 or 24+ × H100.
5. **Big hardware pays only when many groups run at once.** One 60k-token draw takes ≥ ~6 min on
   any GPU (decode latency floor). Our trainer runs one group per update and reloads the model
   between phases, so it cannot use an 8-GPU node well. Scaling up means a different stack (verl,
   slime or NeMo RL) with batched groups, sharding and context parallelism.
6. **Measure the base before training.** Evaluate a candidate's base on the benchmark first; a
   training budget makes sense only if the base already audits well.

## 1. Where the memory goes

| Phase | GPU memory | Host RAM | Scales with |
|---|---|---|---|
| Rollout (vLLM/SGLang) | all weights + KV cache of every concurrent sequence | small | total params; KV per token × concurrent tokens |
| Update (LoRA) | resident weight share (or a small slot pool when streamed) + sequence memory + LoRA/optimizer state + fixed | streamed weights (pinned) + checkpointed layer inputs | sequence length × hidden size; active params for time |

- **Weights:** total parameters × 2 bytes (BF16). In a mixture-of-experts model every expert
  counts, not only the active ones.
- **KV cache:** per token, full-attention layers × cached width × 2 B. Latent (MLA), sparse (DSA),
  sliding-window and linear attention keep it small: 1–11 GiB per 128k sequence for the models
  below.
- **Sequence memory:** with gradient checkpointing and checkpoint inputs on host, the GPU holds
  one layer's recompute for the whole sequence (MLP intermediates, attention projections, an fp32
  copy of the LoRA input). MoE layers add routed copies and expert intermediates (estimated, not
  measured).
- **Host RAM:** checkpointed inputs = layers × hidden × 2 B per token (Qwen3.8-27B: 42 GB per
  64k sequence, measured), plus pinned weights when streamed.

## 2. What more GPUs buy

| Approach | Splits | Sequence memory per GPU | Compute | Needs |
|---|---|---|---|---|
| Layer split (our runner) | weights by layer | unchanged | one GPU at a time | — |
| Weight streaming (block swap) | weights to host RAM | unchanged | copies hide behind long-sequence compute | pinned host RAM |
| DDP | nothing (replicas) | unchanged | N sequences in parallel | model fits one GPU |
| FSDP / ZeRO-3 | weights, grads, optimizer | unchanged | all GPUs | fast interconnect |
| Tensor parallel | each layer's matrices | partly divided | all GPUs | NVLink |
| Expert parallel (MoE) | experts | unchanged | all GPUs | NVLink / InfiniBand |
| Context parallel (ring, Ulysses) | the tokens | **÷ CP degree** | all GPUs | NVLink / InfiniBand; support for the attention type |

For GRPO the largest gain from more GPUs is different: **sample many groups at once** (decoding
is bandwidth- and latency-bound, so 32 groups cost little more than one) and update on several
groups per step. That is a trainer design change, not a hardware setting.

## 3. Measured on the A6000 host

Long-sequence update fits, 2026-10-22: Qwen3.8-27B, recorded draws cut to length
(`diagnostic-tools/long_fit_inputs.py`), synthetic grades, 2-GPU layer split 32/32, no judge.

| Sequence | Settings | Result | Peak per GPU | Update, 4 sequences |
|---|---|---|---|---|
| ~31k | resident weights | passed | 42.1 / 41.6 GiB | 516 s |
| 36k | resident | passed | 44.7 / 44.0 GiB | 666 s |
| 60k | 32 layers streamed | OOM in MLP recompute | 46.1 GiB at failure | — |
| 64k | 64 streamed | passed | 39.7 / 38.7 GiB | 1,203 s |
| 96k | 64 streamed + `tiled_mlp` | passed | 42.7 / 41.4 GiB | 1,987 s |
| 128k | 64 streamed + `tiled_mlp` | OOM in a linear-attention LoRA projection (fp32 input copy, 2.44 GiB) | 45.2 GiB at failure | — |

- Fixed per GPU: ~1.5 GiB plus the resident weight share (25 GiB of the 50 GiB model), or ~3 GiB
  of slot pool + 1.8 GiB staging when all layers stream. Slope 0.52 GiB per 1k tokens untiled,
  0.38 tiled. Host RAM (whole host) peaked at 104 GB (64k), 123 GB (96k), 143 GB (128k); the
  update container is capped at 128 GB.
- Rollout: 47–68 tok/s for 4 concurrent draws (vLLM, tensor parallel 2, PCIe, no NVLink).
- Past 96k the limit is the fp32 LoRA input copy in attention projections: a code change
  (chunked projections or bf16 adapter compute), not a setting.
- nvidia-smi shows *reserved* memory: freed weights stay in PyTorch's cache and are reused, so it
  cannot show what streaming saves. Use the update receipt's `peak_allocated_gib_by_device`.
- For comparison, one H100 80 GB with 32 layers streamed (October 2026): 34.6 GiB at 8.8k →
  61.8 GiB at 52.9k (~0.62 GiB per 1k, ~29 GiB fixed).

## 4. Candidate models (official Hugging Face configs, 2026-10-23)

Ranked open-weights models from [Artificial Analysis](https://artificialanalysis.ai/models/open-source)
(Intelligence Index v4.3) plus our current and small candidates:

| Model | Total / active | BF16 weights | KV per 128k seq | AA index | Licence |
|---|---|---|---|---|---|
| Qwen3.8-27B (current) | 27B dense | 50 GiB | 7.8 GiB | 34 | see model card |
| GLM-4.7-Flash | 30B / 3B | 56 GiB | 6.5 GiB | — | MIT |
| **GLM-5.3-Flash** | 320B / 18B | 596 GiB | 1.3 GiB | 42 | MIT |
| GLM-5.3 | 753B / 40B | 1,403 GiB | 10.7 GiB | 45 | custom, permissive |
| DeepSeek-V4.1-Flash | 552B / 16B | 1,028 GiB | ≤ 4.9 GiB | 45 | MIT |
| MiMo-V2.6-Pro | 1.02T / 42B | 1,900 GiB | 6.1 GiB | 46 | MIT |
| Step-5-Preview | 600B / 27B | 1,118 GiB | 8.4 GiB | 44 | see model card |
| Kimi-K3 | 2.8T / 104B | 5,215 GiB | 3.3 GiB | 44 | see model card |

Shapes, KV notes and sources per model: [`planning-tools/models.json`](../planning-tools/models.json).
GLM-5.3 and GLM-5.3-Flash ship FP8 by default; official `-BF16` repositories exist (GLM-5.3-Flash
BF16: 643 GB). GLM-5.3-Flash is multimodal with hybrid linear + sparse attention and a 1M context.

## 5. Hardware tiers

Tiers differ in workload and batching; each picks the smallest whole-node GPU count that fits BF16
rollout for the batched sequences and the update with tiled MLP (streaming only in pilot; CP when
needed):

| Tier | Groups sampled at once | Weights in update | Updates × groups per update | Groups sampled (≈ half give a reward contrast) |
|---|---|---|---|---|
| pilot | 4 | streaming allowed | 100 × 4 | 800 |
| standard | 16 | resident | 100 × 8 | 1,600 |
| extended | 32 | resident | 200 × 16 | 6,400 |

GPUs and GPU-hours per tier (GPU-hours include a 3× factor over the ideal for judge, weight sync,
reloads and framework overhead; "stream" = weights streamed from host RAM; CP = context-parallel
degree):

**30k prompt + 60k output per draw (medium-effort-like):**

| Model | Tier | H100 80 GB | H200 141 GB | B200 180 GB | B300 288 GB |
|---|---|---|---|---|---|
| GLM-5.3-Flash | pilot | 16 (1,563 h, stream) | 8 (1,122 h) | 8 (656 h) | 8 (656 h) |
| GLM-5.3-Flash | standard | 16 (1,814 h, CP 2) | 8 (1,330 h) | 8 (764 h) | 8 (764 h) |
| GLM-5.3-Flash | extended | 16 (4,559 h, CP 2) | 8 (3,435 h) | 8 (1,926 h) | 8 (1,926 h) |
| GLM-5.3 | pilot | 24 (3,936 h, stream) | 16 (2,817 h, stream) | 16 (1,653 h) | 8 (1,653 h) |
| GLM-5.3 | standard | 32 (4,385 h, CP 2) | 16 (3,201 h, CP 2) | 16 (1,846 h) | 8 (1,846 h) |
| GLM-5.3 | extended | 40 (10,704 h, CP 2) | 24 (8,033 h) | 16 (4,521 h) | 16 (4,521 h) |
| DeepSeek-V4.1-Flash | pilot | 16 (1,631 h, stream) | 16 (1,166 h) | 8 (685 h, stream) | 8 (685 h) |
| DeepSeek-V4.1-Flash | standard | 24 (2,295 h, CP 2) | 16 (1,658 h) | 8 (965 h, CP 2) | 8 (965 h) |
| DeepSeek-V4.1-Flash | extended | 24 (6,470 h, CP 2) | 16 (4,741 h) | 16 (2,725 h) | 8 (2,725 h) |
| MiMo-V2.6-Pro | pilot | 32 (3,908 h, stream) | 16 (2,802 h, stream) | 16 (1,642 h, stream) | 16 (1,642 h) |
| MiMo-V2.6-Pro | standard | 32 (5,029 h, CP 4) | 24 (3,658 h, CP 2) | 16 (2,116 h, CP 2) | 16 (2,116 h) |
| MiMo-V2.6-Pro | extended | 40 (13,330 h, CP 4) | 24 (9,895 h, CP 2) | 16 (5,623 h, CP 2) | 16 (5,623 h) |
| Kimi-K3 | pilot | > 64 | 48 (6,793 h, stream) | 40 (3,979 h, stream) | 24 (3,979 h, stream) |
| Kimi-K3 | standard | > 64 | 48 (9,268 h, CP 4) | 40 (5,366 h, CP 4) | 24 (5,366 h, CP 4) |
| Kimi-K3 | extended | > 64 | 48 (25,900 h, CP 4) | 40 (14,763 h, CP 4) | 24 (14,763 h, CP 4) |

**GLM-5.3-Flash with 30k prompt + 100k output (xhigh-like traces):**

| Tier | H100 | H200 | B200 | B300 |
|---|---|---|---|---|
| pilot | 16 (2,581 h, stream) | 8 (1,847 h, stream) | 8 (1,084 h) | 8 (1,084 h) |
| standard | 16 (2,978 h, CP 2) | 8 (2,170 h, CP 2) | 8 (1,253 h) | 8 (1,253 h) |
| extended | 16 (7,412 h, CP 2) | 8 (5,539 h, CP 2) | 8 (3,129 h) | 8 (3,129 h) |

Reading the tiers: for GLM-5.3-Flash in BF16 a single 8 × B200 or 8 × B300 node covers every tier
with weights resident and no context parallelism; 8 × H200 works but needs streaming or CP for
long traces; H100 needs two nodes. Not included: fit tests, base and checkpoint evaluations,
engineering time, storage and idle time.

## 6. Host RAM

| Case | Need | Notes |
|---|---|---|
| Our host, Qwen3.8-27B to 96k | ~110 GB in the update container | 251 GB is enough; the 128 GB container cap binds near 128k |
| GLM-5.3-Flash, streamed BF16 | ~0.6 TB pinned + ~23 GiB per 90k sequence per GPU | 1 TB per node is enough; H100/H200 nodes usually have 2 TB |
| GLM-5.3 / MiMo-V2.6-Pro, streamed BF16 | 1.4–1.9 TB pinned per node set + checkpoints | 2 TB nodes; prefer resident weights (B200/B300) |
| Kimi-K3, streamed BF16 | > 5 TB pinned | multi-node only |

More host RAM helps only the weights side; it never reduces per-GPU sequence memory, which is the
binding limit for long traces. Pinned memory needs container memory limits and ulimits that allow
it. NVMe offload works for the update but is too slow for rollout.

## 7. Software at scale

- **Our trainer** (Unsloth, one process, layer split, one group per update, model reloaded per
  phase) fits this host and the proof of concept. It cannot shard across nodes, batch groups per
  update or split sequences.
- **Candidate stacks** (not yet verified on GLM-5.3-Flash):
  - [slime](https://github.com/THUDM/slime) (Megatron + SGLang): the RL framework GLM-4.5 to
    GLM-5.2 were trained with; TP, PP, DP, CP and EP. LoRA support not found in its docs.
  - [verl](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html): Megatron or FSDP +
    vLLM/SGLang; LoRA through Megatron-Bridge; MoE expert LoRA not documented.
  - [NeMo RL](https://github.com/nvidia-nemo/rl): LoRA GRPO on DTensor and Megatron backends,
    including a 30B-A3B MoE example on 2 nodes × 8 GPUs.
- **To verify before renting:** LoRA on GLM-5.3-Flash's hybrid linear + sparse attention in the
  trainer; serving that LoRA in vLLM/SGLang (attention-only or per-expert); CP support for linear
  and sparse attention; then one full cycle (generate, grade, update, save, resume) on the node.
- **LoRA scope on MoE:** r 16 on every expert multiplies adapter size (GLM-5.3: ~7.5B parameters);
  attention + shared-expert LoRA keeps it small and is the usual starting point.

## 8. Recommendations

1. Finish the proof of concept on Qwen3.8-27B on our host.
2. Measure GLM-5.3-Flash's base on the benchmark (provider API or a short rental) before buying
   training compute.
3. If it audits well: one-day rental of the target node for a full-cycle fit on the chosen stack,
   then the pilot tier. For GLM-5.3-Flash in BF16 the simplest node is 8 × B200 or 8 × B300.
4. Optional, free: evaluate GLM-4.7-Flash's base on our host (fits to 128k) to try the GLM family.

## Sources

- Model configs and cards: huggingface.co (zai-org, deepseek-ai, XiaomiMiMo, moonshotai,
  stepfun-ai), fetched 2026-10-23; copies in `planning-tools/model-configs/`.
- Ranking: Artificial Analysis open-weights models page and Intelligence Index v4.3.
- GPU specifications: NVIDIA data sheets (memory, HBM bandwidth, dense BF16 throughput).
- Frameworks: linked above; Hugging Face blog "Keep the Tokens Flowing" (RL library survey).
- Block swap and tiled MLP: `unsloth_zoo/block_swap.py`, `unsloth_zoo/tiled_mlp.py` in the pinned
  image (Unsloth 2026.10.1).
