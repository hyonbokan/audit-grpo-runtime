# Planning tools

Estimates GPU memory, time and hardware tiers for LoRA GRPO with long reasoning traces, for any
model in `models.json` on any GPU in `hardware.json`. The reasoning and the results are in
[docs/compute-planning.md](../docs/compute-planning.md). Standalone: stdlib Python, no GPU, no
network.

```bash
cd planning-tools
python3 compute_estimate.py models                     # size, BF16 weights, KV cache
python3 compute_estimate.py fit --model GLM-4.7-Flash --gpu A6000 --gpus 2 --tokens 128000 --tiled --offload
python3 compute_estimate.py tiers --models GLM-5.3-Flash GLM-5.3 --prompt 30000 --output 60000
uvx --with pytest pytest -q -p no:cacheprovider        # tests
```

| File | Contents |
|---|---|
| `compute_estimate.py` | The estimator: update memory per GPU, rollout fit and time, tiers |
| `models.json` | Candidate models: total/active parameters (model card or Artificial Analysis), shape, KV per token, licence, source |
| `hardware.json` | GPU types: usable memory, HBM bandwidth, BF16 throughput, node size, typical host RAM |
| `model-configs/` | The models' Hugging Face `config.json` files the shape fields come from |

How it estimates:
- **Update memory per GPU** = resident weight share (or a slot pool of 8 layers when weights
  stream from host RAM, plus staging) + sequence memory ÷ context-parallel degree + 1.5 GiB.
  Sequence memory is calibrated on the measured A6000 fits (0.52 GiB per 1k tokens untiled, 0.38
  tiled, at hidden 5120, scaled by hidden size) plus an unmeasured MoE term.
- **Rollout** fits when BF16 weights + KV cache of all concurrent sequences ≤ 90% of GPU memory.
  Decode step time = max(6 ms floor, bytes read ÷ (0.5 × HBM bandwidth), compute bound).
- **Update time** = 4 draws × tokens × 6 × active params ÷ (MFU × peak × GPUs); MFU 0.17 measured
  on the A6000 layer split, 0.25 assumed for sharded H100-class training.
- **Tiers** pick the smallest whole-node GPU count that fits; wall time includes a 3× practical
  factor and assumes half the groups give a reward contrast.

The test file checks the calibration against the measured fits. Change a constant only with a
new measurement, and record it in docs/compute-planning.md.
