# Preparing a new GPU host

How to bring a Linux/NVIDIA machine to the point where `training-code/` launchers run,
written so an agent can follow it without asking. Host-specific values (backup repository,
paths, progress) live in `private/`; start with `private/README.md`.

Each step ends with **Done when**; do not move on until it passes. Nothing here starts GPU
work or paid judge calls without the user's approval (AGENTS.md).

## Order

1. `bash preflight.sh`: checks everything below, changes nothing, names the step that fixes
   each FAIL. Rerun after every step.
2. Steps 1–3 (layout, credentials, restore) take minutes once downloaded.
3. Start step 4 (base model, ~54 GB) first and do step 5 (images) while it runs. About 70 GB
   is downloaded in all: ~20 min at 1 Gb/s, ~2 h at 100 Mb/s.
4. Steps 6–7: tests, then the GPU fit.

## 0. Requirements

| Need | Why |
|---|---|
| NVIDIA driver ≥ 575 | The images bundle CUDA 12.9; only the driver matters, not a host CUDA toolkit. Drivers run every older CUDA. |
| Docker with the `nvidia` runtime, user in the `docker` group | Every phase runs in a container. No root is needed. |
| RAM ≥ 160 GB, ≥ 250 GB free on the cache volume | Update container cap 128 GB; model, one merge, images, results. |
| `jq`, `uv`, Python ≥ 3.10 with SSL | Launchers, `provenance.py`, downloads. |

Do not change a host's driver to fit the images without the user's approval. For a driver
below 575, re-lock for an older CUDA (step 5) instead.

## 1. Layout and host settings

```
<runtime root>/   this repository; workspace/ (frozen backend, configs), runs/, results/, private/ are git-ignored
<cache>/          outside the repository, mounted as /workspace/cache: huggingface/, wheels/, merges
```

Copy `host.env.example` to `host.env` (git-ignored) and set:
- `AUDIT_GRPO_CACHE`: the absolute cache directory.
- `AUDIT_GRPO_PEER_ACCESS`: `enabled`, or `disabled` when `preflight.sh` finds GPU peer copies
  corrupt (see Known problems).
- `AUDIT_GRPO_HF_REPO`: a private Hugging Face model repository for run backups
  (`launch-segment.sh` refuses to start a run without it).

Launchers exit 2 without the first two. Image tags (`audit-grpo-{deps,vllm}:cu129-v1`) are in code; a
new tag must be changed there.

**Done when** `preflight.sh` passes the host.env lines and `git status` shows no data.

## 2. Credentials

Copy `.env.example` to `.env`, fill it, `chmod 600`. Never print it; load it with
`set -a; . ./.env; set +a` in the command that needs it.

Start the judge container, which holds these credentials:
`bash training-code/start-live.sh`.

**Done when** `preflight.sh` passes the `.env` lines and `docker ps` lists `audit-grpo-live`.

## 3. Restore from the private backup

Download only what is needed (layout in the backup's `README.md`) and verify every file
against `SHA256SUMS.json`; a full bundle also carries `recovery/FILE_SHA256.json`.

| From the bundle (`recovery/...`) | To |
|---|---|
| `frozen-backend/`, `workspace-configs/` | `workspace/code/backend/`, `workspace/configs/` |
| `frozen-corpus/` | `runs/<corpus run>/dataset100/` |
| `experiment/`, record archives | `runs/<run>/`, `runs/...` unchanged |
| `resume/checkpoint-N/` (separate download) | the checkpoint path a run config names |

**Done when** all hashes match, `sha256sum training-code/private/policy_rendering.json` equals
`policy_rendering_sha256` in the corpus `manifest.json`, and `data/private/train.jsonl` exists.

## 4. Base model

Download over plain HTTP: `huggingface_hub` without `hf_xet`, in its own venv (Known problems).

```bash
HF_HOME=<cache>/huggingface huggingface-cli download Qwen/Qwen3.8-27B --revision <model_revision>
cd <cache>/huggingface/hub/models--Qwen--Qwen3.8-27B/blobs   # every blob's name is its SHA-256
for f in *; do [ ${#f} -eq 64 ] && [ "$(sha256sum "$f" | cut -c1-64)" != "$f" ] && echo "BAD $f"; done
```

**Done when** no blob is BAD and `preflight.sh` finds every shard.

## 5. Images

**Load the saved archive** (`images/<suffix>/` in the backup). Loading keeps the image IDs, so
runs stay comparable with earlier hosts; a rebuild is a new environment even with equal pins.

```bash
sha256sum -c images-cu129-v1.tar.zst.sha256 && zstd -dc images-cu129-v1.tar.zst | docker load
cp build.json <runtime root>/runs/images-cu129-v1/   # its IDs must equal docker image inspect
```

**Build** only when no archive fits the driver: `bash images/build.sh <cache>/wheels`. It fetches
every locked wheel to the host first (resumable), then builds offline and hash-locked, checks
each environment against its lock and records IDs in `runs/images-<suffix>/`. Locks:
`gpu-requirements-cu129.lock` (deps: torch, Unsloth, TRL, PEFT) and `vllm-requirements-cu129.lock`
(the same plus vLLM). They hold every non-CUDA pin of the original CUDA 13 locks; re-lock
commands are in `private/`. After a build, archive it for the next host (upload needs approval):
`docker save <both tags> | zstd -T0 -3 > images-<suffix>.tar.zst`.

**Done when** both tags exist and their IDs match `build.json`.

## 6. Tests

```bash
uv pip install --target test-tools --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 pytest pytest-asyncio   # once; not in the images
bash training-code/run-tests.sh
cd training-code && uvx ruff check . && uvx ruff format --check .
```

**Done when** every suite passes. There is no known-failure list.

## 7. GPU fit (needs approval)

Run one complete cycle (update, acceptance) before any training schedule, and record
per-GPU peak memory, wall time and that the adapter changed. Inference fit does not prove
training fit.

Replay a saved visit to avoid judge cost: `launch-cycle.sh --resume` reuses phases with
complete receipts and skips grading when `grading/rewards.json` exists.

1. Pick a visit whose starting checkpoint has full resume state (`optimizer.pt`,
   `scheduler.pt`, `rng_state.pth`).
2. Register a fit-only schedule in `schedules.json` with this host's `update_placement`,
   `tensor_parallel_size`, `offload_layers` and `tiled_mlp`; run `provenance.py`.
3. If the images were rebuilt, copy the run's `source-plan.json` to `runs/<fit>/` and update
   only the `package_versions` that differ (e.g. torch `2.13.0+cu129`), noting why in
   `change_record`. Never edit the original.
4. Copy the visit's `config.json` to `runs/<fit-run>/` and change ids, images, deadline,
   caps, placement and `source_plan`.
5. Run `cycle_contract prepare` (as `launch-cycle.sh` does), copy the visit's `rollout/` and
   `grading/` and `SOURCE_SHA256SUMS` in, then start
   `setsid nohup bash training-code/launch-cycle.sh <run> <run>/config.json --resume &`.
   Keep monitors outside the run directory: `--resume` archives its `*.log`/`*.txt`.

**Long-sequence fit (no judge).** To find how long a sequence the update fits, seed a cycle with
recorded draws cut to a target length (`diagnostic-tools/long_fit_inputs.py`; see its README):
`build` writes `runs/<fit-run>/config.json` and stages rollout and synthetic grades beside it, then
run `cycle_contract prepare` as in step 5, `long_fit_inputs.py seed --run runs/<fit-run>`, and
`launch-cycle.sh --resume`. Fit-run directories must sit directly under `runs/` (containers see
`/results/<name>`). Read `peak_allocated_gib_by_device` from `update/receipt.json`; nvidia-smi shows
reserved memory and cannot show what streaming saves. Results and the A6000 limits:
[docs/compute-planning.md](docs/compute-planning.md).

**Done when** `cycle-result.json` approves all 14 conditions. vLLM's copy of the new adapter is
verified by the next rollout; a fit replay checks it by starting a rollout from the new checkpoint.

## Known problems

- **Corrupt GPU peer copies.** With the IOMMU translating between PCIe root ports, the driver
  reports peer access but copies of freshly computed data arrive wrong (a split update then
  fails with an out-of-range embedding index). `preflight.sh` compares copied data. Set
  `AUDIT_GRPO_PEER_ACCESS=disabled`: launchers preload a shim so CUDA stages copies through
  host memory. Host fix: `iommu=pt` on the kernel command line (root, reboot).
- **`hf_xet` stalls**, and an HTTP resume of a xet partial file gives the right size with the
  wrong content. Use plain HTTP from the start; only the SHA-256 check catches corruption.
- **Stale tools early on PATH** (e.g. a copied `~/bin`: Python without SSL, broken `tmux`).
  `preflight.sh` warns; call `/usr/bin/<tool>` or pass `HOST_PYTHON` to `images/build.sh`.
- **Run outputs are owned by root.** Move a failed attempt aside, or delete it from a
  container: `docker run --rm -v "$PWD/runs:/results" <image> rm -rf /results/<dir>`.
- **NCCL needs a large `/dev/shm`.** Launchers pass `--shm-size 16g`; do the same ad hoc.
- **Copied plans override code defaults.** Check effective values (e.g. `judge_call_retries`)
  in the generated `judge-plan.json`.
