# Preparing a new GPU host

How to bring a fresh Linux/NVIDIA machine to the point where the launchers in
`training-code/` can run. Written so that an agent can follow it without asking.
Host-specific values (backup repository, paths, progress on the current host) live in
`private/`; start with `private/README.md`.

Every step ends with a **Done when** check. Do not move on until it passes. Nothing here
starts GPU work or paid judge calls; those need the user's approval (see AGENTS.md).

## 0. Requirements

| Need | Why | Check |
|---|---|---|
| NVIDIA driver ≥ 575 | The image locks in `images/` use CUDA 12.9. (The original CUDA 13 locks needed ≥ 580.) | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| Docker with the `nvidia` runtime, user in `docker` group | Every phase runs in a container. | `docker info \| grep -i runtimes` lists `nvidia` |
| Passwordless `sudo systemd-run` / `systemctl` | `launch-lib.sh` starts a transient systemd timer that stops containers at the deadline. | `sudo -n true` |
| Host RAM ≥ 160 GB | The update container is capped at 128 GB and frozen layers are offloaded to host memory. | `free -g` |
| ≥ 250 GB free on the cache volume | Base model (~54 GB), one temporary merged model (~54 GB), images (~40 GB), run results. | `df -h` |
| `jq`, `uv`, Python ≥ 3.10 with working SSL on the host | Launchers, `provenance.py`, downloads. If the first `python3` on PATH cannot `import ssl`, pass a working one as `HOST_PYTHON` to `images/build.sh`. | `which jq uv python3`; `python3 -c 'import ssl'` |
| Network link ≥ 1 Gb/s | About 70 GB is downloaded (model, wheels, backups). At 100 Mb/s that takes hours. | `cat /sys/class/net/<interface>/speed` prints `1000` or more |

A GPU with less than about 60 GB cannot hold the BF16 base for vLLM rollout on its own;
see section 7.

## 1. Layout

```
<runtime root>/            this repository
  workspace/               frozen backend and configs, mounted as /workspace (not in git)
    code/backend/
    configs/
  results/                 runs, corpus, evaluation assets (not in git)
  private/                 notes (not in git)
<cache>/                   large, outside the repository; mounted as /workspace/cache
  huggingface/             HF_HOME: base model snapshot
  wheels/                  locked wheels for offline image builds (~15 GB)
  weekend-merges/          temporary merged models
<restore staging>/         downloaded backups; delete after the restore is verified
```

Copy `host.env.example` to `host.env` in the runtime root (git-ignored) and set
`AUDIT_GRPO_CACHE` to the absolute cache directory. Every launcher sources it through
`launch-lib.sh` and exits 2 if it is missing; `run-tests.sh` also reads it to mount the
cache read-only for the tokenizer test. The frozen workspace is always
`<runtime root>/workspace`. The image tags `audit-grpo-deps:cu129-v1` and
`audit-grpo-vllm:cu129-v1` are in code (`launch-lib.sh`, `run-tests.sh`,
`evaluation_scope.json`, `benchmark_scope.json` and the fixtures); a rebuild under a new
tag suffix must update them there.

**Done when** `workspace/`, `results/` and the cache directory exist, `host.env` names the
cache, and `git status` shows none of them.

## 2. Credentials

Copy `.env.example` to a file that git ignores (for example `.env` in the runtime root),
fill in the three values and `chmod 600` it. Never print it; load it with
`set -a; . ./.env; set +a` in the command that needs it. The judge reads its file inside
the live container at the path named by `judge_env_file` in the run config.

**Done when** `stat -c %a .env` prints `600` and each key is non-empty (check names only).

## 3. Restore from the private backup

The backup repository is a private HF model repo (name in `private/`). Download only
what is needed:

| Backup path | Contents |
|---|---|
| `SHA256SUMS.json`, `README.md` | File hashes and layout of the repository |
| `runs/<run>/resume/checkpoint-N/` | The only resumable state: adapter, optimizer, scheduler, RNG |
| `weekend/<run>/training-final/recovery.tar.gz` | Newest full bundle: frozen corpus, backend, configs, ledgers, run records |
| `runs/<run>/records/*.tar.gz` | Visit records and evaluations (evidence) |
| `legacy/evaluations/full-context-progress-*.tar.gz` | Held-out evaluation template and assets |

Verify every download against `SHA256SUMS.json`. A full bundle carries its own
`recovery/FILE_SHA256.json`; verify the extracted files against it too. Then restore:

| From the bundle (`recovery/...`) | To the runtime |
|---|---|
| `frozen-backend/` | `workspace/code/backend/` |
| `workspace-configs/` | `workspace/configs/` |
| `frozen-corpus/` | `results/<corpus run>/dataset100/` (the run config's `dataset_host`) |
| `experiment/` | `results/<run>/` |
| `judge-budget-<phase>.json` | the ledger paths named in the run config |
| `resume/checkpoint-N/` (separate download) | the path in the next run's `initial_checkpoint` |
| `results/...` inside record and evaluation archives | `results/...` unchanged |

**Done when** all hashes match, the checkpoint's `adapter_model.safetensors` hash equals
the run config's adapter hash, `sha256sum training-code/private/policy_rendering.json`
equals `policy_rendering_sha256` in the corpus `manifest.json`, and
`data/private/train.jsonl` exists in the corpus.

## 4. Base model

```bash
HF_HOME=<cache>/huggingface huggingface-cli download Qwen/Qwen3.8-27B --revision <pinned revision>
```

The pinned revision is `model_revision` in `fixtures/cycle.json` and every run config.

**Done when** `<cache>/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/<revision>/`
holds all shards listed in `model.safetensors.index.json` **and** every 64-character blob
in `.../blobs/` has a SHA-256 equal to its file name:

```bash
cd <cache>/huggingface/hub/models--Qwen--Qwen3.8-27B/blobs
for f in *; do [ ${#f} -eq 64 ] && [ "$(sha256sum "$f" | cut -c1-64)" != "$f" ] && echo "BAD $f"; done
```

## 5. Images

```bash
bash images/build.sh <cache>/wheels   # tags audit-grpo-deps:cu129-v1 and audit-grpo-vllm:cu129-v1
```

`build.sh` first runs `images/fetch_wheels.py`, which downloads every locked wheel once
into the wheel directory (hash-checked; rerun to resume). The Docker build then installs
offline from that directory, so a rebuild needs no package servers. Only the Python
interpreters and the judge's tokenizer encoding are fetched during the build.

`images/Dockerfile` builds two targets that share one base with the Python 3.14 judge
environment (`judge-requirements.lock`, unchanged from the original):

| Target | `/opt/pilot/gpu` (Python 3.12) | Used for |
|---|---|---|
| `deps` | `gpu-requirements-cu129.lock`: torch, Unsloth, TRL, PEFT, Transformers | Native update, controller, CPU phases |
| `vllm` | `vllm-requirements-cu129.lock`: the same plus vLLM 0.31.0+cu129 | Rollout, refresh, evaluation, native test suite |

Each environment is installed hash-locked and binary-only, then checked by
`images/dependency_check.py`: the installed distributions must equal the lock exactly.
`build.sh` writes image IDs and a `uv pip freeze` per environment to
`results/images-<suffix>/`.

The locks target **CUDA 12.9**, which needs driver ≥ 575 and supports Ampere and newer.
They were derived from the original CUDA 13 locks frozen in `workspace/configs/`
(`dependencies.Dockerfile`, `gpu-requirements.lock`). Every non-CUDA pin is held at its
original version; only torch/torchvision builds, `cuda-*` and `nvidia-*` wheels differ. To
re-lock (for example for a different CUDA version), compile `gpu-requirements.in` with the
original lock minus CUDA packages as constraints, then `vllm-requirements.in` with the new
GPU lock as constraints; see `private/` for the exact commands used.

A rebuilt image is a new environment, even with the same pins. Cite the image IDs from
`results/images-<suffix>/build.json` in run records, and never pool results with runs on
other images without saying so.

**Done when** both tags exist, `build.json` is written, and
`docker run --rm --gpus all audit-grpo-deps:cu129-v1 python -c "import torch; print(torch.cuda.device_count())"`
prints the host's GPU count.

## 6. Tests

```bash
# pytest is not in the pinned images; install it once into the git-ignored test-tools/.
uv pip install --target test-tools --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 pytest pytest-asyncio
bash training-code/run-tests.sh
cd training-code && uvx ruff check . && uvx ruff format --check .
```

**Done when** every suite passes. There is no known-failure list.

## 7. GPU fit (needs approval)

Before any training schedule, run one complete cycle on the host without paid judging:
rollout, update on saved samples, refresh, acceptance. Record peak memory on every GPU,
host RAM, wall time, and that the adapter weights changed. Inference fit does not prove
training fit.

## Known problems

- **`hf_xet` downloads stall or fail** (`ChunkCache ... No such file or directory`,
  `IncompleteBody`), or report the full size and never exit. Download with
  `huggingface_hub` installed without `hf_xet` in a separate venv (plain HTTP) from the
  start, and verify hashes afterwards.
- **Never resume an `hf_xet` partial download over HTTP.** Xet writes `.incomplete`
  files out of order; an HTTP resume appends to them and produces files with the right
  size and the wrong content. If a xet download was interrupted, delete its
  `*.incomplete` blobs (and any blob written after the switch) before downloading again.
  Size checks do not catch this; only the SHA-256 check does.
- **Slow package servers time out inside `docker build`** (seen with `pypi.nvidia.com`
  at under 1 MB/s). That is why wheels are fetched on the host first with long timeouts
  and many retries; a failed fetch resumes where it stopped.
- **A model download can look finished but be incomplete.** Check that every shard named
  in `model.safetensors.index.json` exists, that no `*.incomplete` blob remains, and that
  each blob's SHA-256 equals its file name (HF stores LFS files by content hash).
- **CUDA version newer than the driver** fails at the first CUDA call. Check
  `nvidia-smi` (top right shows the highest CUDA version the driver supports) against
  the lock's `nvidia-cuda-runtime` version before building.
- **Copied plans override code defaults.** Check effective values such as
  `judge_call_retries` in the generated `judge-plan.json`, not in the source.
