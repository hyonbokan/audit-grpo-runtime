# GRPO training code

Trains a LoRA adapter on `Qwen/Qwen3.8-27B` with GRPO: each visit samples four audits of
one project with vLLM, grades them with an LLM judge against private ground truth, and
applies one native optimizer update when the rewards differ. It runs on a Linux/NVIDIA
host with the pinned Docker images; nothing here trains on a Mac.

Current state, budgets, next steps and the dated experiment history are private notes
in `../private/` (not in git; start with its README).

## Layout

| Path | What it does |
|---|---|
| `launch-training.sh` | Multi-visit run: prepare, one cycle per visit, backups, merge, final evaluation. |
| `launch-cycle.sh` | One cycle: rollout → grading → update → refresh → acceptance. |
| `launch-evaluation.sh` | One matched evaluation (validation panel or native benchmark). |
| `launch-benchmark-suite.sh` | Runs `launch-evaluation.sh` per held-out project and writes the report. |
| `launch-lib.sh` | Shared mounts, environment, deadline timer and live-container exec. |
| `controller.py` | Phases `launch-training.sh` calls: `prepare`, `next`, `record`, `identity`, ... |
| `schedules.py`, `schedules.json` | The reviewed schedules: shared defaults plus per-schedule overrides. |
| `cycle_contract.py`, `cycle_scope.json` | Validates a cycle config and decides acceptance (14 conditions). |
| `panels.py` | Validation panels during training and the final evaluation suite configs. |
| `frozen_requests.py` | Freezes each training row's prompt tokens for a run. |
| `progress.py` | Training order, update/skip decision, checkpoint selection. |
| `hf_backup.py` | Verified private HF backups (thin per visit, full at start and end). |
| `retention.py` | Disk retention: prunes resume state and adapter copies, releases evaluated merges, checks free space. |
| `audited_corpus.py`, `corpus_*.py` | Builds the frozen training corpus from human-audit labels. |
| `private/` | Not in git: the frozen policy rendering (production audit prompt and answer schema) that the corpus builder and reward worker hash. Copy it in before a run; both fail loudly without it. |
| `grpo/` | Native update (`corrected_update`, `corrected_loss`, `native_policy`), judge (`judge_worker`, `reward`, `budget`, `judge_cache`). |
| `inference/` | vLLM rollout, adapter transfer checks, evaluation and benchmark grading. |
| `repeat_evaluation.py`, `inference/heal_samples.py` | Manual follow-ups: fresh-seed repeats, symmetric parser healing. |
| `runtime_io.py` | JSON files, hashes, deadlines and result directories. |
| `provenance.py` | Writes `SOURCE_SHA256SUMS`, which every launcher verifies. |

`grpo/` and `inference/` are on `PYTHONPATH` as `/training-code` and
`/training-code/inference`. The frozen backend (`/workspace/code/backend`) supplies
`llm_router`, `core.schemas` and `scripts.finetuning.tools`. GPU imports stay inside
functions; `native_policy.load_policy` must import `unsloth` first.

Finished-experiment code (TRL pilot trainer, expanded-experiment controller, LLM-reviewed
corpus, sampler probes, extraction provenance) is in `../archive/`, not maintained.

## Tests

```bash
bash training-code/run-tests.sh            # rental: CPU suites in the pinned images
bash training-code/run-tests-local.sh      # workstation: local amd64 deps image under emulation
bash training-code/run-tests.sh judge      # one suite: judge (Python 3.14) or native
bash training-code/test_live_status.sh     # offline check of ../live-status.sh
bash training-code/test_preflight.sh       # offline check of ../preflight.sh driver rule
```

Neither runs GPU work, network calls or paid judging. Every listed test must pass; there
is no known-failure list. `run-tests-local.sh` uses this repository's backend in place of
the frozen one and cannot import vLLM, so it names the vLLM test files it does not run.
Judge tests use the synthetic reward contract and one-row workspace in `testing/judge.py`;
shared config builders are in `testing/` (excluded from `SOURCE_SHA256SUMS`).

## Start a training run

Each host needs `../host.env` (git-ignored; copy `../host.env.example`) naming its cache
directory. The launchers exit 2 without it.

1. Add the schedule to `schedules.json` (`training_cutoff`, `final_deadline`, `training`
   steps and group limits; anything unset takes `defaults`). An unregistered
   `schedule_id` is rejected.
2. Write `results/RUN/config.json` with that `schedule_id`, the starting checkpoint and
   its adapter hash, dataset paths, `hf_repo` and `judge_env_file`.
3. Freeze and launch:

```bash
python3 training-code/provenance.py
cp training-code/SOURCE_SHA256SUMS results/RUN/
bash training-code/launch-training.sh results/RUN results/RUN/config.json
```

Do not edit `training-code/` while a run is live: every visit re-verifies the hashes.

## Resume and failures

- `launch-training.sh RUN CONFIG --resume` continues from `state.json` and resumes an
  unrecorded pending visit. It refuses a finished run.
- `launch-cycle.sh RUN CONFIG --resume` reuses phases with complete receipts and moves the
  failed attempt's logs to `attempts/`.
- `failure_policy` in the run config (defaults:
  `{"visit_attempts": 2, "skip_failed_visits": true, "maximum_consecutive_failed_visits": 2}`)
  resumes a failed visit once, then records it as `failed` and keeps the accepted
  checkpoint. Exit 124 (deadline), 130 and 143 (operator stop) are never retried.
- Checkpoints are written as `checkpoint-N.partial` and renamed when complete. The shared
  judge ledger is replaced atomically under `<ledger>.lock`.

## GPU placement

A schedule's `training` block chooses where each phase puts the model; `cycle_contract`
holds every cycle to it and `launch-cycle.sh` exposes only those GPUs to each container.

- `update_placement` (default `{"gpus": [0], "embedding": "offloaded"}`): one GPU, or a layer
  split such as `{"gpus": [0, 1], "embedding": "gpu", "max_memory_gib": 44}` placed by
  Unsloth's planner (`grpo/placement.py`). A split keeps the embedding on GPU because Unsloth
  declines embedding offload for a dispatched model. `offload_layers` still streams that many
  frozen layers from host RAM (0: none). The update receipt records `layer_devices` and the
  peak memory of every GPU. The planner keeps the final norm on the first GPU and may put the
  head on the last; the update copies the final hidden states to the head's GPU and records
  each such move in `final_hidden_transfers`.
- `AUDIT_GRPO_PEER_ACCESS` in `../host.env`: `disabled` preloads `peer_access/no_peer_access.c`
  (compiled once into `<cache>/peer-access/`) in every container, for hosts whose GPU peer
  copies are corrupt. `../preflight.sh` tells which a host needs.
- `tensor_parallel_size` (default 1): vLLM tensor parallelism for rollout and refresh. With
  more than one rank, each rank reports its LoRA buffers (`adapter_shards`) and the driver
  rebuilds and verifies the full adapter (`adapter_compat.verify_tensor_parallel`).

## Disk retention

Nothing has to be cleaned by hand during a run; the policy is `storage` in `schedules.json`.

- After every recorded visit, `controller prune` deletes the vLLM adapter copies
  (`rollout/` and `refresh/mapped-adapter/`; the phase receipts keep their hashes) and the
  resume-only files (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`) of every accepted
  checkpoint except the newest `resume_states_kept`. Adapters are never deleted. Each
  deletion is appended to the run's `retention.json`.
- Merged models are deleted once evaluated: validation merges in `record-validation`,
  the final merge after the benchmark suite. The `MERGE_MANIFEST.json` is kept in the
  run's `merge-receipts/`; `launch-training.sh` re-merges from the adapter if needed again.
- A failed visit upload discards its local staging bundle; the visit itself stays on disk.
- `controller next` refuses to start or resume a visit when the run, artifact or merge
  disk has less than `minimum_free_gib` free. Free space, then `--resume`.

## Judge calls

- Transport errors (`TransientProviderError`, `RateLimitError`) retry with 30/60/120 s
  backoff, `judge_call_retries` times (0-5, default 3). Budget exhaustion stops grading.
- Paid responses are cached per visit in `judge-cache/`, so a retried or resumed group
  pays only for calls that never completed.
- The live container (`audit-grpo-live`) holds the credentials and the network access;
  start or reuse it with `bash training-code/start-live.sh`. Every other container runs
  without network.
- `judge_env_file` names a dotenv file inside the live container. It must set
  `LITELLM_GATEWAY_URL` and `LITELLM_GATEWAY_API_KEY`; grading stops if either is missing.
  Keep credentials out of source, configs and logs.

## Evaluation

`launch-evaluation.sh` picks the module from the config's `evaluation_kind`:
`grpo_held_out_panel` → `inference/panel_evaluation.py`,
`native_solidity_benchmark` → `inference/benchmark_evaluation.py`, otherwise
`inference/evaluation.py`. Invalid or truncated answers stay unknown, never zero.
Healed (parser-recovered) results are a separate protocol and are never pooled with strict ones.

The current benchmark is the free-text suite (`benchmark_suite: solidity_free_text`,
`inference/benchmark_free_text.py`): the prompt sets no output format; one judge call extracts
the vulnerabilities a report asserts (kept in `<arm>-grading/extraction/`), and the reviewed
one-to-one matcher grades them against the ground truth. A draw that stops inside its
reasoning is graded on that reasoning (`answer_source: unclosed_reasoning`); a draw cut off at
the context limit stays `truncated`. The base arm comes first; a trained arm is compared
through its merged weights. Reasoning effort is `medium` or `xhigh` (the pinned template has no `high`), four draws per arm.
