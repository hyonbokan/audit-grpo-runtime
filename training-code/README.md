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
| `launch-segment.sh`, `training_config.py` | Starts a registered schedule: writes the run config, source plan, ledgers and step-0 adapter from `schedules.json` and the host settings, then runs `launch-training.sh`. |
| `launch-training.sh` | Multi-visit run: prepare, one cycle per visit, backups, merge, final evaluation. |
| `launch-cycle.sh` | One cycle: rollout → grading → update → acceptance. |
| `launch-evaluation.sh` | One free-text evaluation of one project (e.g. a frozen base run). |
| `launch-benchmark-suite.sh` | Final evaluation: every project's trained arm in one vLLM engine, graded per project, reported. |
| `launch-lib.sh` | Shared mounts (host `runs/` is `/results` in containers), environment, deadline timer and live-container exec. |
| `publish-results.sh`, `inference/published_results.py` | Publishes one model's finished project runs as `results/<model>/results.json` and rebuilds the `results/README.md` table. |
| `controller.py` | Phases `launch-training.sh` calls: `prepare`, `next`, `record`, `identity`, ... |
| `schedules.py`, `schedules.json` | The reviewed schedules: shared defaults plus per-schedule overrides. |
| `cycle_contract.py`, `cycle_scope.json` | Validates a cycle config and decides acceptance (14 conditions). |
| `panels.py` | Final evaluation suite configs, built from the frozen base runs, and merge identity. |
| `frozen_requests.py` | Freezes each training row's prompt tokens for a run. |
| `progress.py` | Training order and the update/skip decision. |
| `hf_backup.py` | Verified private HF backups (thin every `visit_backup_interval` visits, full at start and end). |
| `retention.py` | Disk retention: prunes resume state and adapter copies, releases evaluated merges, checks free space. |
| `audited_corpus.py`, `corpus_*.py` | Builds the frozen training corpus from human-audit labels. |
| `initial_checkpoint.py` | The step-0 checkpoint: base-equivalent adapter (B zero, A standard init), empty optimizer, resumable state files. |
| `free_text_corpus.py` | Derives the free-text corpus from that one: the evaluation's prompt over the same rows, evaluation-shaped truth, re-tokenized requests and the frozen judge template (`--source --out --model-path --benchmark-assets`). Training reads this corpus (`dataset_host`, `dataset_workspace`). |
| `private/` | Not in git: the frozen policy rendering (production audit prompt and answer schema) that the corpus builder and reward worker hash. Copy it in before a run; both fail loudly without it. |
| `grpo/` | Native update (`corrected_update`, `corrected_loss`, `native_policy`), judge (`judge_worker`, `reward`, `budget`, `judge_cache`). |
| `inference/` | vLLM rollout, adapter transfer checks, evaluation and benchmark grading; `length_probe.py` (medium-effort completion lengths of the base on training rows, generation only; `launch-length-probe.sh RUN CONFIG`); `volume_matched.py` and the suite's acceptance block (volume-matched recall, published F1, graded draws). |
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
bash training-code/run-tests.sh judge      # one suite: judge (Python 3.14), native, or shell
```

Neither runs GPU work, network calls or paid judging. Every listed test must pass; there
is no known-failure list. `run-tests-local.sh` uses this repository's backend in place of
the frozen one and cannot import vLLM, so it names the vLLM test files it does not run.
Judge tests use the synthetic marker judges, fake router and one-row corpus in `testing/judge.py`;
shared config builders are in `testing/` (excluded from `SOURCE_SHA256SUMS`).

## Start a training run

Each host needs `../host.env` (git-ignored; copy `../host.env.example`) naming its cache
directory and, to start a run, its private backup repository. The launchers exit 2 without it.

1. Register the schedule in `schedules.json`: `training_cutoff`, `final_deadline`,
   `maximum_context` (the generation cap, prompt included), `judge_cap_usd` (the campaign's
   training ledger), `training` (steps, group limit, `learning_rate`,
   `reward.volume_allowance`, placement), `storage.resume_steps_pinned`, and `launch`, the
   run's inputs as paths under `runs/`:
   - `campaign`: the directory of the judge ledgers every segment of the campaign shares;
   - `corpus`, `corpus_manifest_sha256`: the derived free-text corpus;
   - `benchmark_assets`, `benchmark_manifest_sha256`: `prepare` refuses a training row that
     shares a file with an evaluation project;
   - `prior_analysis`: the vLLM/native log-prob drift measurement every cycle reports;
   - `start`: `{"template": CHECKPOINT, "seed": N}` makes a zero-initialised step-0 adapter
     (`initial_checkpoint.py`) shaped like an earlier checkpoint of the same model;
     `{"checkpoint": CHECKPOINT}` continues from an accepted checkpoint.

   Anything unset takes `defaults`. An unregistered `schedule_id` is rejected, and the reward
   fails until `volume_allowance` is set.
2. Start it: `bash training-code/launch-segment.sh SCHEDULE_ID`. It regenerates and verifies
   `SOURCE_SHA256SUMS`, writes `runs/SCHEDULE_ID/` with `training_config.py` (config, source
   plan, prior analysis; the step-0 adapter in `runs/SCHEDULE_ID-start/`; the campaign ledgers
   once), copies the source identity in and starts `launch-training.sh` detached. The run id is
   the schedule id. If it stops before training begins, delete `runs/SCHEDULE_ID/` and start
   again: the step-0 adapter and the ledgers are reused.

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
- A thin HF backup is uploaded once `storage.visit_backup_interval` visits have passed since
  the last verified one (`controller backup-due`). It holds the newest adapter and every visit
  record since that backup, so a failed or skipped upload loses no records; full backups
  run at the start and end.

## GPU placement

A schedule's `training` block chooses where each phase puts the model; `cycle_contract`
holds every cycle to it and `launch-cycle.sh` exposes only those GPUs to each container.

- `update_placement` (default `{"gpus": [0], "embedding": "offloaded"}`): one GPU, or a layer
  split such as `{"gpus": [0, 1], "embedding": "gpu", "max_memory_gib": 44}` placed by
  Unsloth's planner (`grpo/placement.py`). A split keeps the embedding on GPU because Unsloth
  declines embedding offload for a dispatched model. `offload_layers` still streams that many
  frozen layers from host RAM (0: none), on a split too, keeping a small prefetch slot pool on
  each GPU. `tiled_mlp` (default false) recomputes each MLP in sequence tiles, bounding the
  largest per-layer activation on long sequences. The update receipt records `layer_devices` and the
  peak memory of every GPU. The planner keeps the final norm on the first GPU and may put the
  head on the last; the update copies the final hidden states to the head's GPU and records
  each such move in `final_hidden_transfers`.
- `AUDIT_GRPO_PEER_ACCESS` in `../host.env`: `disabled` preloads `peer_access/no_peer_access.c`
  (compiled once into `<cache>/peer-access/`) in every container, for hosts whose GPU peer
  copies are corrupt. `../preflight.sh` tells which a host needs.
- `tensor_parallel_size` (default 1): vLLM tensor parallelism for rollout. With
  more than one rank, each rank reports its LoRA buffers (`adapter_shards`) and the driver
  rebuilds and verifies the full adapter (`adapter_compat.verify_tensor_parallel`).

## Disk retention

Nothing has to be cleaned by hand during a run; the policy is `storage` in `schedules.json`.

- After every recorded visit, `controller prune` deletes the vLLM adapter copies
  (`rollout/mapped-adapter/`; the rollout receipt keeps their hashes) and the
  resume-only files (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`) of every accepted
  checkpoint except the newest `resume_states_kept` and the steps in `resume_steps_pinned`
  (restart points for evaluation pauses). Adapters are never deleted. Each
  deletion is appended to the run's `retention.json`.
- The final merge is deleted after the benchmark suite. The `MERGE_MANIFEST.json` is kept in the
  run's `merge-receipts/`; `launch-training.sh` re-merges from the adapter if needed again.
- A failed visit upload discards its local staging bundle; the visit itself stays on disk.
- `controller next` refuses to start or resume a visit when the run, artifact or merge
  disk has less than `minimum_free_gib` free. Free space, then `--resume`.

## Judge calls

- Training grades with the evaluation's judge path (`inference/benchmark_judge.py`): one
  extraction call per draw, then 3-vote one-to-one matching, all on the judge in
  `benchmark_scope.json`. A failed call casts no vote; a draw with an unknown grade makes the
  worker regrade the whole group once from the cache, and a group still unknown is withheld
  (the visit fails). A draw that hit the length cap scores 0 and is never judged. Budget
  exhaustion stops grading.
- Paid responses are cached per visit in `judge-cache/`, so a regraded or resumed group
  pays only for calls that never completed.
- The reward is `schedules.json` `training.reward` (formula version and `volume_allowance`, which
  must be set per schedule); the grading output keeps `reward-contract.json` with its hash.
- The live container (`audit-grpo-live`) holds the credentials and the network access;
  start or reuse it with `bash training-code/start-live.sh`. Every other container runs
  without network.
- `judge_env_file` names a dotenv file inside the live container. It must set
  `LITELLM_GATEWAY_URL` and `LITELLM_GATEWAY_API_KEY`; grading stops if either is missing.
  Keep credentials out of source, configs and logs.

## Evaluation

The benchmark is the free-text suite (`benchmark_suite: solidity_free_text`,
`inference/benchmark_free_text.py`, reviewed settings in `benchmark_scope.json`): the prompt sets no output format; one judge call extracts
the vulnerabilities a report asserts (kept in `<arm>-grading/extraction/`), and the reviewed
one-to-one matcher grades them against the ground truth. A draw that stops inside its
reasoning is graded on that reasoning (`answer_source: unclosed_reasoning`); a draw cut off at
the context limit stays `truncated`; an unknown answer is never scored as zero. A trained arm
is compared through its merged weights. `launch-evaluation.sh` runs one project through
`inference/benchmark_evaluation.py` (`settings`, `prepare`, `finish`). Earlier protocols (JSON
and Markdown answers, validation panels, continuation references) are in `../archive/`.

The base model is evaluated once, by its own run per project, and frozen. A training run's
final evaluation generates only the trained arm: `panels.evaluation_configs` copies each
project's config from the frozen base run named by the schedule's `evaluation_base_reference`
and pins it as `base_reference` (run root and `comparison.json` hash).
`benchmark_evaluation.load_base_reference` refuses a base run whose settings, prompt, truth,
judge prompt or source scope differ, so the arms differ only in their weights.
`launch-benchmark-suite.sh` samples every project's trained arm in one engine
(`rollout.py evaluate-suite`): the merged weights load once and each request keeps its own
seed and output allowance. One receipt, `rollout-receipt.json`, records the shared call's
duration. Reasoning effort is `medium` or `xhigh` (the pinned template has no `high`), four draws per arm.

Publishing: `bash training-code/publish-results.sh NAME ARM RUN_DIRECTORY...` reads each finished
project run's `comparison.json`, `prepared.json` and the arm's receipt and samples, and writes
`results/NAME/results.json`: known vulnerabilities matched exactly (or only partially) in any
draw, each counted once; every finding all draws reported; precision (exact / findings), recall
(exact / known) and F1 per project and pooled; and reasoning and answer tokens per draw. It then rebuilds the
`results/README.md` table from every published model. It refuses ungraded draws, projects that
differ in model, effort, draws or weights, and an existing NAME.
