# Audit GRPO runtime

GRPO fine-tuning of `Qwen/Qwen3.8-27B` (LoRA) for smart-contract audit findings, with
an LLM judge grading each answer against private ground truth. Training runs on a
rented Linux/NVIDIA host; this copy holds the code, the documents and selected evidence.

Rules for changing anything here are in [AGENTS.md](AGENTS.md).

## Layout

| Path | Contents |
|---|---|
| `private/` | Operational state and experiment records (not in git). Start with `private/README.md`. |
| `training-code/` | The maintained code: launchers, controller, native update, judge, evaluation. See its [README](training-code/README.md). |
| `images/` | Dockerfile, locks and build script for the pinned runtime images. See [HOST_SETUP.md](HOST_SETUP.md). |
| `diagnostic-tools/`, `backup-tools/` | Standalone utilities: checkpoint merge and comparison, checkpoint backup. `merge_checkpoint.py` is called by `launch-training.sh`. |
| `archive/` | Finished-experiment code and the original extraction provenance. Read-only. |
| `results/` | Run configs, receipts, grades, model output and reports: dated evidence (not in git). |
| `workspace/` | Frozen backend snapshot used as `/workspace` (not in git). |
| `live-status.sh` | Containers, GPU memory and the newest run's accepted checkpoint. |
| `HOST_SETUP.md` | Preparing a new GPU host: requirements, restore from backup, images, checks. |

## Working on the code

```bash
cd training-code
bash run-tests-local.sh          # workstation: Docker + local deps image, no GPU or paid calls
bash run-tests.sh                # rental: pinned images
ruff check . && ruff format --check .
python3 provenance.py            # after any source or config change
```

Starting a training run, resume behavior and judge credentials are covered in
[training-code/README.md](training-code/README.md). Any GPU or paid work needs the
user's approval and a reviewed schedule (see AGENTS.md).

## Secrets

Credentials never live in the repository. Copy [.env.example](.env.example) to a
restricted file outside it and name that file in the run config (`judge_env_file`,
`hf_env_file`).

## Version control

This directory is a local git repository with no remote; the parent `ai-auditor`
repository ignores `backend/scripts/finetuning/`. [.gitignore](.gitignore) keeps out
secrets, `private/`, `results/`, `workspace/`, model weights and archives, and logs.
Check `git status --ignored` before any push.
