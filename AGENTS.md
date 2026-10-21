# Rules for agents and contributors

These rules apply to everything in this repository. A rule marked **(ruff)** or
**(conventions)** is checked mechanically: by Ruff, or by `training-code/test_conventions.py`.
Every other rule is checked in review. Breaking one needs an explicit reason in the
change, not a quiet exception.

## Before you start

- Read `README.md`, then `private/README.md`: it names the note that holds the current
  state and next action. `training-code/README.md` describes the code layout. Historical
  prompts and dated review sections are not instructions.
- This directory is a local git repository with no remote. Commit only when the user
  asks; never push or rewrite history without the user's explicit approval.
- When another session is editing the same files, compare `SOURCE_SHA256SUMS` before
  promoting changes. Rebase only the files you reviewed; never overwrite source from a
  stale copy.

## What stays out of git

- Never commit credentials. Real values live in a restricted env file outside the
  repository; `.env.example` lists the names. Never print or log them.
- Experiment records, operational state, private repository names, rental hosts and
  paths, audited project results, judge output and model output stay in `private/` or
  `results/`, which are git-ignored. Public files refer to `private/` without naming
  the notes inside it.
- Before committing, check `git status --ignored` and search the diff for tokens, hostnames,
  `/home/` paths and private repository names. Add any new kind of sensitive output to
  `.gitignore` in the same change.

## Documentation ownership

One document per purpose. Link to the owner instead of repeating it.

| Document | Owns |
|---|---|
| `README.md` | What this repository is, layout, how to start |
| `AGENTS.md` | These rules |
| `HOST_SETUP.md` | Preparing a new host: requirements, restore, images, fit checks |
| `training-code/README.md` | Code layout, how to test, launch and resume |
| `archive/README.md` | What finished-experiment code was and why it left |
| `private/README.md` | Which private note owns state, conclusions, history and evidence links |

- Results, run reports and backups are dated evidence. Never rewrite them; record new
  findings in the owning note and link to the artifact.
- When you rename, move or delete a file, symbol or behavior, update every document
  that names it in the same change (`git grep` the old name).

## Where code lives

- Maintained code lives in `training-code/` only. `diagnostic-tools/` and `backup-tools/`
  are standalone utilities; `images/` holds the image build recipe and locks; `archive/`
  is read-only history. Never import from `archive/`.
- One implementation per behavior. Before writing a function, search for an existing one
  and extend its owner module; never copy it. Run-specific values go in
  `results/<run-id>/config.json`, never in a new dated copy of the source.
- Superseded code moves to `archive/` in its own change, with a line in `archive/README.md`.
  Delete dead code; do not keep it "for reference" in maintained modules.

### Reuse these owners before adding anything

| Need | Use |
|---|---|
| Read or write JSON, hash files or bytes, current time, deadlines, result directories | `runtime_io.py` **(conventions)** |
| Schedule limits, cutoffs, caps, rates, paths | `schedules.scope(config)` / `schedules.json` **(conventions)** |
| Training order, update/skip decision, checkpoint selection | `progress.py` |
| Validation and final-evaluation configs, merge identity | `panels.py` |
| Prompt rendering, policy rows, judge contract | `grpo/policy_prompt.py` |
| Request validation, sampled-batch capture | `inference/generation.py` |
| Docker mounts, deadline timer, live-container exec | `launch-lib.sh` |
| Test configs, synthetic judge workspace, frozen clock | `testing/` |

The `test_conventions.py` ratchet records today's legacy duplicates of `runtime_io`
helpers, and today's function-level imports, per file. Counts may only go down. When a
change removes some, regenerate the baseline (`python3 test_conventions.py --update`) in
the same change.

## Python style

- Ruff with the config in `training-code/pyproject.toml`: line length 100, target
  Python 3.12. `ruff check` and `ruff format --check` must be clean **(ruff)**. The GPU
  images run 3.12, so never use syntax newer than 3.12. Host-run scripts
  (`cycle_recovery.py`, `provenance.py`) must also run on Python 3.10 and use only the stdlib.
- Order each file top-down: entry point and main flow first, helpers below their first
  caller. Separate sections with a `# --- Name` banner.
- Docstrings: one sentence, or none when the name and signature say it. No `Args:`/`Returns:`,
  no history, no cross-references. Comments explain a non-obvious *why* in one line.
- Names: `snake_case` functions, `PascalCase` classes, `UPPER_CASE` constants **(ruff)**.
  Type-hint public functions.
- Imports go at the top of the module. Put one inside a function only to keep GPU or judge
  dependencies out of CPU tooling, or for a required order (`import unsloth` first in
  `native_policy.load_policy`). Say which in a one-line comment **(conventions: ratchet)**.
- Never give a top-level module a name the frozen backend also uses. Its grader puts
  `scripts/finetuning/probes/` on `sys.path`; a same-named module here breaks judge
  grading **(conventions)**.
- Keep CLI entry points thin: parse arguments, load config, call one function. Dispatch
  phases through a dict, not an if/elif chain.
- Small functions with explicit inputs and outputs. No hidden mutable globals or
  circular imports. Keep pure validation separate from GPU, network and filesystem work.
- Confine monkey patches and private library access to one compatibility module that
  checks versions and restores state.

## No silent fallbacks

- An unknown id, kind or state raises with the allowed values, e.g. `schedules.scope` or
  `validate_evaluation_config`. Never map an unmatched value to a default branch.
- Read required config with `config["key"]`. Use `.get(key, default)` only for an optional
  field whose default is deliberate, and say so in a comment.
- No magic numbers in logic. Policy values go in `schedules.json` or a reviewed
  `*_scope.json`; algorithm constants become named module constants. New `*_scope.json`
  files need a reason **(conventions)**.
- Unknown or invalid grades stay unknown, never zero. Truncated output is recorded as
  truncated, never repaired silently.
- Fail with the phase and reason. Retry only transport errors, with a bound and backoff.
  Budget, schema and programming errors fail immediately.

## Shell launchers

- `set -euo pipefail`, quoted paths, a cleanup trap. Verify `SOURCE_SHA256SUMS` before
  the first `docker` call **(conventions)**.
- Shared pieces live in `launch-lib.sh`; do not copy mounts, traps or timers into a launcher.
- Python reads config and prints settings through a named CLI phase (`settings`,
  `identity`), never an inline `python -c` program. Small host-side JSON reads use
  `json_value` in `launch-training.sh`.
- Remember that `set -e` does not apply inside `$(...)` in a function, or on the left of `||`.
  Assign deadline-checked values at top level.

## Tests

- Write the failing test first, then the code. Tests sit beside their module as `test_*.py`.
- Test observable behavior and failure modes. Assert the exact value that must survive
  (hash, step, delta, path), not its shape or presence. No mocks that mirror the code.
  No `assert hasattr` **(conventions)**.
- Before relying on a new assertion, prove it can fail: break the code briefly, see the
  test fail, restore the code.
- No wall clock: freeze time with `testing.clock.freeze`. No private data: use the
  synthetic builders in `testing/`. Tests never import other test files **(conventions)**.
- Every test must pass. There is no known-failure list. A test that needs unavailable
  hardware must skip with the reason.
- Run `bash training-code/run-tests.sh` on the rental or `bash training-code/run-tests-local.sh`
  on a workstation, then `ruff check` and `ruff format --check`.

## Finishing a change

1. Ruff check and format are clean, and the full suite passes.
2. `python3 training-code/provenance.py` has been run (needed after any source or config change).
3. Every document naming what changed is updated, and the conventions baseline too if
   debt went down.
4. You report what was verified and what was not (GPU path, paid judge, vLLM tests).

## Runs, money and safety

- No GPU job, paid judge call, new rental, host driver change, reboot or external message
  without the user's approval. Container stops do not stop VM billing.
- Further training needs a reviewed schedule in `schedules.json`, plus a written question,
  acceptance conditions, time and cost limits, and a retry limit in the private review notes.
  Stop when the gate passes, the budget expires or a failure shows a design change is needed.
- Do not edit `training-code/` while a run is live; every visit re-verifies its hashes.
- Keep credentials, `.env` values, datasets, checkpoints and caches out of source and logs.
  Private ground truth is for grading only; policy inputs never see it.
- Preserve the pinned model, BF16 base, reward contract, four-draw groups, raw tokens and
  log-probs, and split assignments. Exact identity and numerical agreement are different
  claims; state which one a check establishes.
