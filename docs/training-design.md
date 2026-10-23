# Training design: free-text GRPO proof of concept

The design of the next training run: reward, prompt, reasoning effort and length, learning rate,
schedule, acceptance and budget rules. Current version 2026-10-23. Numbers measured on the
benchmark draws and the training corpus are experiment records and stay in `private/`; this
document states what they showed. Hardware limits are in [compute-planning.md](compute-planning.md).

## Decided

| Decision | Date |
|---|---|
| Free-text output; no format in the prompt or the reward | 2026-10-22 |
| Fresh LoRA from the base model | 2026-10-21 |
| One reasoning effort for training and evaluation: **medium** (xhigh traces do not fit the update on the A6000 host) | 2026-10-22 |
| Learning rate from measured adapter movement and a probe | 2026-10-22 |
| No hard time stop: training in segments with evaluation pauses | 2026-10-22 |
| Judge spend under a ledger cap with headroom | 2026-10-22 |
| Update memory: all layers streamed (`offload_layers` 64) + `tiled_mlp` on the 2-GPU split | 2026-10-22 |

## Open decisions (recommendation first)

| # | Decision | Recommendation |
|---|---|---|
| O1 | Volume allowance N0 | The report length at which the base's per-draw F1 peaks, measured on the medium base run |
| O2 | Truncated draw | Reward 0, recorded `truncated` (an explicit exception to "truncated stays unknown") |
| O3 | Partial matches | Exact matches only |
| O4 | Continue rule at pauses | Continue while a pause improves by ≥ 0.03 or the reward trend still rises |
| O5 | Judge ledger cap | $150 for the campaign |

## 1. Prompt and grading: train like we evaluate

- **Prompt:** one user message, `benchmark_free_text.audit_prompt(sources)` with the evaluation's
  file rendering, shared by both paths; no system prompt and no answer format. The training rows
  are re-rendered into a new derived corpus (same rows, splits and labels). The previous training
  prompt added a production system prompt and a JSON answer contract; both go.
- **Grading:** the evaluation's path unchanged: one judge call extracts the vulnerabilities the
  report asserts, then the one-to-one matcher grades them against the known findings (2-of-3
  vote, batches of 30). Training labels are mapped to the evaluation's truth shape.
- **Removed from the reward:** JSON parsing and fence stripping, the −2 format floor, the
  32-finding cap and the reasoning-quality term (evaluation does not grade reasoning).
- A new reward contract version, hashed into every visit.

## 2. Reward: no recall gains from listing more

`R = (exact matches / known) × min(1, N0 / N)`, N = findings the judge extracted.

- Up to N0 findings the reward is plain recall. Beyond N0 a finding pays only if it is at least
  as likely to match as the report's average finding, so padding lowers the reward.
- An empty report scores 0. Exact matches only; partial matches are logged.
- No severity filter (it would reward relabelling findings as High, and the evaluation counts all
  severities). No deduplication step: the one-to-one matcher already gives a duplicate nothing,
  and N counts it.

Why this form: the base model pads its reports, and listing more barely finds more. Training rows
mostly have one known finding, so an unmatched finding is often a real but unlabelled bug. On the
frozen base draws, an additive per-finding cost and per-draw F1 both moved their best report
length toward "almost nothing" when labels were sparse, while the multiplicative form kept its
best length at N0 for any label density and ordered draws exactly as the evaluation's F1 does.
N0 therefore only sets the target length, which O1 ties to where the base's F1 peaks.

## 3. Effort and output length

- **Medium for both training and evaluation.** xhigh traces plus prompts reach ~130k tokens; the
  update on the A6000 host fits 96k and fails at 128k. A frozen medium base run becomes the
  comparison; the published xhigh base row stays as a reference.
- **Cap:** `max_tokens = 96k − prompt` (or the length probe's choice), and only rows whose prompt
  leaves room for the medium p95 completion.
- **O2:** a draw that hits the cap scores 0 and is recorded `truncated`; it is never graded from
  partial text. On the base draws, longer outputs did not find more, so the cap does not punish
  useful reasoning. Stops: 2 consecutive all-truncated groups; more than 10% of the last 20 draws
  truncated.

## 4. Learning rate

Measured from the October adapters (one step at 5e-6 between two checkpoints; LoRA r 16, α 32,
496 modules, 8-bit Adam):

| Quantity | Value |
|---|---|
| Mean \|B\| after 13 steps | 1.75e-5 (every module and layer within ±5%) |
| Mean \|ΔB\| per step | 1.2e-6 (0.24 × LR) |
| ‖ΔW‖/‖W‖ after 13 steps / per step | 8.5e-5 / 5.6e-6 |

- Adam normalises per element, so total movement ≈ LR × number of updates whatever the
  gradients. October moved 13 × 5e-6 = 6.5e-5 cumulative; published LoRA GRPO recipes reach about
  2e-3 to 2.5e-2. The earlier run was slow mainly because it had few updates.
- About half the groups give a reward contrast, so one pass over ~50–98 usable rows is 25–50
  updates. 100 updates at 2e-5 reach 2e-3.
- A fresh adapter's first Adam steps move ~LR per element (~4× steady state): 5-update warmup.
- **Probe** (GPU ~2 h, judge ≈ $1): from the step-0 adapter, one update each at 1e-5, 2e-5 and
  5e-5 on the same two graded groups; score held-out draws; take the largest LR with one-update KL
  ≤ 0.01 and < 5% of tokens outside the [0.8, 1.2] ratio range. Prior if skipped: 2e-5.

## 5. Groups, judge failures and overlap

- **One group per update** (no code change). Several groups per update is the main lever at
  larger scale (compute-planning.md §2); revisit after the proof of concept.
- **Unknown grades:** re-grade the group once from saved responses; still unknown → withhold the
  group (no update); stop after 2 consecutive withheld groups. Never score unknown as 0; never
  update on three of four draws.
- **Overlap with the five evaluation projects:** checked by file hashes and code shingles. No
  training row shares code or a labelled vulnerability with them. One corpus row with the same
  code as an evaluation project sits in the regression split and is never trained. Guard to add:
  `prepare` fails if a training row's file hash appears in an evaluation project's scope.

## 6. Schedule, acceptance and budget

**Question.** From the base model, does GRPO with this reward make the model find more known
vulnerabilities at a given report length on the five-project benchmark (medium effort), without
lowering published F1?

**Acceptance** at each pause (4 draws × 5 projects; per-draw values pooled as the unweighted
project mean; against the frozen medium base):
1. Volume-matched recall gain ≥ 0.03: the trained arm's per-draw exact recall minus the base's
   recall with each base draw cut to the trained arm's mean report length on that project
   (≈ 2 standard errors of a difference at four draws per project).
2. Published F1 (unique over four draws) ≥ the base's.
3. ≥ 19 of 20 trained draws graded.

A null result is reported, not extended.

**Segments.** One `schedules.json` entry per segment: step 0 → 25 from a zero-initialised adapter,
then → 50, → 100 and every 50. At each pause: merge, evaluate, publish with `publish-results.sh`,
decide. Every evaluated checkpoint is published (choosing the best on the same five projects would
select on the test set). O4: continue while a pause improves on the previous one by ≥ 0.03 in
volume-matched recall or per-draw F1, or the mean reward over the last 25 groups still rises.
Cutoffs are set at ~3× the expected segment time, as hang detectors.

Entry fields: `policy_reasoning_effort` medium, `tensor_parallel_size` 2, the 2-GPU
`update_placement`, `offload_layers` 64, `tiled_mlp` true, cap and row filter from the length
probe, learning rate from the probe with 5-update warmup.

**Time.** Medium visit ≈ 45–70 min (rollout 20–40, grading 5–10, update 15–25, reloads): 25
updates ≈ 2–2.5 days; an evaluation pause ≈ 5 h.

**Judge budget (O5).** ≈ $0.2–0.4 per group and $4–5 per evaluation pause; 200 groups + 6 pauses
≈ $70–110. Ledger cap $150 (the ledger refuses calls past it); $40 per segment; $15 per evaluation.

**Retries and stops.** Visits follow `failure_policy` (2 attempts; 2 consecutive failed visits
stop the segment). One relaunch per segment after a fix; a second failure means a design change.
Stops: 2 consecutive withheld groups; > 10% of the last 20 draws truncated; mean findings per draw
< 3 over 5 consecutive groups; ledger cap reached.

## 7. Work before training

1. Shared free-text prompt rendering and the derived corpus.
2. Free-text reward on `benchmark_free_text.grade_draw` with the §2 formula; new reward contract;
   JSON reward path archived.
3. Truncation scored 0 and recorded; truncation-rate and collapse stops in `controller`.
4. Zero-initialised step-0 adapter checkpoint.
5. Overlap guard in `prepare`; volume-matched recall in the comparison and publish output.
6. GPU, no judge: medium length probe (rollout only) for the cap and row filter.
7. GPU + judge: frozen medium base run (~5 h, ~$4); learning-rate probe (~$1).
8. Segment 1 schedule entry.

## Decision log

- 2026-10-22: train at medium and evaluate at xhigh was proposed and rejected (the effort is part
  of the prompt, so that would test transfer, not training).
- 2026-10-22: N0 = 15 proposed; revised to the F1-peak rule after comparing 10, 15 and 20.
- 2026-10-22: a 48 h window and fixed judge caps proposed; revised to segments with pauses and a
  $150 ledger cap (GPUs are free on the A6000 host).
- 2026-10-22: a fixed LR of 2e-5 proposed; revised to measured movement plus a probe.
- 2026-10-22: the long-sequence fit settled the effort (medium). An interim reading that weight
  streaming freed nothing on the 2-GPU split came from nvidia-smi and was wrong.
