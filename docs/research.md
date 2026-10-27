# Research: what we are testing and what we have learned

The main research document of this project: the question, the method, the findings so far, the
literature that bears on the next decisions, the open questions and the next experiments. It
synthesizes the pilot run, the base-model measurements, the reward analysis and segment 1 of the
free-text campaign. Updated 2026-10-27. Segment 1 training stopped early (§2.4); its evaluation was still running, so
everything about segment 1's effect is provisional until that evaluation is in.

**Reading it.** The measured figures (benchmark draws, judge grades, corpus counts) are in
[research-data.md](research-data.md). Project names, row ids and prices stay in the git-ignored
`private/`. This document states what the
figures showed. Design decisions and their rules are in [training-design.md](training-design.md);
hardware limits are in [compute-planning.md](compute-planning.md). Literature claims marked
(summary) were read from search summaries or preprints, not full texts: check the paper before
relying on one in a decision.

## 1. Question and method

**Question.** Does GRPO with LoRA, starting from the unchanged base model, make a 27B open model
find more of the known vulnerabilities in unseen Solidity projects, at a given report length and
without lowering published F1? Matching flagship models is a longer-term ambition, not the
success threshold. The result says nothing about other fine-tuning methods or larger models.

**Setup.** The model reads a project's source and writes a free-text audit. A judge extracts the
findings and matches each to the private ground-truth findings (exact, partial or none). The
reward is recall times a volume factor. Four draws form a group; the update uses reward minus the
group mean; one group per update; a group with equal rewards is skipped. Ground truth is for
grading only and never enters a prompt.

## 2. Findings so far

### 2.1 Pilot run (October, JSON output)

- **Feasibility holds, improvement is not established.** The pipeline made real, recoverable
  updates over a handful of projects. Small positive signals sat beside mixed results and a
  regression; no conclusion beyond "the pipeline moves the model" is supported.
- **Measurement was the weakest part.** Strict parsers rejected finished answers (code fences,
  early stops mid-reasoning), so some arms had no valid grade. Missing grades must never be scored
  as zero, and structural validity must be reported apart from content.
- **The reward mixed two jobs.** A large floor for malformed answers dominated updates in groups
  where the only contrast was format, and the per-finding cost barely punished false findings.
  This is why the redesign dropped format from the reward.
- **Generation dominates cost.** Long prompts and long reasoning make each group slow; the pilot's
  update budget was too small to expect large gains. Movement of the adapter is roughly learning
  rate times number of updates, whatever the gradients, so few updates at a low rate moved the
  weights far less than published LoRA GRPO recipes.
- **Operational lessons.** A copied plan silently overrode a safer judge-retry default; one
  ambiguous judge match withheld a whole group; cumulative per-visit backups grew with the square
  of the visit count; a CUDA launch failure and two consecutive withheld groups ended a run early;
  caps that truncated all four draws gave equal rewards and a skipped group. Each is now a rule or
  a check in the controller and launchers.

### 2.2 The base model pads its reports

- At xhigh the base writes far more findings than there are known vulnerabilities, many of them
  checklist items (missing zero-address checks, admin power, missing events), so precision is low.
- Extra findings add almost no exact matches within a project; matches concentrate early in the
  report. Output length is uncorrelated with matches.
- Precision falls steeply with the report's own severity label, and a few "Critical" labels are
  centralization notes.
- A reward without a volume term would let training raise recall by listing more, which is why
  the reward carries one (§2.3).

### 2.3 Reward design under sparse labels

- Labels are sparse: most training rows carry a single known finding, and real but unlabelled bugs
  score as no-match. We decided unlabelled bugs are **not** rewarded: the judge cannot assess them
  and many labelled findings are auditors' false positives anyway.
- Penalty-style rewards (hits minus a per-finding cost) and F1 choose very different report
  lengths when labels are thinned, so they are unstable under sparse labels. `recall × min(1, N0/N)`
  puts its best report length at N0 whatever the label density, and N0 at the base's per-draw F1
  peak keeps most matches. Every N0 tried ranks base draws within a project the way F1 does.
- A tighter N0 would teach dropping Low-severity findings that the benchmark counts but the
  training labels never contain.
- **The judge is mostly stable.** Nearly every match prompt settles on the first two votes. Every
  disagreement involved the partial boundary; the tie rule favours the higher category and has not
  fired.
- **Partial matches** mean right place and core issue with the consequences only partly described.
  Most partials miss the impact or exploit path, so converting partial to exact is the most
  valuable step and the gap between the two must stay large.
- **Idea, not applied: a step ladder** (location about 0.1, root cause about 0.25, exact 1, best
  rung per known finding, volume factor unchanged). Advantage has no std scaling, so reward gaps
  set update size directly, which caps partial at 0.25. Location is mostly reached already and is
  noisy to judge, and it would pay shallow checklist findings in the right function, so the first
  version pays partial only. Log location, mechanism and impact as separate judge booleans
  (log-only) and price location later (≤ 0.05) if they prove stable.
- **Label gaps.** Training labels are thinner than the benchmark's: some have short or missing
  impact text, so exact versus partial rests partly on the judge's inference, and the training
  `contracts` field is unreliable.

### 2.4 Segment 1: why so many groups were flat

Segment 1 is the first free-text training run (N0 = 20, constant learning rate, one pass over
the training rows). Its update rate was well below the design's assumption that about half the
groups give a contrast. A review of the flat groups found, by cause:

| Cause | Share of flat groups reviewed | Fix |
|---|---|---|
| Right place, wrong issue, scored no-match | largest | None wanted: crediting location rewards spreading findings across functions |
| Labelled code absent from the prompt (unanswerable) | several | Remove the row |
| Root cause right, impact missing (judge partial) | one | Partial credit pays exactly this gap |
| Easy row, all draws exact | one | None |

- **Some rows cannot be solved from their prompt.** The single-label rows were read by hand: a
  small fraction cite code that is not in the prompt, often an audit of an integrating project
  paired with the code of the upstream project it used. Hypothesis, not yet checked: the corpus
  builder joins findings to repositories too loosely. Each such row costs a group of GPU time for
  a certain zero.
- **Reward variants on the saved grades.** Partial credit and a lower N0 each add a contrast group
  or two among those graded; the volume factor never binds on these reports at N0 = 20.
- **Failures were few and environmental.** Three visits failed in grading (two were judge connection
  timeouts on both attempts; grading has no backoff between retries). Two consecutive visits then
  failed because vLLM could not start: another process on the shared host held GPU memory, and the
  controller's two-consecutive-failures rule ended training with about a day and a half of the
  schedule unused. Lessons: check free GPU memory before a rollout, back off between judge retries,
  and reserve the GPUs with the other users of a shared host before a long run.
- **Outcome.** The run did one partial pass: most visited rows were skipped, a minority updated,
  and the unvisited rows remain available for a continuation (training-design.md §8).
- **Reports in flat groups** are short checklists; the known findings are protocol-logic and
  economic. No draw was truncated and report volume stayed far above the stop floor.

### 2.5 Data and leakage

Held-out projects were checked by file hashes and code shingles against the training corpus: no
training row shares a file with them beyond vendored libraries, and one corpus row with the same
code as an evaluation project is in the regression split and never trained. Alias-level overlap
with the wider archive is still open, and the controls do not address pretraining exposure to
public benchmark material.

## 3. Epochs, replay and resampling

"Epoch" means two things in GRPO:

- **Passes over the prompt set.** Each pass samples fresh rollouts from the current policy, so a
  second pass does not replay labelled outputs the way a supervised epoch does.
- **Several gradient steps per rollout batch** (GRPO's μ). This reuses samples off-policy and the
  PPO-style clip bounds it. Our controller takes one update per group, so it does not apply.

Three meanings of "replay" must stay apart: prompt revisiting with fresh answers; reuse of old
answers (off-policy, needs a probability-correction contract); and retention evaluation. Prompt
revisiting is the first scheduling candidate; old-answer reuse is not planned.

**What the literature says.**

- DeepSeekMath's GRPO samples prompt batches from a dataset with the current policy; its 10%
  historical replay was for the reward model, not a policy retention rule.
- Repeated passes over small curated prompt sets are common. One training example repeated for
  thousands of steps still improved held-out accuracy, with overfitting only after about 1.4k steps
  (1-shot RLVR, arXiv 2504.20571; a math base model).
- Many epochs over a full dataset eventually collapse diversity: training success keeps rising
  while validation Pass@1 plateaus, reported after about epoch 5 (arXiv 2606.15455; summary).
- Consecutive epochs of the same prompts overlap heavily in tokens (SPEC-RL, arXiv 2509.23232;
  summary).
- Re-queuing medium-difficulty prompts speeds early gains but converges to the baseline when reuse
  is too aggressive (Prompt Replay, arXiv 2603.21177). Prioritized replay by observed success
  statistics is proof-of-concept only (arXiv 2601.02648).
- A group is flat when all draws score the same. With G draws at success rate p the flat rate is
  p^G + (1−p)^G, high at both extremes (arXiv 2607.00152; summary). DAPO's dynamic sampling drops
  flat groups and samples until the batch has contrast, at a rollout cost (arXiv 2503.14476).
- Dr. GRPO's length and difficulty biases come from particular normalization choices. Our
  objective mean-centers rewards with no std scaling and normalizes by group tokens, so audit the
  actual objective before claiming to apply a fix (arXiv 2503.20783).

**For us.** At one group per update and an hour or more per group, one pass gives tens of updates,
far below the step counts in those papers: too few updates is the risk, not overfitting. A
resample helps rows where the model finds the bug only sometimes; it cannot help unanswerable
rows, and it does not help a row that is flat for lack of reward separation unless the reward
changes. Audit-specific cautions: success is not binary, four draws are a noisy estimate, so
priority must not react strongly to one group, and coverage across vulnerability types and
severities matters as much as sampling efficiency.

### How the controller treats rows today

- The schedule is a list of unique row ids; `reviewed_training_order` rejects `epochs != 1` and
  duplicates.
- `next_cycle` takes the row at index `len(visits)`. A skipped or failed visit consumes its slot
  and is never requeued. The only repeats are a pending visit resumed after a restart and one
  in-visit retry of a failed grading that reuses the saved rollout.
- A continuation, a second pass or a resample is a new segment over its own list of unique rows, so
  `epochs` stays 1. It needs a new reviewed schedule, a way to give the schedule that row list
  (`training_row_ids` is accepted by the controller but not yet written by the segment launcher),
  and the user's approval, and can only be built while no run is live.

## 4. Open questions

1. Does the step-25 evaluation show a volume-matched recall gain with F1 held? (Pending; the
   comparison is trained-at-medium against the base at xhigh, a transfer test.)
2. Does the corpus builder join findings to repositories too loosely? How many multi-label rows
   carry an unanswerable label?
3. How stable are the judge's partial verdicts, and do separate location, mechanism and impact
   verdicts agree across calls?
4. Does the model's reasoning change? Evaluation grades the final answer only; compare reasoning
   length, files inspected and support for each reported finding, base versus trained.
5. Do training and evaluation disagree about validity (training accepts a fenced answer)?
6. Alias-level overlap between trained protocols and the held-out projects is unchecked.
7. Total cost and staff effort are not reconciled, so the method's practicality is unassessed.

## 5. Options for the next segment (none applied)

1. Drop unanswerable rows and check the corpus builder's join first.
2. Partial credit of 0.25 on the judge's partial verdict under a new reward contract version, with
   exact recall as the headline and partial coverage reported alongside; N0 stays 20 until a lower
   value is checked against the base's findings-versus-F1 curve.
3. A second pass (a new segment over a chosen row list) over answerable rows, ordered by informativeness (partial or root-cause matches
   first, saturated rows last).
4. One bounded resample of a flat group on the current policy when its draws were all partial or
   all zero on an answerable row.
5. A paid calibration of the location judge field on saved draws, with approval.

**A measured comparison, once a bounded protocol is reviewed.** Compare the single-pass schedule
with a shuffled mixed pool, then with limited revisiting of high-signal rows. Fix the starting
checkpoint, row pool, reward, draws per group, effort and held-out evaluation; fix the replay
fraction, cooldown, reuse cap and coverage rules in advance. Match compute and judge budgets and
report actual use. Measure quality and retention at fixed milestones, useful updates per GPU-hour,
skips and cost. Change scheduling before objective or memory so effects stay separable.

Each option needs a written question, acceptance conditions and time and cost limits, as in
[training-design.md](training-design.md) §6.

## 6. What would support a conclusion

Set thresholds before collecting results.

1. A defined improvement over the unchanged base on held-out projects, with repeated matched
   evaluation, variation and failures by category.
2. Reasoning and unsupported claims measured apart from the training reward; validity apart from
   content.
3. Prompts, scope, effort, budgets, grading and recovery protocol frozen for both arms.
4. Accepted updates and row exposure recorded; retention evaluated before it is claimed.
5. Cost and staff effort reconciled before practicality is assessed.
6. A larger-model replication only after an adequate small-model result.

## 7. Change log

- 2026-10-27: segment 1 stopped early after two consecutive rollout failures caused by GPU memory
  held by another process; the evaluation of its last accepted checkpoint is running. Corrected §3:
  a second pass needs a new row list, not a relaxed `epochs` guard.
- 2026-10-25: first version: pilot lessons, base behaviour, reward analysis, segment 1 flat groups,
  epochs and replay literature, open questions and options.
