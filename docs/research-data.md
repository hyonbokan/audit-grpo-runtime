# Research data

The measured figures behind the claims in [research.md](research.md), kept public so later
experiments can reuse them. Project names, dollar amounts and per-row identifiers are not
listed here. Update this file in the same change as `research.md`.

## 1. Pilot run (JSON prompt, medium effort, H100)

- 13 accepted updates over 12 distinct projects; 15 visits, 60 draws: 42 format-valid, 9 hits over
  59 findings, 18 draws scored -2 for format. About 48 min per visit; about 20 tok/s in training
  rollouts against 90-97 in standalone vLLM.
- Matched comparison on one project (11 reference issues, 4 draws): exact coverage 9.09% -> 13.64%,
  partial 18.18% -> 22.73%. All 8 reports failed strict JSON (code fences); scores come from copies
  with the outer fence removed. Later single-answer comparisons were mixed.
- Medium evaluation: 8 of 8 finished naturally, 5 failed strict Markdown parsing; only one project
  graded (23 issues: exact 0 -> 1, partial 2 -> 4).
- Checkpoint 13 vs base: 32 vs 32 matches; the base's 32 cover 20 distinct findings (8 High, 9
  Medium, 3 Low).
- Reward then: hits + 0.25 quality - 0.05 findings, malformed -2, empty 0. A -2 is 13-40x a typical
  wrong answer.
- LR: 13 steps at 5e-6 moved weights roughly 75-150x less than published LoRA GRPO recipes
  (1e-5 to 5e-5, 200-500 steps); delta-B per step 1.2e-6 (0.24 x LR).

## 2. Base model padding (frozen xhigh base, 5 projects, 20 draws, 100 known issues)

- 19-55 findings per draw (mean 31) against 11-34 known; slope 0.008 exact matches per extra finding.
- 59% of the 104 exact matches are in the report's first third, 6% in the last third.
- Precision by stated severity: High 80/307 (0.26), Medium 19/179 (0.11), Low 5/124 (0.04), Info 0/11.
- Per-draw precision 0.174, recall 0.284, F1 0.211. Published precision 0.077 (unique exact over all
  findings across four draws). Within-project corr(output tokens, exact) = -0.07.
- Keeping the first k findings: 10 -> 60 exact, F1 0.206; 15 -> 76, 0.226; 20 -> 92, 0.234;
  all -> 104, 0.211.
- Reasoning is about 83% of output tokens.

## 3. Reward design

- Training corpus: 98 rows, 216 labels (209 usable), all High/Medium; 64 rows have exactly one label.
  Benchmark: 19 of 100 labels are Low; train has none (Medium 144, High 65).
- Best report length under thinned labels (median [p10, p90]): hits - 0.05 N -> 2 [1, 5];
  hits - 0.02 N -> 12 [4, 20]; F1 -> 4 [1, 12]; recall x min(1, 15/N) -> 15 [15, 20] at every label
  density. Every N0 orders the base draws as F1 does (28 of 28 pairs). Cutting to N0 keeps 58% / 73% /
  88% of matches at N0 = 10 / 15 / 20.
- One-label group contrast with 4 draws: 42% (46% with partial). Partial credit at 0.5 leaves the
  optimum unchanged.
- Judge noise, base grading: 514 prompts, first two votes disagreed on 6.4%; 22 of 113 exact votes
  were contested against partial. Segment 1 cache: 105 of 109 prompts settled on two votes; 4 splits,
  all at the partial boundary; 10 of 14 prompts with any partial vote were unanimous.
- Benchmark base: 104 exact, 60 partial, 236 none; about two thirds of partials miss the impact or
  exploit path; 15 of 100 project-level findings are partial-only, and counting them lifts recall
  0.48 -> 0.63; the judge names a nearest finding in 137 of 236 no-match verdicts. Base xhigh recall
  0.48, precision 0.077.
- Gradient norm: 0.0023 at advantage +-0.05, 0.036-0.042 at +-0.5.
- Label gaps: 81% of train labels name a function, 78% have an Impact section, 9% are under 600
  characters; 99% of benchmark labels name the function.

## 4. Segment 1 (N0 = 20, LR 2e-5, seed 3407, cap 65,536 tokens, 4 draws per group, medium effort)

- One pass over 98 unique rows. Stopped after 62 visits by the "two consecutive failed visits" rule
  (both failures: vLLM could not start because another process held about 6.5 GiB on one GPU).
- Outcomes: 15 updates, 42 skips, 5 failures (two judge connection timeouts, one withheld grading,
  two GPU-memory). Update rate 15/62 = 24%, against the design's 50%. Final step 15.
- At visit 31: 6 updates (19%); visit 7 failed (judge timeout, 60,075-token rollout lost); about 1.5 h
  per visit.
- Skips are flat groups: all 0 (most), all 1 (one visit), all 0.5 (one visit).
- Unanswerable single-label rows: 7 of 98 (8 with a borderline one). Four pair an integrating
  project's finding with the upstream code it calls. A rough check flagged 37 of 209 findings.
  Six of them were visited and all skipped.
- Reward variants over the first 10 graded groups: N0 20, w 0 -> 3 groups with contrast; w 0.25 -> 4;
  N0 5, w 0 -> 4; N0 5, w 0.25 -> 5. N0 = 20 never binds (max 15 findings in the first 12 visits,
  up to about 19 later).
- First update health: loss 0.0036, grad norm 0.0023, ESS 0.9993; peak 22.2 GiB.
- Unused: 36 schedule slots (63-98) were never tried.

## 5. Overlap and leakage

- One evaluation project shares 12 of 22 files with a regression training row (44% of shingles);
  another shares 10.5% of shingles with a train row, all vendored Uniswap libraries. One library file
  matches an evaluation-corpus row that was not trained.
- Protocol census: about 1,000 protocols; 46 same-protocol clusters split across train / eval /
  calibration; 125 of 617 evaluation records (20%) belong to protocols that also have training
  families.

## 6. Epochs and replay

- Run state: the schedule is 98 unique ids, one pass; skipped and failed visits spend their slot.
- Papers read in full: DeepSeekMath 2402.03300, Prompt Replay 2603.21177, Prioritized Replay
  2601.02648, DAPO 2503.14476, Dr. GRPO 2503.20783. Seen only as search summaries: 1-shot RLVR
  2504.20571, diversity collapse 2606.15455, SPEC-RL 2509.23232, flat-rate identity 2607.00152.
- An earlier two-shuffled-pass plan and later unused-project schedules were pilot choices, not GRPO
  requirements.

## 7. Evaluation setup

- Training used medium reasoning effort. The only frozen base arm is xhigh, so the segment 1
  evaluation compares the trained adapter at xhigh against that base at xhigh. It tests whether a
  medium-trained adapter transfers; a null result does not rule out a medium-effort gain. A frozen
  medium base arm does not exist.
