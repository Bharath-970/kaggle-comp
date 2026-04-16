# NeuroGolf 4280 Roadmap

## Objective
- Beat current first place 1780 by +2500 buffer.
- Target score: 4280.
- Strategy constraint: exact correctness first, then minimize cost.

## Baseline Snapshot
- Previous local run (from git history): 4 solved tasks, total 53.01 score.
- Gap to target 4280: 4226.99.

## What 4280 Means Quantitatively
- With 400 solved tasks, minimum required average score is 10.70.
- Practical high-confidence target band:
  - 330 solved at 13.0 average (4290)
  - 306 solved at 14.0 average (4284)
  - 286 solved at 15.0 average (4290)
- Any plan below 11 average score is non-viable unless nearly all tasks are solved.

## 3-Lane Solve Strategy
- Lane A: Cheap deterministic transforms
  - Examples: identity variants, color remaps, constant fills, simple shifts, nearest-shape copy rules.
  - Goal: very high solve volume with low-cost networks.
  - Score target per solved task: 14 to 18.
- Lane B: Medium composition transforms
  - Examples: crop+translate, mirrored placement, component extraction + recolor.
  - Goal: strong solve count while controlling model size.
  - Score target per solved task: 11 to 14.
- Lane C: Hard or ambiguous tasks
  - Use only after A and B throughput is stable.
  - Accept lower per-task score if correctness is achieved.
  - Score target per solved task: 6 to 11.

## 30-Day Execution Plan
1. Days 1-3: Foundation rebuild
- Recreate parser, validator, ONNX compliance gate, and cost profiler.
- Add deterministic search harness with repeatable seeds.

2. Days 4-10: Lane A saturation
- Prioritize easy archetypes first.
- Target cumulative solved tasks: 120.
- Target cumulative score: 1600 to 1900.

3. Days 11-18: Lane B expansion
- Add template families for object-level transformations.
- Target cumulative solved tasks: 220.
- Target cumulative score: 2900 to 3400.

4. Days 19-26: Lane C selective push
- Solve only tasks with favorable expected score gain per engineering hour.
- Target cumulative solved tasks: 290 to 330.
- Target cumulative score: 3900 to 4500.

5. Days 27-30: Stabilization and leaderboard defense
- Lock two final candidate submissions.
- Hold one exploratory daily submission slot, keep one exploit slot.

## Daily Submission Policy (5/day limit)
- Submission 1: baseline exploit model set.
- Submission 2: high-probability improvements from Lane A/B.
- Submission 3: one controlled risky variant.
- Submission 4: rollback-safe hybrid.
- Submission 5: best projected score package from same-day results.

## Gate Criteria Before Any Submission
- 100 percent exact correctness on available train, test, and arc-gen for all included tasks.
- ONNX static shape pass.
- No banned operators.
- File size under competition limit.
- Cost and score report generated per task and globally.

## Decision Rule for Task Prioritization
- Sort unsolved tasks by this objective:
  - expected_score_gain / expected_engineering_hours
- Re-rank daily after each submission result.

## Risk Controls
- Avoid overfitting to public examples by requiring multi-template agreement before finalizing fragile tasks.
- Keep a stable champion package that is never overwritten without measurable gain.
- Preserve reproducible build metadata for each candidate package.

## Representation Breakthrough Track (Mandatory for 4280+)

### Why this is mandatory
- Pure template expansion likely saturates around 2500 to 3200 because many ARC tasks require multi-step state manipulation, not just one-shot pattern matching.
- To break past that ceiling, the system needs compact task-specific models that still execute algorithm-like behavior inside static ONNX graphs.

### Breakthrough axis 1: Channel-as-memory
- Treat channel groups as fixed registers instead of only color logits.
- Proposed channel budget per 30x30 grid:
  - 10 input/output color channels (required)
  - 6 to 12 scratch channels for intermediate state
  - 2 to 4 channels for object masks and write-protect flags
  - 1 to 2 channels for phase markers (step tokens)
- Build minimal update blocks that repeatedly read and write these registers using small convolutions and gating, while keeping static shapes.

### Breakthrough axis 2: Implicit logic encoding
- Encode boolean logic through saturating linear transforms and threshold-like activations.
- Implement reusable primitives with ONNX-safe ops:
  - neighbor tests via shifted kernels
  - connected-region growth via iterative gated updates
  - symmetry and alignment checks via mirrored convolution templates
  - conditional recolor via mask-channel multiplication
- Goal: represent if/then-style behavior without banned control-flow operators.

### Breakthrough axis 3: Reusable neural programs
- Build a small library of neural micro-programs that compile into task-level ONNX models:
  - object extraction
  - object transport/placement
  - palette remapping
  - repetition and tiling
  - border and frame logic
- Each task search chooses and composes a few micro-programs with minimal parameterization instead of inventing a model from scratch.

### 14-day implementation sprint
1. Days 1-3: Memory-register backbone
- Implement register channel layout and 2 to 3-step read/write update blocks.
- Validate exact correctness on known easy tasks with no score regression.

2. Days 4-7: Logic primitive layer
- Add 8 to 12 ONNX-safe primitives for neighborhood logic, masking, and object selection.
- Benchmark primitive cost and keep only Pareto-efficient variants.

3. Days 8-11: Micro-program composer
- Add program assembler that stacks selected primitives into static graphs.
- Introduce search objective: exact correctness first, then minimize cost.

4. Days 12-14: Stress test and lock criteria
- Run on mixed archetype batch and compare against prior template-only baseline.
- Keep this track only if acceptance criteria are met.

### Acceptance criteria (go/no-go)
- Correctness lift: at least +60 newly solved tasks over template-only baseline on the same task set.
- Cost discipline: median score of newly solved tasks >= 12 (implied median cost <= 442,413).
- Stability: deterministic reruns select identical winners for at least 95 percent of tested tasks.
- Generalization proxy: no drop in arc-gen pass rate on already solved tasks.

### Score-aware budgeting for this track
- If targeting around 300 solved tasks, average score needed is about 14.27, so typical per-task cost must stay near 45,859.
- If targeting around 330 solved tasks, average score needed is about 12.97, so per-task cost can be near 167,762.
- Therefore, prioritize primitives that increase solved-task count while preserving low-cost composition.
