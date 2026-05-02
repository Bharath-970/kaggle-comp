# NeuroGolf 2026 Kaggle Competition — Detailed Strategy Report

**Goal:** Reach 7000 points from current 5028.9 (gap: +1971 pts)  
**Current State:** 374/400 tasks solved, 26 missing  
**Deadline Assumption:** Ongoing Kaggle competition  

---

## I. COMPETITION OVERVIEW

### What is NeuroGolf 2026?
- **Task:** Solve 400 ARC-AGI-style grid transformation problems
- **Submission Format:** 400 ONNX neural network models (one per task), each ≤1.44MB
- **Input/Output Encoding:** [1, 10, 30, 30] float32 tensors
  - Batch=1, Channels=10 (ARC colors), Height=30, Width=30
  - One-hot color encoding: pixel (r,c) = color i means tensor[0, i, r, c] = 1.0
  - All grids zero-padded to 30×30 (variable-size content regions)
- **Constraints:** 
  - ONNX opset 10 only
  - No LOOP, SCAN, NONZERO, UNIQUE operators (no dynamic control flow)
  - Max 1.44MB per model file
  - Each task has train + test examples, plus 262 ARC-GEN validation examples per task

### Scoring Formula
```
score_per_task = max(1.0, 25.0 - ln(MACs + memory + params))
total_score = sum of all 400 task scores
```
- **MACs:** Floating-point multiply-accumulate ops (dominates cost for Conv layers)
- **Memory:** Size of intermediate tensors in bytes (NOT input/output, NOT float params counted as size)
- **Params:** Count of float32 parameters in initializers
- **Key insight:** Int64 initializers count as params but are small; Conv layers dominate cost
- **Best possible:** Score 25 per task = 9350 total (all zero-cost models)

---

## II. CURRENT STATE BREAKDOWN

### Score Distribution (374 solved tasks)
| Band | Count | Total Pts | Example |
|------|-------|-----------|---------|
| 22-25 (ultra-cheap) | 4 | 88.6 | Identity, Gather-only models |
| 19-22 (cheap) | 2 | 38.1 | Crop + color remap |
| 15-19 (medium) | 68 | 1080.1 | Mirror patterns, multi-op (Gather+Slice+Pad) |
| 13-15 (conv1x1) | 137 | 1867.6 | Single 1×1 Conv: ~90k MACs each |
| 11-13 (conv3x3) | 163 | 1954.4 | Single 3×3 Conv: ~810k MACs each |

### Missing Tasks (26 tasks = ~300-400 pts potential)
```
14, 30, 31, 36, 49, 65, 80, 135, 150, 155, 170, 177, 184, 202, 
249, 269, 300, 307, 310, 326, 339, 351, 366, 370, 384, 396
```
- **Shapes:** Highly variable (3×3 to 30×30 inputs; transform outputs range 1×2 to 15×19)
- **Patterns:** Most detected as "unknown" by pattern analyzer
- **Key tasks:**
  - task150 (4×4 → 4×4): **Content-aware hflip** ← needs feature
  - task155 (5×5 → 5×5): **Content-aware vflip** ← needs feature
  - Others: Complex spatial transforms, pattern matching, stamping

### ONNX Model Breakdown (374 files)
- **Conv-based:** 148 models (1×1 Conv: 90k MACs ~13.6pts; 3×3 Conv: 810k MACs ~11.4pts)
- **Non-conv:** 247 models
  - 10 models: Gather+Pad+Slice (cost ~30-50, score ~22.5)
  - 7 models: Concat+Pad+Slice (4-way mirror ops, score ~16.5-17.7)
  - Rest: Complex multi-op chains (ArgMax, ReduceMax, CumSum, Reshape, etc.)

### Upgrade Potential
- **Theoretical max:** 9350 (all tasks score 25)
- **Current:** 5028.9
- **Gap:** 4321.1 points (46% upside)

---

## III. ROOT CAUSE ANALYSIS

### Why Scores Are Low

#### 1. **OLD Solver System (Now Deleted)**
- Previous `src/neurogolf/solvers.py` trained Conv models for "unknown" tasks
- Generated 368 expensive models that are now in `output/`
- Models are Pareto-optimal for accuracy, not for cost
- 163 use 3×3 Conv (810k MACs → ~11.4 pts each = 1860 pts total)
- Cannot recover original solver logic (code deleted)

#### 2. **NEW Symbolic Pipeline (Current)**
- `src/pipeline_symbolic.py` detects simple patterns: identity, hflip, vflip, rot180, rot90, color_perm, tile_2x2/3x3, constant
- Only 6 tasks solved by new system (rest "unknown" → skipped)
- Does NOT detect content-aware transforms (transforms that apply only to actual content region, not padding)
- Does NOT handle variable-size content → misses many hflip/vflip opportunities

#### 3. **Missing Content-Aware Transforms**
Current `check_hflip()` flips full 30×30:
```python
def check_hflip(examples) -> bool:
    return apply_network_numpy(examples, lambda x: x[:, :, ::-1].copy())
    # Flips ALL 30 columns, padding included ← WRONG for variable-size content
```

Correct for task150 (4×4 content in 30×30):
```
Input:  [.... 4×4 content at cols 0-3 ...]
Output: [.... 4×4 content at cols 0-3, VALUES FLIPPED within region ...]
```
But `check_hflip` produces output with flipped content at cols 26-29 (right side) ← fails match

---

## IV. SOLUTION STRATEGY

### Phase 1: Quick Wins (500-700 pts, 3-5 days)

#### A. Fix Content-Aware Transforms (100-150 pts)
**Implementation:**
1. Add `check_content_aware_hflip()`, `check_content_aware_vflip()`, `check_content_aware_rot180()` to `analyze.py`
   - Detect content bounds using `_active_bounds()`
   - Verify bounds are FIXED across all examples
   - Crop input, test transform, verify output matches (cropped region + zero padding outside)
   
2. Add ONNX builders to `onnx_builder.py`
   ```python
   def content_aware_hflip(r0, r1, c0, c1):
       # Slice(input, [0, c0:c1, c0:c1]) → hflip → Pad(restore to full 30×30)
   ```
   Cost: Slice params (8 int64) + Pad params (8 int64) ≈ ~20 total
   Score: ~23-24 pts

3. Update `pipeline_symbolic.py` PATTERN_TO_BUILDER
   - Test these patterns before returning "unknown"

**Expected gain:**
- task150, task155 + similar: ~6-10 tasks → ~150 pts
- Replaces old conv models with symbolic

#### B. Solve 26 Missing Tasks (200-300 pts)
**Strategy:**
1. Run enhanced pipeline locally with content-aware detectors
2. For remaining "unknown" among missing tasks:
   - Push code to RunPod
   - Train small neural models (Conv1x1 or Conv3x3)
   - Score and submit
   
**Expected breakdown:**
- Content-aware fixes: ~10-12 tasks solved
- Small neural: ~14-16 tasks at ~15-18 pts each = 210-270 pts
- Total: ~300 pts

#### C. Replace Expensive Conv Models (500-1000 pts)
**Why?** 300 tasks currently score 11-15 pts. If even 50% can be replaced with 22+ pt symbolic models = +550 pts.

**How?** Reverse-engineer old models:
1. For each Conv task in output/, examine actual trained weights
2. Check if weights show patterns (diagonal shifts, block structure, etc.)
3. If pattern found → build cheap ONNX equivalent
4. Run pipeline with new detectors

**Practical approach:**
- Focus on Conv1x1 models first (90k MACs, easier to pattern-match)
- Low-hanging fruit: All-zero regions in weights indicate unused channels → simpler models
- Medium fruit: Weight matrices with rank-1 structure → reducible via Gather

**Expected gain:** 50-100 tasks upgraded → 550-1100 pts (if successful)

---

### Phase 2: Advanced Patterns (300-500 pts, 1-2 weeks)

#### D. Hardcoded Crop + Pattern Detection (100-200 pts)
196 "unknown" solved tasks have **fixed content size**.

**Implementation:**
```python
def check_crop_and_transform(examples, target_pattern):
    # Examples: task087 is 3×3 content, rot180, then color remap
    # Detect: for all examples, bounds = (r0,r1,c0,c1) constant
    # Then: crop → apply target_pattern → color_remap → pad back
```

**Gain:** 196 fixed-size tasks, maybe 20% have simple patterns → 40 tasks × 10 pts = 400 pts

#### E. Mirror & Reflect Patterns (50-100 pts)
Already detected: 7 tasks with 4-way mirror (task083, 142, 152, etc.) at ~16.5 pts each.

Extend to:
- 2-way mirrors (horizontal + vertical separately)
- Diagonal mirrors (transpose + mirror)
- Rotational symmetry detections

**Expected:** 10-20 additional tasks → 150-200 pts

---

### Phase 3: Marginal Optimization (200-300 pts, 2-3 weeks)

#### F. Tune Conv Model Replacements
- For Conv tasks that can't be made symbolic, try:
  - Depth-wise Conv (separate per channel) instead of full Conv
  - Smaller kernels (1×1 instead of 3×3)
  - Separable Conv (H-sep + W-sep)

#### G. Fix optimize_onnx_all.py (20-50 pts upside, likely negative)
- Current MaxPool → 8 Slice+Pad replacement INCREASES memory cost
- Should remove or redesign
- Low priority (likely neutral or negative ROI)

---

## V. IMMEDIATE ACTION PLAN (Next 48 Hours)

### Step 1: Implement Content-Aware Detectors (4 hours)
**File: `src/analyze.py`**

Add functions:
```python
def check_content_aware_hflip(examples) -> bool:
    """Content stays in region, values flip within region."""
    r0, r1, c0, c1 = None, None, None, None
    
    for i_oh, o_oh in examples:
        bounds = _active_bounds(i_oh)
        if r0 is None:
            r0, r1, c0, c1 = bounds
        elif bounds != (r0, r1, c0, c1):
            return False  # Bounds not fixed
        
        # Crop content
        in_crop = i_oh[:, r0:r1, c0:c1]
        out_crop = o_oh[:, r0:r1, c0:c1]
        
        # Verify padding is zero outside
        out_zero = o_oh.copy()
        out_zero[:, r0:r1, c0:c1] = 0
        if out_zero.any():
            return False
        
        # Test hflip
        pred = in_crop[:, :, ::-1].copy()
        if not np.array_equal(pred, out_crop):
            return False
    
    return (r0, r1, c0, c1)  # Return bounds if match
```

Similar for `check_content_aware_vflip()`, `check_content_aware_rot180()`

### Step 2: Add ONNX Builders (3 hours)
**File: `src/onnx_builder.py`**

```python
def content_aware_hflip(r0, r1, c0, c1):
    """Crop [r0:r1, c0:c1], hflip, pad back."""
    # Slice to crop region
    # Slice.starts = [0, 0, c0, c0], ends = [1, 10, r1-r0, c1-c0]
    # Hflip on axis 3
    # Pad back to [0, 30, c0, 30-c1]
```

### Step 3: Update Pipeline (2 hours)
**File: `src/pipeline_symbolic.py`**

In `analyze_task()`, before "unknown" return:
```python
bounds = detect_content_aware_hflip(examples)
if bounds:
    return {"pattern": "content_aware_hflip", "params": bounds, ...}
```

### Step 4: Test Locally (2 hours)
```bash
python3 src/pipeline_symbolic.py --data_dir data --output_dir output_test
# Check task150, task155 are now solved
```

### Step 5: Push to RunPod & Score (4 hours)
```bash
git add -A && git commit -m "Add content-aware transforms"
git push origin main

# SSH to RunPod
ssh g6jiiomctlp94f-6441139d@ssh.runpod.io -i ~/.ssh/id_ed25519
cd kaggle-comp && git pull
python3 src/pipeline_symbolic.py --data_dir data --output_dir output
# This auto-scores and saves improved models
```

---

## VI. RUNPOD EXECUTION

### Why RunPod?
- **Local:** Pipeline + scoring takes ~2-3 minutes (blocking machine)
- **RunPod:** GPU-accelerated, $300 credit remaining, no local slowdown

### Connection Details
```bash
# SSH (proxied, recommended)
ssh g6jiiomctlp94f-6441139d@ssh.runpod.io -i ~/.ssh/id_ed25519

# Direct TCP (if proxied fails)
ssh root@69.30.85.75 -p 22146 -i ~/.ssh/id_ed25519

# Jupyter (for interactive work)
# Port 8888 via proxied domain
```

### Typical Workflow
```bash
# 1. Commit + push local changes
git add src/analyze.py src/onnx_builder.py src/pipeline_symbolic.py
git commit -m "Implement content-aware transforms"
git push

# 2. SSH to RunPod
ssh g6jiiomctlp94f-6441139d@ssh.runpod.io -i ~/.ssh/id_ed25519

# 3. Pull + run in persistent screen
cd /workspace/kaggle-comp
git pull
screen -S solver
python3 src/pipeline_symbolic.py --data_dir data --output_dir output
# Ctrl+A then D to detach

# 4. Monitor progress
screen -r solver

# 5. When done, check results
cat output/needs_neural.json
wc -l output/task*.onnx
python3 -c "import json; sb=json.load(open('output/score_breakdown.json')); print(f'Total: {sb[\"total\"]:.1f}')"
```

---

## VII. FILE STRUCTURE & KEY MODULES

### Core Pipeline
```
src/
  analyze.py                 # Pattern detection (identity, hflip, vflip, etc.)
  onnx_builder.py           # ONNX model builders
  pipeline_symbolic.py      # Main pipeline: detect → build → verify → score
  pack_submission.py        # Zip ONNX models for Kaggle submission
  
data/                       # 400 task JSON files (task001.json to task400.json)
output/                     # Final ONNX models + score_breakdown.json + needs_neural.json
```

### Key Functions

#### `analyze.py`
- `load_task(task_num)` → dict with train/test/arc-gen examples
- `get_examples(task, include_arcgen=True)` → list of (input_onehot, output_onehot) pairs
- `analyze_task(task_num)` → `{"pattern": ..., "params": ...}`
- Pattern checkers: `check_identity()`, `check_hflip()`, `check_vflip()`, etc.

#### `onnx_builder.py`
- Builders: `identity()`, `hflip()`, `vflip()`, `color_permutation(perm)`, `tile_2x2()`, etc.
- All return `onnx.ModelProto` with [1,10,30,30] input/output
- Cost = MACs + memory + params
- Score = `max(1.0, 25.0 - ln(cost))`

#### `pipeline_symbolic.py`
- `solve_task(task_num, data_dir, output_dir, existing_score)` → attempts to solve one task
- `run_pipeline()` → processes all 400 tasks, saves ONNX to output/, outputs score_breakdown.json
- Requires `neurogolf_utils` for scoring (only on RunPod)

---

## VIII. RISK ASSESSMENT & MITIGATION

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Content-aware bounds vary across examples | Low (easy to detect) | Check `_active_bounds()` consistency |
| ONNX Slice/Pad params are counted as float (not int64) | Medium | Verify onnx_tool counts only actual float params, not int64 |
| RunPod code diverges from local | Low | Always git push before RunPod, git pull before run |
| Old conv models can't be reverse-engineered | Medium (affects Phase 2) | Skip reverse-engineering if not viable; focus on new patterns instead |
| Training neural models takes too long | Low (we have $300 credit) | Use small Conv1x1 models, quick training |

---

## IX. SUCCESS METRICS

### 7000 Point Target Breakdown
```
Current:                                        5028.9
+ Content-aware transforms (10-15 tasks):        +150
+ Solve missing tasks (26 tasks):                +300
+ Heuristic pattern replacements:                +400
+ Conv model replacements (best effort):         +400
————————————————————————————————————————————————————
Estimated:                                      ~6300

Additional needed:                                +700
  (More aggressive reverse-engineering / training)
```

### Achievable Target: 6200-6500 (Very Confident)
- Content-aware: +150 (high confidence)
- Missing tasks: +250-300 (medium-high confidence)
- Pattern heuristics: +200-250 (medium confidence)
- Conv replacements: +300-400 (medium confidence, depends on reverse-engineering success)

### Stretch Goal: 7000 (Possible but requires intensive work)
- Requires solving additional 70-100 "unknown" tasks with trained neural models
- Or finding additional patterns across existing solved tasks
- Timeline: 2-3 weeks if feasible

---

## X. SUMMARY

**Current Status:** 5028.9 pts (374/400 tasks solved)  
**Gap to 7000:** +1971 pts  
**Next 48 hours:** Implement content-aware transforms, solve missing tasks locally, push to RunPod  
**Expected 1-week result:** 6000-6300 pts  
**Next 3 weeks:** Aggressive pattern detection + selective neural training to reach 6500-7000  

All code changes push to RunPod for scoring (neurogolf_utils required). Local machine for development only.
