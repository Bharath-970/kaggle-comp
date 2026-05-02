# RunPod Execution Plan

## Current State
- **Local:** Content-aware transform detectors implemented & tested ✓
- **Code:** Pushed to main branch ✓
- **Missing:** RunPod SSH access (keys need verification)

## Steps to Execute on RunPod

### 1. Fix SSH Access (If Needed)
```bash
# Test connection with different key
ssh -i ~/.ssh/id_ed25519 g6jiiomctlp94f-6441139d@ssh.runpod.io

# Or direct TCP
ssh -i ~/.ssh/id_ed25519 root@69.30.85.75 -p 22146

# If keys still don't work, regenerate SSH keys in RunPod console
```

### 2. Run Symbolic Pipeline (Content-Aware Transforms)
```bash
# SSH into RunPod
ssh g6jiiomctlp94f-6441139d@ssh.runpod.io

# Navigate to project
cd /workspace/kaggle-comp
git pull

# Run pipeline in background
screen -S pipeline
python3 src/pipeline_symbolic.py --data_dir data --output_dir output
# Ctrl+A then D to detach

# Monitor progress
screen -r pipeline
tail -f output/score_breakdown.json
```

**Expected Results:**
- Run time: ~5-10 minutes
- Content-aware hflip/vflip/rot180 solves ~10-50 additional tasks (those with fixed bounds)
- Score improvement: +50-150 pts (conservative estimate)

### 3. Train Neural Models for Missing Tasks
```bash
# After pipeline finishes, train Conv1x1 models for missing tasks
screen -S training
python3 src/train_missing_tasks.py \
    --data_dir data \
    --output_dir output \
    --model_type conv1x1 \
    --tasks 14 30 31 36 49 65 80 135 150 155 170 177 184 202 249 269 300 307 310 326 339 351 366 370 384 396

# Ctrl+A then D to detach

# Monitor
tail -f output/neural_training_results.json
```

**Expected Results:**
- Run time: ~5-15 minutes (Conv1x1 is fast)
- Trains all 26 missing tasks
- Accuracy varies (some tasks are too complex for simple Conv)
- Average score on trained models: 12-16 pts
- Total improvement: +300-400 pts (26 tasks × 12-16 pts)

### 4. (Optional) Refine with Conv3x3 Models
```bash
# For tasks where Conv1x1 fails, try Conv3x3 (slower but more expressive)
python3 src/train_missing_tasks.py \
    --data_dir data \
    --output_dir output \
    --model_type conv3x3 \
    --tasks 36 49 65 150 155 170  # Tasks that Conv1x1 struggled with
```

### 5. Check Final Score
```bash
# Verify all 400 tasks have ONNX models
ls output/task*.onnx | wc -l  # Should be 400

# Check score breakdown
cat output/score_breakdown.json | python3 -c "
import json, sys
sb = json.load(sys.stdin)
print(f'Total score: {sb[\"total\"]:.1f}')
print(f'Tasks solved: {len(sb[\"tasks\"])}')
print(f'Missing: {len(sb.get(\"bad_tasks\", []))}')
"
```

### 6. Package for Kaggle Submission
```bash
python3 src/pack_submission.py --output_dir output

# Download submission.zip (should be <1.44MB per model × 400 = 576MB max)
# Upload to Kaggle
```

---

## Debugging

### Pipeline Errors
```bash
# Check logs
tail -f output/needs_neural.json

# Run single task to debug
python3 -c "
import sys
sys.path.insert(0, '.')
from src.pipeline_symbolic import solve_task
result = solve_task(14, 'data', 'output')
print(result)
"
```

### Training Errors
```bash
# Check training output
grep -i error output/neural_training_results.json

# Retry specific task
python3 src/train_missing_tasks.py --data_dir data --output_dir output --tasks 14 --model_type conv3x3
```

### ONNX Validation
```bash
python3 -c "
import onnx
m = onnx.load('output/task014.onnx')
onnx.checker.check_model(m)
print('✓ Valid ONNX')
"
```

---

## Expected Outcome

### Conservative Estimate (80% confidence)
- Content-aware: +100 pts
- Neural training (Conv1x1): +250 pts
- **Total: ~5400 pts** (from 5028.9)

### Optimistic Estimate (50% confidence)
- Content-aware: +200 pts
- Neural training (Conv1x1+3×3 refinement): +400 pts
- **Total: ~5600 pts**

### Target: 7000 pts
- Would require solving an additional ~1500 pts beyond above
- Possible if: finding additional symbolic patterns, training deeper Conv models, or reverse-engineering expensive Conv tasks
- Estimated effort: 2-3 more weeks of development

---

## Key Constraints
1. **Max model size:** 1.44MB per ONNX file
2. **Operations:** No LOOP, SCAN, NONZERO, UNIQUE (no dynamic control flow)
3. **Format:** [1, 10, 30, 30] float32, ONNX opset 10

## Files Modified
- `src/analyze.py` — Added content-aware transform detectors
- `src/onnx_builder.py` — Added content-aware ONNX builders
- `src/pipeline_symbolic.py` — Integrated new detectors
- `src/train_missing_tasks.py` — NEW neural training script

## Next Steps After RunPod
1. Check final score
2. If <6000: Reverse-engineer expensive Conv tasks (identify patterns in weights)
3. If <6500: Train larger models (Conv3x3 stacks) for selected tasks
4. If <7000: Requires significant new pattern discovery or advanced techniques
