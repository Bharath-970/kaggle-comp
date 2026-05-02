#!/usr/bin/env python3
"""Train Conv1x1 models for all tasks flagged as needing neural."""
import json
import sys
sys.path.insert(0, '.')

from src.train_missing_tasks import solve_task_neural

needs = json.load(open('output/needs_neural.json'))
print(f"Training {len(needs)} tasks...")

results = []
for i, task_num in enumerate(needs):
    result = solve_task_neural(task_num, 'data', 'output', 'conv1x1')
    results.append(result)
    if (i + 1) % 50 == 0 or i == len(needs) - 1:
        score_str = f"{result.get('score', 'N/A'):.1f}" if isinstance(result.get('score'), (int, float)) else str(result.get('score', 'N/A'))
        print(f"[{i+1}/{len(needs)}] task{task_num:03d}: {result['status']}, score={score_str}")

# Save results
json.dump(results, open('output/neural_training_full_results.json', 'w'), indent=2)
print("\nDone! Trained results saved.")

# Check final score
sb = json.load(open('output/score_breakdown.json'))
if 'tasks' in sb and isinstance(sb['tasks'], dict):
    total = sum(sb['tasks'].values())
    print(f"Final score: {total:.1f}")
    print(f"Tasks solved: {len(sb['tasks'])}")
