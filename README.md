# NeuroGolf 2026 - Fresh Iteration

This repository is a clean-slate iteration focused on reaching a 4280 leaderboard target (1780 + 2500 buffer).

## Implemented Core
- tools/score_planner.py: Quantifies solvability thresholds and score-cost tradeoffs.
- plans/roadmap_4280.md: Execution roadmap to reach and defend target score.
- src/neurogolf/scoring.py: Cost aggregation and score functions.
- src/neurogolf/grid_codec.py: ARC grid to tensor encoding and decoding.
- src/neurogolf/task_io.py: ARC task JSON loader for train/test/arc-gen splits.
- src/neurogolf/backbone.py: Channel-register backbone for memory-style reasoning.
- src/neurogolf/export.py: Static-shape ONNX export helper.
- src/neurogolf/onnx_rules.py: Banned-op, dynamic-shape, and file-size checks.
- scripts/export_register_backbone.py: CLI to export a register backbone ONNX model.
- tests/: Unit tests for scoring, codec, task loader, and ONNX file-size checks.

## Quick Start
1. Run score planner with default target:

   /opt/homebrew/bin/python3.11 tools/score_planner.py --current-score 53.01 --current-solved 4

2. Test a custom scenario:

   /opt/homebrew/bin/python3.11 tools/score_planner.py --portfolio "220:16,80:10"

3. Run unit tests:

   "/Users/bharath/Documents/kaggle comp/.venv-1/bin/python" -m pytest -q

4. Export an initial register-backbone ONNX model:

   "/Users/bharath/Documents/kaggle comp/.venv-1/bin/python" scripts/export_register_backbone.py --output artifacts/register_backbone.onnx

5. Run full dataset evaluation (Downloads/neurogolf-2026):

   "/Users/bharath/Documents/kaggle comp/.venv-1/bin/python" scripts/run_backbone_dataset.py --dataset-root "/Users/bharath/Downloads/neurogolf-2026" --report "artifacts/eval/register_backbone_eval.json" --quiet

6. Run local training slice (tasks 1 to 12):

   "/Users/bharath/Documents/kaggle comp/.venv-1/bin/python" scripts/run_local_slice_train.py --dataset-root "/Users/bharath/Downloads/neurogolf-2026" --start-task 1 --end-task 12 --epochs 40 --arcgen-train-sample 16 --batch-size 8 --report "artifacts/eval/local_slice_train_report.json" --eval-report "artifacts/eval/local_slice_eval_report.json"

7. Use roadmap:

   Open plans/roadmap_4280.md and execute milestones by lane.
