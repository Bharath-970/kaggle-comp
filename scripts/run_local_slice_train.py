#!/usr/bin/env python3
"""Train RegisterBackbone on a local slice, then evaluate on that slice."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.backbone import RegisterBackbone
from neurogolf.evaluate import evaluate_task
from neurogolf.train import (
    SliceRunSummary,
    TrainConfig,
    iter_task_files,
    load_task_for_training,
    save_slice_training_report,
    select_task_files,
    train_task_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RegisterBackbone on a local task slice")
    parser.add_argument("--dataset-root", default="/Users/bharath/Downloads/neurogolf-2026")
    parser.add_argument("--start-task", type=int, default=1)
    parser.add_argument("--end-task", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--arcgen-train-sample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--scratch-channels", type=int, default=8)
    parser.add_argument("--mask-channels", type=int, default=2)
    parser.add_argument("--phase-channels", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--use-coords", type=bool, default=True)
    parser.add_argument("--use-depthwise", type=bool, default=True)
    parser.add_argument("--report", default="artifacts/eval/local_slice_train_report.json")
    parser.add_argument("--eval-report", default="artifacts/eval/local_slice_eval_report.json")
    return parser


def _safe_ratio(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return num / den


def main() -> None:
    args = build_parser().parse_args()

    all_files = iter_task_files(args.dataset_root)
    selected_files = select_task_files(all_files, start_index=args.start_task, end_index=args.end_task)

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        arcgen_train_sample=args.arcgen_train_sample,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    trained = 0
    skipped: list[dict[str, str]] = []
    solved = 0

    train_summaries = []
    task_eval_rows = []

    train_pairs_total = train_pairs_exact = 0
    test_pairs_total = test_pairs_exact = 0
    arc_pairs_total = arc_pairs_exact = 0

    for idx, task_file in enumerate(selected_files, start=1):
        task_id = task_file.stem
        try:
            task = load_task_for_training(task_file)
        except ValueError as exc:
            skipped.append({"task_id": task_id, "reason": str(exc)})
            print(f"[{idx}/{len(selected_files)}] skipped {task_id}: {exc}")
            continue

        model = RegisterBackbone(
            steps=args.steps,
            scratch_channels=args.scratch_channels,
            mask_channels=args.mask_channels,
            phase_channels=args.phase_channels,
            hidden_channels=args.hidden_channels,
            use_coords=args.use_coords,
            use_depthwise=args.use_depthwise,
        )

        train_summary = train_task_model(model=model, task=task, task_id=task_id, config=config)
        train_summaries.append(train_summary)
        trained += 1

        metrics = evaluate_task(model=model, task_id=task_id, task_data=task)
        if metrics.solved:
            solved += 1

        train_pairs_total += metrics.train.total_pairs
        train_pairs_exact += metrics.train.exact_pairs
        test_pairs_total += metrics.test.total_pairs
        test_pairs_exact += metrics.test.exact_pairs
        arc_pairs_total += metrics.arc_gen.total_pairs
        arc_pairs_exact += metrics.arc_gen.exact_pairs

        task_eval_rows.append(
            {
                "task_id": task_id,
                "solved": metrics.solved,
                "train_exact": metrics.train.exact_pairs,
                "train_total": metrics.train.total_pairs,
                "test_exact": metrics.test.exact_pairs,
                "test_total": metrics.test.total_pairs,
                "arc_gen_exact": metrics.arc_gen.exact_pairs,
                "arc_gen_total": metrics.arc_gen.total_pairs,
                "best_loss": train_summary.best_loss,
                "final_loss": train_summary.final_loss,
            }
        )

        print(
            f"[{idx}/{len(selected_files)}] {task_id} "
            f"solved={metrics.solved} "
            f"train={metrics.train.exact_pairs}/{metrics.train.total_pairs} "
            f"test={metrics.test.exact_pairs}/{metrics.test.total_pairs} "
            f"arc={metrics.arc_gen.exact_pairs}/{metrics.arc_gen.total_pairs}"
        )

    summary = SliceRunSummary(
        total_task_files=len(all_files),
        selected_task_files=len(selected_files),
        trained_tasks=trained,
        skipped_tasks=len(skipped),
    )

    save_slice_training_report(
        report_path=args.report,
        dataset_root=args.dataset_root,
        summary=summary,
        train_summaries=train_summaries,
        eval_report_path=args.eval_report,
        skipped=skipped,
        config=config,
    )

    eval_payload = {
        "dataset_root": args.dataset_root,
        "start_task": args.start_task,
        "end_task": args.end_task,
        "summary": {
            "selected_files": len(selected_files),
            "trained_tasks": trained,
            "skipped_tasks": len(skipped),
            "solved_tasks": solved,
            "train_exact": f"{train_pairs_exact}/{train_pairs_total}",
            "test_exact": f"{test_pairs_exact}/{test_pairs_total}",
            "arc_gen_exact": f"{arc_pairs_exact}/{arc_pairs_total}",
            "train_accuracy": _safe_ratio(train_pairs_exact, train_pairs_total),
            "test_accuracy": _safe_ratio(test_pairs_exact, test_pairs_total),
            "arc_gen_accuracy": _safe_ratio(arc_pairs_exact, arc_pairs_total),
        },
        "tasks": task_eval_rows,
        "skipped": skipped,
    }

    eval_path = Path(args.eval_report)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(__import__("json").dumps(eval_payload, indent=2))

    print("---")
    print("Slice training complete")
    print(f"Selected files: {len(selected_files)}")
    print(f"Trained tasks: {trained}")
    print(f"Skipped tasks: {len(skipped)}")
    print(f"Solved tasks: {solved}")
    print(f"Train exact: {train_pairs_exact}/{train_pairs_total} ({_safe_ratio(train_pairs_exact, train_pairs_total):.4%})")
    print(f"Test exact: {test_pairs_exact}/{test_pairs_total} ({_safe_ratio(test_pairs_exact, test_pairs_total):.4%})")
    print(f"Arc-gen exact: {arc_pairs_exact}/{arc_pairs_total} ({_safe_ratio(arc_pairs_exact, arc_pairs_total):.4%})")
    print(f"Train report: {args.report}")
    print(f"Eval report: {args.eval_report}")


if __name__ == "__main__":
    main()
