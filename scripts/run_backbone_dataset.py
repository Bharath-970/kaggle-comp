#!/usr/bin/env python3
"""Run RegisterBackbone on ARC dataset tasks and report exact-match metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.backbone import RegisterBackbone
from neurogolf.evaluate import evaluate_dataset, save_dataset_report


DEFAULT_DATASET = "/Users/bharath/Downloads/neurogolf-2026"
DEFAULT_REPORT = "artifacts/eval/register_backbone_eval.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RegisterBackbone on ARC dataset")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--scratch-channels", type=int, default=8)
    parser.add_argument("--mask-channels", type=int, default=2)
    parser.add_argument("--phase-channels", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    args = build_parser().parse_args()

    model = RegisterBackbone(
        steps=args.steps,
        scratch_channels=args.scratch_channels,
        mask_channels=args.mask_channels,
        phase_channels=args.phase_channels,
        hidden_channels=args.hidden_channels,
    )

    summary, task_results, skipped_tasks = evaluate_dataset(
        model=model,
        dataset_root=args.dataset_root,
        max_tasks=args.max_tasks,
        verbose=not args.quiet,
    )

    save_dataset_report(
        output_path=args.report,
        dataset_root=args.dataset_root,
        model_name="RegisterBackbone",
        dataset_metrics=summary,
        task_results=task_results,
        skipped_tasks=skipped_tasks,
    )

    train_acc = _safe_ratio(summary.train.exact_pairs, summary.train.total_pairs)
    test_acc = _safe_ratio(summary.test.exact_pairs, summary.test.total_pairs)
    arc_acc = _safe_ratio(summary.arc_gen.exact_pairs, summary.arc_gen.total_pairs)
    overall_acc = _safe_ratio(summary.exact_pairs, summary.total_pairs)

    print("Dataset evaluation complete")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Total task files: {summary.total_task_files}")
    print(f"Evaluated tasks: {summary.evaluated_tasks}")
    print(f"Skipped tasks: {summary.skipped_tasks}")
    print(f"Solved tasks: {summary.solved_tasks}")
    print(f"Train exact: {summary.train.exact_pairs}/{summary.train.total_pairs} ({train_acc:.4%})")
    print(f"Test exact: {summary.test.exact_pairs}/{summary.test.total_pairs} ({test_acc:.4%})")
    print(f"Arc-gen exact: {summary.arc_gen.exact_pairs}/{summary.arc_gen.total_pairs} ({arc_acc:.4%})")
    print(f"Overall exact: {summary.exact_pairs}/{summary.total_pairs} ({overall_acc:.4%})")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
