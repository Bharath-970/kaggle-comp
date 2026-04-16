"""Evaluation utilities for running a model on ARC task datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

from .grid_codec import decode_tensor_to_grid, encode_grid_to_tensor
from .task_io import GridPair, TaskData, load_task_json

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None


@dataclass(frozen=True)
class SplitMetrics:
    total_pairs: int
    exact_pairs: int


@dataclass(frozen=True)
class TaskMetrics:
    task_id: str
    solved: bool
    train: SplitMetrics
    test: SplitMetrics
    arc_gen: SplitMetrics


@dataclass(frozen=True)
class DatasetMetrics:
    total_task_files: int
    evaluated_tasks: int
    skipped_tasks: int
    solved_tasks: int
    train: SplitMetrics
    test: SplitMetrics
    arc_gen: SplitMetrics

    @property
    def total_pairs(self) -> int:
        return self.train.total_pairs + self.test.total_pairs + self.arc_gen.total_pairs

    @property
    def exact_pairs(self) -> int:
        return self.train.exact_pairs + self.test.exact_pairs + self.arc_gen.exact_pairs



def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate RegisterBackbone models.")


def _model_device(model: Any) -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch is required to resolve model device.")
    if hasattr(model, "parameters"):
        try:
            first_param = next(model.parameters())
            return first_param.device
        except StopIteration:
            return torch.device("cpu")
        except TypeError:
            return torch.device("cpu")
    return torch.device("cpu")


def _predict_grid(model: Any, input_grid: list[list[int]], output_height: int, output_width: int) -> list[list[int]]:
    _require_torch()

    model.eval()
    encoded = encode_grid_to_tensor(input_grid)
    tensor = torch.from_numpy(encoded).to(_model_device(model))

    with torch.no_grad():
        output = model(tensor).detach().cpu().numpy()

    return decode_tensor_to_grid(output, output_height=output_height, output_width=output_width)


def _evaluate_pairs(model: Any, pairs: tuple[GridPair, ...]) -> SplitMetrics:
    total = len(pairs)
    exact = 0

    for pair in pairs:
        expected = pair.output_grid
        pred = _predict_grid(
            model=model,
            input_grid=pair.input_grid,
            output_height=len(expected),
            output_width=len(expected[0]),
        )
        if pred == expected:
            exact += 1

    return SplitMetrics(total_pairs=total, exact_pairs=exact)


def evaluate_task(
    model: Any,
    task_id: str,
    task_data: TaskData,
    require_all_splits: bool = True,
) -> TaskMetrics:
    train_metrics = _evaluate_pairs(model, task_data.train)
    test_metrics = _evaluate_pairs(model, task_data.test)
    arc_gen_metrics = _evaluate_pairs(model, task_data.arc_gen)

    if require_all_splits:
        solved = (
            train_metrics.exact_pairs == train_metrics.total_pairs
            and test_metrics.exact_pairs == test_metrics.total_pairs
            and arc_gen_metrics.exact_pairs == arc_gen_metrics.total_pairs
        )
    else:
        solved = test_metrics.exact_pairs == test_metrics.total_pairs

    return TaskMetrics(
        task_id=task_id,
        solved=solved,
        train=train_metrics,
        test=test_metrics,
        arc_gen=arc_gen_metrics,
    )


def _iter_task_files(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    return sorted(root.glob("task*.json"))


def evaluate_dataset(
    model: Any,
    dataset_root: str | Path,
    max_tasks: int | None = None,
    verbose: bool = True,
) -> tuple[DatasetMetrics, list[TaskMetrics], list[dict[str, str]]]:
    task_files = _iter_task_files(dataset_root)
    if max_tasks is not None:
        task_files = task_files[:max_tasks]

    train_total = train_exact = 0
    test_total = test_exact = 0
    arc_total = arc_exact = 0
    solved_tasks = 0
    results: list[TaskMetrics] = []
    skipped: list[dict[str, str]] = []

    for index, task_file in enumerate(task_files, start=1):
        task_id = task_file.stem
        try:
            task_data = load_task_json(task_file)
            task_metrics = evaluate_task(model, task_id, task_data)
        except ValueError as exc:
            skipped.append({"task_id": task_id, "reason": str(exc)})
            if verbose:
                print(f"Skipped {task_id}: {exc}")
            continue

        results.append(task_metrics)

        train_total += task_metrics.train.total_pairs
        train_exact += task_metrics.train.exact_pairs
        test_total += task_metrics.test.total_pairs
        test_exact += task_metrics.test.exact_pairs
        arc_total += task_metrics.arc_gen.total_pairs
        arc_exact += task_metrics.arc_gen.exact_pairs

        if task_metrics.solved:
            solved_tasks += 1

        if verbose and (index % 25 == 0 or index == len(task_files)):
            print(
                f"Processed {index}/{len(task_files)} files "
                f"| evaluated={len(results)} | skipped={len(skipped)} | solved={solved_tasks}"
            )

    dataset_metrics = DatasetMetrics(
        total_task_files=len(task_files),
        evaluated_tasks=len(results),
        skipped_tasks=len(skipped),
        solved_tasks=solved_tasks,
        train=SplitMetrics(total_pairs=train_total, exact_pairs=train_exact),
        test=SplitMetrics(total_pairs=test_total, exact_pairs=test_exact),
        arc_gen=SplitMetrics(total_pairs=arc_total, exact_pairs=arc_exact),
    )

    return dataset_metrics, results, skipped


def save_dataset_report(
    output_path: str | Path,
    dataset_root: str | Path,
    model_name: str,
    dataset_metrics: DatasetMetrics,
    task_results: list[TaskMetrics],
    skipped_tasks: list[dict[str, str]],
) -> None:
    report = {
        "dataset_root": str(dataset_root),
        "model_name": model_name,
        "summary": asdict(dataset_metrics),
        "total_pairs": dataset_metrics.total_pairs,
        "exact_pairs": dataset_metrics.exact_pairs,
        "skipped_tasks": skipped_tasks,
        "task_results": [asdict(item) for item in task_results],
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
