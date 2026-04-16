from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from src.onnx.native_builder import (
    build_maxpool_model,
    build_maxpool_then_color_map_model,
    build_depthwise_shift_model,
    build_identity_model,
    build_minpool_model,
    build_minpool_then_color_map_model,
    build_pointwise_color_map_model,
    build_single_conv_model,
    build_shift_then_color_map_model,
    save_model,
)
from src.pipeline.submission_gate import evaluate_candidate
from src.pipeline.task_heuristics import (
    derive_global_color_mapping,
    derive_global_translation,
    mapping_to_weight_matrix,
)
from src.pipeline.linear_conv_fit import fit_single_conv3
from src.search.weird_finder import generate_sparse_weight_matrices


@dataclass
class CandidateArtifact:
    name: str
    path: Path


@dataclass
class TaskResult:
    task_id: str
    solved: bool
    selected_candidate: str | None
    output_model: str | None
    total_cost: int | None
    score: float | None
    tested_candidates: int
    failure_reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NeuroGolf submission.zip from task JSON files")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing task001.json ... task400.json",
    )
    parser.add_argument(
        "--output-zip",
        default="submission.zip",
        help="Output submission zip path",
    )
    parser.add_argument(
        "--output-model-dir",
        default="artifacts/submission_models",
        help="Directory to store selected task ONNX files",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts/work",
        help="Temporary work directory for candidate ONNX files",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=3,
        help="Maximum absolute shift for translation candidate inference",
    )
    parser.add_argument(
        "--weird-trials",
        type=int,
        default=0,
        help="Number of random sparse color-map candidates per task",
    )
    parser.add_argument(
        "--linear-conv",
        action="store_true",
        help="Enable least-squares 3x3 single-conv candidate fitting",
    )
    parser.add_argument(
        "--linear-conv-l2",
        type=float,
        default=1e-2,
        help="L2 regularization for least-squares conv fitting",
    )
    parser.add_argument(
        "--linear-conv-samples",
        type=int,
        default=5000,
        help="Max samples used for linear conv fitting per task",
    )
    parser.add_argument(
        "--max-examples-per-subset",
        type=int,
        default=None,
        help="Optional cap per subset for faster dry-runs",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Optional limit for debugging",
    )
    return parser.parse_args()


def _load_task(task_path: Path) -> dict[str, Any]:
    with task_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def _collect_task_files(data_dir: Path) -> list[Path]:
    task_files = sorted(data_dir.glob("task[0-9][0-9][0-9].json"))
    if len(task_files) == 0:
        raise FileNotFoundError(f"No task JSON files found in {data_dir}")
    return task_files


def _task_id_from_path(path: Path) -> str:
    return path.stem


def _is_identity_matrix(matrix: np.ndarray) -> bool:
    return bool(np.array_equal(matrix, np.eye(10, dtype=np.float32)))


def _build_candidate_models(
    task_id: str,
    payload: dict[str, Any],
    *,
    max_shift: int,
    weird_trials: int,
    enable_linear_conv: bool,
    linear_conv_l2: float,
    linear_conv_samples: int,
) -> list[tuple[str, Any]]:
    labeled_pairs = (
        list(payload.get("train", []))
        + list(payload.get("test", []))
        + list(payload.get("arc-gen", []))
    )

    candidates: list[tuple[str, Any]] = []
    candidates.append(("identity", build_identity_model(model_name=f"{task_id}_identity")))

    color_mapping = derive_global_color_mapping(labeled_pairs)
    color_matrix: np.ndarray | None = None
    if color_mapping is not None:
        color_matrix = mapping_to_weight_matrix(color_mapping)
        if not _is_identity_matrix(color_matrix):
            candidates.append(
                (
                    "color_map",
                    build_pointwise_color_map_model(
                        color_matrix,
                        model_name=f"{task_id}_color_map",
                    ),
                )
            )

    translation = derive_global_translation(labeled_pairs, max_shift=max_shift)
    if translation is not None:
        dx, dy = translation
        candidates.append(
            (
                f"shift_dx{dx}_dy{dy}",
                build_depthwise_shift_model(dx, dy, model_name=f"{task_id}_shift"),
            )
        )
        if color_matrix is not None and not _is_identity_matrix(color_matrix):
            candidates.append(
                (
                    f"shift_color_map_dx{dx}_dy{dy}",
                    build_shift_then_color_map_model(
                        dx,
                        dy,
                        color_matrix,
                        model_name=f"{task_id}_shift_color_map",
                    ),
                )
            )

    candidates.append(("maxpool3", build_maxpool_model(3, model_name=f"{task_id}_maxpool3")))
    candidates.append(("minpool3", build_minpool_model(3, model_name=f"{task_id}_minpool3")))
    if color_matrix is not None and not _is_identity_matrix(color_matrix):
        candidates.append(
            (
                "maxpool3_color_map",
                build_maxpool_then_color_map_model(
                    3,
                    color_matrix,
                    model_name=f"{task_id}_maxpool3_color_map",
                ),
            )
        )
        candidates.append(
            (
                "minpool3_color_map",
                build_minpool_then_color_map_model(
                    3,
                    color_matrix,
                    model_name=f"{task_id}_minpool3_color_map",
                ),
            )
        )

    if enable_linear_conv:
        fitted = fit_single_conv3(
            labeled_pairs,
            l2_lambda=linear_conv_l2,
            max_samples=linear_conv_samples,
        )
        if fitted is not None:
            weights, bias = fitted
            candidates.append(
                (
                    "linear_conv3",
                    build_single_conv_model(
                        weights,
                        bias=bias,
                        model_name=f"{task_id}_linear_conv3",
                    ),
                )
            )

    if weird_trials > 0:
        for idx, matrix in enumerate(
            generate_sparse_weight_matrices(trials=weird_trials, sparsity=0.92, seed=17),
            start=1,
        ):
            candidates.append(
                (
                    f"weird_sparse_{idx}",
                    build_pointwise_color_map_model(
                        matrix,
                        model_name=f"{task_id}_weird_sparse_{idx}",
                    ),
                )
            )

    return candidates


def _evaluate_task(
    task_id: str,
    payload: dict[str, Any],
    candidates: list[tuple[str, Any]],
    *,
    work_dir: Path,
    output_model_dir: Path,
    max_examples_per_subset: int | None,
) -> TaskResult:
    task_work_dir = work_dir / task_id
    task_work_dir.mkdir(parents=True, exist_ok=True)

    best_name: str | None = None
    best_path: Path | None = None
    best_total_cost: int | None = None
    best_score: float | None = None
    failure_reasons: list[str] = []

    for candidate_name, model in candidates:
        candidate_path = task_work_dir / f"{candidate_name}.onnx"
        save_model(model, candidate_path)

        decision = evaluate_candidate(
            candidate_path,
            payload,
            max_examples_per_subset=max_examples_per_subset,
            stop_on_first_failure=True,
        )

        if not decision.ok:
            if decision.reasons:
                failure_reasons.append(f"{candidate_name}: {decision.reasons[0]}")
            else:
                failure_reasons.append(f"{candidate_name}: unknown rejection")
            continue

        assert decision.cost is not None
        total_cost = decision.cost.total_cost
        if best_total_cost is None or total_cost < best_total_cost:
            best_name = candidate_name
            best_path = candidate_path
            best_total_cost = total_cost
            best_score = decision.cost.score

    if best_path is not None and best_name is not None and best_total_cost is not None:
        output_path = output_model_dir / f"{task_id}.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, output_path)
        return TaskResult(
            task_id=task_id,
            solved=True,
            selected_candidate=best_name,
            output_model=output_path.as_posix(),
            total_cost=best_total_cost,
            score=best_score,
            tested_candidates=len(candidates),
            failure_reasons=[],
        )

    return TaskResult(
        task_id=task_id,
        solved=False,
        selected_candidate=None,
        output_model=None,
        total_cost=None,
        score=None,
        tested_candidates=len(candidates),
        failure_reasons=failure_reasons[:5],
    )


def _write_report(results: list[TaskResult], report_path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(results),
        "solved_tasks": sum(1 for r in results if r.solved),
        "unsolved_tasks": sum(1 for r in results if not r.solved),
        "results": [r.__dict__ for r in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_zip(model_dir: Path, output_zip: Path) -> int:
    model_files = sorted(model_dir.glob("task[0-9][0-9][0-9].onnx"))
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as archive:
        for model_path in model_files:
            archive.write(model_path, arcname=model_path.name)

    return len(model_files)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_zip = Path(args.output_zip)
    output_model_dir = Path(args.output_model_dir)
    work_dir = Path(args.work_dir)

    if output_model_dir.exists():
        shutil.rmtree(output_model_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)

    task_files = _collect_task_files(data_dir)
    if args.limit_tasks is not None:
        task_files = task_files[: args.limit_tasks]

    print(f"[build] tasks discovered: {len(task_files)}")
    print(f"[build] data dir: {data_dir}")

    results: list[TaskResult] = []

    for index, task_file in enumerate(task_files, start=1):
        task_id = _task_id_from_path(task_file)
        payload = _load_task(task_file)
        candidates = _build_candidate_models(
            task_id,
            payload,
            max_shift=args.max_shift,
            weird_trials=args.weird_trials,
            enable_linear_conv=args.linear_conv,
            linear_conv_l2=args.linear_conv_l2,
            linear_conv_samples=args.linear_conv_samples,
        )

        result = _evaluate_task(
            task_id,
            payload,
            candidates,
            work_dir=work_dir,
            output_model_dir=output_model_dir,
            max_examples_per_subset=args.max_examples_per_subset,
        )
        results.append(result)

        if result.solved:
            print(
                f"[{index:03d}/{len(task_files):03d}] {task_id} solved "
                f"with {result.selected_candidate} | cost={result.total_cost}"
            )
        else:
            reason = result.failure_reasons[0] if result.failure_reasons else "no passing candidate"
            print(f"[{index:03d}/{len(task_files):03d}] {task_id} unsolved | {reason}")

    report_path = output_zip.parent / "build_report.json"
    _write_report(results, report_path)

    packaged = _build_zip(output_model_dir, output_zip)
    solved = sum(1 for r in results if r.solved)

    print(f"[done] solved: {solved}/{len(results)}")
    print(f"[done] packaged models: {packaged}")
    print(f"[done] report: {report_path}")
    print(f"[done] submission zip: {output_zip}")


if __name__ == "__main__":
    main()
