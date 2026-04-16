from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data.encoding import GridValidationError, exact_output_match, grid_to_tensor


@dataclass
class CorrectnessReport:
    total_examples: int
    passed_examples: int
    failed_examples: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total_examples > 0 and self.passed_examples == self.total_examples


def _iter_examples(
    payload: dict[str, Any],
    subsets: Iterable[str],
    max_per_subset: int | None,
):
    for subset in subsets:
        pairs = payload.get(subset, [])
        if max_per_subset is not None:
            pairs = pairs[:max_per_subset]
        for idx, pair in enumerate(pairs):
            yield subset, idx, pair


def evaluate_model_on_payload(
    model_path: str | Path,
    task_payload: dict[str, Any],
    *,
    subsets: tuple[str, ...] = ("train", "test", "arc-gen"),
    max_per_subset: int | None = None,
    stop_on_first_failure: bool = False,
) -> CorrectnessReport:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required for correctness evaluation") from exc

    session = ort.InferenceSession(Path(model_path).as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    total = 0
    passed = 0
    failed: list[str] = []

    for subset, idx, pair in _iter_examples(task_payload, subsets, max_per_subset):
        total += 1
        try:
            encoded = grid_to_tensor(pair["input"]).astype(np.float32)
        except (GridValidationError, KeyError, TypeError) as exc:
            failed.append(f"{subset}[{idx}] invalid_input: {exc}")
            if stop_on_first_failure:
                return CorrectnessReport(total_examples=total, passed_examples=passed, failed_examples=failed)
            continue

        output = session.run(None, {input_name: encoded})[0]

        try:
            is_match = exact_output_match(output, pair["output"])
        except (GridValidationError, KeyError, TypeError) as exc:
            failed.append(f"{subset}[{idx}] invalid_output: {exc}")
            if stop_on_first_failure:
                return CorrectnessReport(total_examples=total, passed_examples=passed, failed_examples=failed)
            continue

        if is_match:
            passed += 1
        else:
            failed.append(f"{subset}[{idx}]")
            if stop_on_first_failure:
                return CorrectnessReport(total_examples=total, passed_examples=passed, failed_examples=failed)

    return CorrectnessReport(total_examples=total, passed_examples=passed, failed_examples=failed)
