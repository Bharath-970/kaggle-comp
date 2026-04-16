from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.data.encoding import grid_shape, validate_grid


@dataclass(frozen=True)
class PairFeatures:
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    input_colors: tuple[int, ...]
    output_colors: tuple[int, ...]
    horizontal_symmetry_input: bool
    vertical_symmetry_input: bool


@dataclass(frozen=True)
class TaskIntelligence:
    task_id: str
    pair_features: tuple[PairFeatures, ...]
    inferred_families: dict[str, float]
    invariance_hints: dict[str, bool]
    difficulty_score: float


def _colors(grid: Sequence[Sequence[int]]) -> tuple[int, ...]:
    values = sorted({cell for row in grid for cell in row})
    return tuple(values)


def _is_horizontal_symmetric(grid: Sequence[Sequence[int]]) -> bool:
    return all(row == list(reversed(row)) for row in grid)


def _is_vertical_symmetric(grid: Sequence[Sequence[int]]) -> bool:
    top = [list(row) for row in grid]
    return top == list(reversed(top))


def extract_pair_features(
    input_grid: Sequence[Sequence[int]],
    output_grid: Sequence[Sequence[int]],
) -> PairFeatures:
    validate_grid(input_grid)
    validate_grid(output_grid)

    in_h, in_w = grid_shape(input_grid)
    out_h, out_w = grid_shape(output_grid)

    return PairFeatures(
        input_height=in_h,
        input_width=in_w,
        output_height=out_h,
        output_width=out_w,
        input_colors=_colors(input_grid),
        output_colors=_colors(output_grid),
        horizontal_symmetry_input=_is_horizontal_symmetric(input_grid),
        vertical_symmetry_input=_is_vertical_symmetric(input_grid),
    )


def compute_family_priors(pair_features: Sequence[PairFeatures]) -> dict[str, float]:
    if not pair_features:
        return {
            "color_map": 0.2,
            "symmetry": 0.2,
            "object": 0.2,
            "composition": 0.2,
            "counting": 0.2,
        }

    priors = {
        "color_map": 0.15,
        "symmetry": 0.15,
        "object": 0.2,
        "composition": 0.2,
        "counting": 0.15,
        "spatial_program": 0.15,
    }

    same_shape_ratio = mean(
        1.0
        if (p.input_height, p.input_width) == (p.output_height, p.output_width)
        else 0.0
        for p in pair_features
    )
    color_shift_ratio = mean(
        1.0 if p.input_colors != p.output_colors else 0.0 for p in pair_features
    )
    symmetry_signal = mean(
        1.0 if p.horizontal_symmetry_input or p.vertical_symmetry_input else 0.0
        for p in pair_features
    )

    priors["color_map"] += 0.25 * color_shift_ratio
    priors["symmetry"] += 0.3 * symmetry_signal
    priors["spatial_program"] += 0.2 * same_shape_ratio
    priors["object"] += 0.2 * (1.0 - same_shape_ratio)

    total = sum(priors.values())
    return {name: value / total for name, value in priors.items()}


def infer_invariance_hints(pair_features: Sequence[PairFeatures]) -> dict[str, bool]:
    same_shape_all = all(
        (p.input_height, p.input_width) == (p.output_height, p.output_width)
        for p in pair_features
    )
    color_set_stable = all(p.input_colors == p.output_colors for p in pair_features)

    return {
        "allow_translation_perturbation": same_shape_all,
        "allow_rotation_perturbation": same_shape_all,
        "allow_color_permutation": color_set_stable,
        "allow_noise": False,
    }


def estimate_difficulty(pair_features: Sequence[PairFeatures]) -> float:
    if not pair_features:
        return 1.0

    size_term = mean((p.input_height * p.input_width) / 900.0 for p in pair_features)
    shape_change_term = mean(
        0.0
        if (p.input_height, p.input_width) == (p.output_height, p.output_width)
        else 1.0
        for p in pair_features
    )
    color_shift_term = mean(
        0.0 if p.input_colors == p.output_colors else 1.0 for p in pair_features
    )

    return 1.0 + size_term + 0.7 * shape_change_term + 0.5 * color_shift_term


def load_task_json(task_path: str | Path) -> dict[str, Any]:
    path = Path(task_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not {"train", "test", "arc-gen"}.issubset(payload.keys()):
        raise ValueError("Task json must include train, test, and arc-gen keys")
    return payload


def analyze_task(task_id: str, task_payload: dict[str, Any]) -> TaskIntelligence:
    train_pairs = task_payload.get("train", [])
    pair_features: list[PairFeatures] = []

    for pair in train_pairs:
        pair_features.append(extract_pair_features(pair["input"], pair["output"]))

    priors = compute_family_priors(pair_features)
    invariances = infer_invariance_hints(pair_features)
    difficulty = estimate_difficulty(pair_features)

    return TaskIntelligence(
        task_id=task_id,
        pair_features=tuple(pair_features),
        inferred_families=priors,
        invariance_hints=invariances,
        difficulty_score=difficulty,
    )
