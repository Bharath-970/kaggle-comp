from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _shape(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def derive_global_color_mapping(train_pairs: Sequence[dict]) -> dict[int, int] | None:
    """Infer a global per-color mapping from training pairs, or return None on conflict."""
    if not train_pairs:
        return None

    mapping: dict[int, int] = {}

    for pair in train_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        if _shape(input_grid) != _shape(output_grid):
            return None

        for row_in, row_out in zip(input_grid, output_grid):
            for color_in, color_out in zip(row_in, row_out):
                previous = mapping.get(color_in)
                if previous is None:
                    mapping[color_in] = color_out
                elif previous != color_out:
                    return None

    for color in range(10):
        mapping.setdefault(color, color)

    return mapping


def mapping_to_weight_matrix(mapping: dict[int, int]) -> np.ndarray:
    if set(mapping.keys()) != set(range(10)):
        raise ValueError("Mapping must contain all input colors 0-9")

    matrix = np.zeros((10, 10), dtype=np.float32)
    for color_in in range(10):
        color_out = mapping[color_in]
        if color_out < 0 or color_out > 9:
            raise ValueError("Mapped color must be in [0, 9]")
        matrix[color_out, color_in] = 1.0
    return matrix


def _shift_matches_pair(input_grid: Sequence[Sequence[int]], output_grid: Sequence[Sequence[int]], dx: int, dy: int) -> bool:
    height, width = _shape(input_grid)
    out_h, out_w = _shape(output_grid)
    if (height, width) != (out_h, out_w):
        return False

    for row in range(height):
        for col in range(width):
            src_row = row - dy
            src_col = col - dx
            if 0 <= src_row < height and 0 <= src_col < width:
                source_value = input_grid[src_row][src_col]
            else:
                # Outside original border is represented as clear/zero-hot -> color 0.
                source_value = 0
            if output_grid[row][col] != source_value:
                return False
    return True


def derive_global_translation(
    train_pairs: Sequence[dict],
    *,
    max_shift: int = 3,
) -> tuple[int, int] | None:
    """Infer one global translation valid for all train pairs, or None if absent."""
    if not train_pairs:
        return None

    candidates: list[tuple[int, int]] = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            candidates.append((dx, dy))

    candidates.sort(key=lambda item: (abs(item[0]) + abs(item[1]), abs(item[0]), abs(item[1])))

    for dx, dy in candidates:
        if all(_shift_matches_pair(pair["input"], pair["output"], dx, dy) for pair in train_pairs):
            return dx, dy

    return None
