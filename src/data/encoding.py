from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

MAX_GRID_SIZE = 30
NUM_COLORS = 10


class GridValidationError(ValueError):
    """Raised when a grid is not valid for ARC encoding."""


def validate_grid(grid: Sequence[Sequence[int]]) -> None:
    if not grid:
        raise GridValidationError("Grid must have at least one row")

    row_length = len(grid[0])
    if row_length == 0:
        raise GridValidationError("Grid must have at least one column")

    if len(grid) > MAX_GRID_SIZE or row_length > MAX_GRID_SIZE:
        raise GridValidationError("Grid dimensions exceed 30x30")

    for row in grid:
        if len(row) != row_length:
            raise GridValidationError("Grid rows must have equal length")
        for value in row:
            if value < 0 or value >= NUM_COLORS:
                raise GridValidationError(f"Grid value {value} is outside [0, 9]")


def grid_to_tensor(
    grid: Sequence[Sequence[int]],
    *,
    height: int = MAX_GRID_SIZE,
    width: int = MAX_GRID_SIZE,
    channels: int = NUM_COLORS,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Encode a color grid to [1, channels, height, width] one-hot tensor."""
    validate_grid(grid)

    if height != MAX_GRID_SIZE or width != MAX_GRID_SIZE:
        raise GridValidationError("Only 30x30 static shape is supported")
    if channels != NUM_COLORS:
        raise GridValidationError("Only 10 channels are supported")

    tensor = np.zeros((1, channels, height, width), dtype=dtype)
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            tensor[0, color, row_index, col_index] = 1.0
    return tensor


def decode_tensor_to_grid(
    tensor: np.ndarray,
    *,
    output_height: int,
    output_width: int,
) -> list[list[int]]:
    """Decode model output tensor into a color grid for the requested output shape."""
    if tensor.shape != (1, NUM_COLORS, MAX_GRID_SIZE, MAX_GRID_SIZE):
        raise GridValidationError(
            "Expected tensor shape [1, 10, 30, 30], "
            f"received {tuple(int(v) for v in tensor.shape)}"
        )

    if output_height < 1 or output_width < 1:
        raise GridValidationError("Output shape must be positive")
    if output_height > MAX_GRID_SIZE or output_width > MAX_GRID_SIZE:
        raise GridValidationError("Output shape exceeds 30x30")

    active = tensor[0, :, :output_height, :output_width]
    argmax = np.argmax(active, axis=0)
    max_scores = np.max(active, axis=0)

    result: list[list[int]] = []
    for row_index in range(output_height):
        row: list[int] = []
        for col_index in range(output_width):
            if max_scores[row_index, col_index] <= 0.0:
                # Empty vectors are treated as color 0 during decode only.
                row.append(0)
            else:
                row.append(int(argmax[row_index, col_index]))
        result.append(row)
    return result


def exact_output_match(predicted: np.ndarray, expected_grid: Sequence[Sequence[int]]) -> bool:
    """Check exact ARC compliance for expected region and padded zero-hot border."""
    validate_grid(expected_grid)

    expected_height = len(expected_grid)
    expected_width = len(expected_grid[0])
    expected_tensor = grid_to_tensor(expected_grid)

    if predicted.shape != expected_tensor.shape:
        return False

    rounded = np.where(predicted > 0.0, 1.0, 0.0)

    # Expected active region must be exact one-hot by color.
    in_region_equal = np.array_equal(
        rounded[:, :, :expected_height, :expected_width],
        expected_tensor[:, :, :expected_height, :expected_width],
    )
    if not in_region_equal:
        return False

    # Border region outside expected dimensions must be zero-hot.
    border = rounded.copy()
    border[:, :, :expected_height, :expected_width] = 0.0
    return bool(np.count_nonzero(border) == 0)


def grid_shape(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    validate_grid(grid)
    return len(grid), len(grid[0])
