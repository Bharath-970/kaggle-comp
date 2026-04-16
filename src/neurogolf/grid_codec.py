"""ARC grid <-> tensor conversion helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .constants import COLOR_CHANNELS, GRID_SIZE


def _validate_grid(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not grid:
        raise ValueError("Grid cannot be empty.")

    height = len(grid)
    width = len(grid[0])
    if width == 0:
        raise ValueError("Grid rows cannot be empty.")

    if height > GRID_SIZE or width > GRID_SIZE:
        raise ValueError(f"Grid shape {height}x{width} exceeds {GRID_SIZE}x{GRID_SIZE}.")

    for row in grid:
        if len(row) != width:
            raise ValueError("Grid rows must have equal width.")
        for value in row:
            if value < 0 or value >= COLOR_CHANNELS:
                raise ValueError(f"Color index out of range: {value}")

    return height, width


def encode_grid_to_tensor(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Encode an ARC grid into [1, 10, 30, 30] one-hot tensor."""
    height, width = _validate_grid(grid)

    tensor = np.zeros((1, COLOR_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for row_idx in range(height):
        for col_idx in range(width):
            color = int(grid[row_idx][col_idx])
            tensor[0, color, row_idx, col_idx] = 1.0

    return tensor


def decode_tensor_to_grid(tensor: np.ndarray, output_height: int, output_width: int, strict: bool = False) -> list[list[int]]:
    """Decode tensor logits/probabilities back to color grid with fixed output shape."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected batch size 1.")
        channel_tensor = tensor[0]
    elif tensor.ndim == 3:
        channel_tensor = tensor
    else:
        raise ValueError("Tensor must have shape [1,C,H,W] or [C,H,W].")

    if channel_tensor.shape[0] != COLOR_CHANNELS:
        raise ValueError(f"Expected {COLOR_CHANNELS} channels, got {channel_tensor.shape[0]}.")

    if output_height < 1 or output_width < 1:
        raise ValueError("output_height and output_width must be positive.")
    if output_height > GRID_SIZE or output_width > GRID_SIZE:
        raise ValueError("Requested output shape exceeds 30x30.")

    output: list[list[int]] = []
    for row_idx in range(output_height):
        row: list[int] = []
        for col_idx in range(output_width):
            pixel_logits = channel_tensor[:, row_idx, col_idx]
            if strict:
                non_zero = int(np.count_nonzero(pixel_logits > 0.5))
                if non_zero != 1:
                    raise ValueError(
                        f"Strict decode expected one active channel at ({row_idx}, {col_idx}), got {non_zero}."
                    )
            row.append(int(np.argmax(pixel_logits)))
        output.append(row)

    return output


def get_color_normalization_map(grids: list[list[list[int]]]) -> list[int]:
    """Returns a unified mapping from present colors across all grids to 0, 1, 2... in order of appearance."""
    # Background (0) is always 0
    mapping = {0: 0}
    next_color = 1
    for grid in grids:
        for row in grid:
            for val in row:
                if val not in mapping:
                    mapping[val] = next_color
                    next_color += 1
    
    # Create the full 10-channel map (permutation)
    # This map says: channel k in original -> channel mapping[k] in normalized
    res = list(range(10))
    for k, v in mapping.items():
        if k < 10:
            res[k] = v
    return res


def apply_color_map(grid: list[list[int]], mapping: list[int]) -> list[list[int]]:
    """Applies a color mapping to a grid."""
    return [[mapping[val] for val in row] for row in grid]
