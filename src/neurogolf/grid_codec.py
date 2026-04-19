"""ARC grid <-> tensor conversion helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .constants import COLOR_CHANNELS, GRID_SIZE, IDENTITY_CHANNELS, STATE_CHANNELS
from scipy.ndimage import label


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
    """Encode an ARC grid into [1, 10+K, GRID_SIZE, GRID_SIZE] tensor with identity tracking."""
    height, width = _validate_grid(grid)

    tensor = np.zeros((1, STATE_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    # 1. Encode Color Channels (0-9)
    for row_idx in range(height):
        for col_idx in range(width):
            color = int(grid[row_idx][col_idx])
            tensor[0, color, row_idx, col_idx] = 1.0

    # 2. Assign Identity Channels (10...10+K) via CCA
    id_slot = COLOR_CHANNELS
    for color in range(1, COLOR_CHANNELS): # Ignore background for initial tracking
        color_mask = tensor[0, color, :height, :width]
        if np.any(color_mask):
            labeled, num_features = label(color_mask)
            for i in range(1, num_features + 1):
                if id_slot >= STATE_CHANNELS:
                    break
                obj_mask = (labeled == i).astype(np.float32)
                # Filter noise: only track objects with area > 0 (all here)
                tensor[0, id_slot, :height, :width] = obj_mask
                id_slot += 1
                
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

    if channel_tensor.shape[0] < COLOR_CHANNELS:
        raise ValueError(f"Expected at least {COLOR_CHANNELS} channels, got {channel_tensor.shape[0]}.")
    
    # Strictly use only the color channels for decoding
    color_tensor = channel_tensor[:COLOR_CHANNELS]

    if output_height < 1 or output_width < 1:
        raise ValueError("output_height and output_width must be positive.")
    if output_height > GRID_SIZE or output_width > GRID_SIZE:
        raise ValueError(f"Requested output shape exceeds {GRID_SIZE}x{GRID_SIZE}.")

    output: list[list[int]] = []
    for row_idx in range(output_height):
        row: list[int] = []
        for col_idx in range(output_width):
            pixel_logits = color_tensor[:, row_idx, col_idx]
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
    """Return a 10-color permutation mapping observed colors to small indices.

    Produces a bijection over the 10 ARC colors:
    - Color 0 always maps to 0.
    - Other colors are assigned in order of first appearance across `grids`.
    - Remaining unseen colors fill the remaining indices in ascending order.
    """
    seen: list[int] = [0]
    seen_set = {0}
    for grid in grids:
        for row in grid:
            for val in row:
                if val not in seen_set:
                    seen.append(val)
                    seen_set.add(val)

    mapping = [0] * 10
    # Assign observed colors to [0..len(seen)-1].
    for new, old in enumerate(seen):
        if 0 <= old < 10:
            mapping[old] = new

    next_new = len(seen)
    for old in range(10):
        if old in seen_set:
            continue
        mapping[old] = next_new
        next_new += 1

    return mapping


def apply_color_map(grid: list[list[int]], mapping: list[int]) -> list[list[int]]:
    """Applies a color mapping to a grid."""
    return [[mapping[val] for val in row] for row in grid]
