"""Object detection utilities for ARC grid program synthesis.

All functions operate on raw 2D numpy int arrays (not tensors).
Zero is treated as background by default.

Public API:
    connected_components(grid, connectivity=4) → (labels, n)
    objects_by_color(grid, bg=0) → {color: [ObjectInfo]}
    bounding_box(mask) → (y1, x1, y2, x2) | None
    canonical_object(grid, mask) → np.ndarray  (cropped, bg=0)
    objects_match(a, b) → bool
"""

from __future__ import annotations

import numpy as np
from typing import TypedDict


# ── ObjectInfo type ───────────────────────────────────────────────────────────

class ObjectInfo(TypedDict):
    color: int
    mask: np.ndarray          # bool [H, W] — True where this object lives
    bbox: tuple[int, int, int, int]   # (y1, x1, y2, x2) inclusive
    pixels: list[tuple[int, int]]     # sorted list of (row, col)
    canonical: np.ndarray     # minimal bounding-box crop, bg=0


# ── Core utilities ────────────────────────────────────────────────────────────

def connected_components(
    grid: np.ndarray,
    connectivity: int = 4,
    bg: int = 0,
) -> tuple[np.ndarray, int]:
    """Label connected non-bg components.

    Args:
        grid: 2D int array.
        connectivity: 4 (cross) or 8 (diagonal).
        bg: background colour (ignored for labelling).

    Returns:
        (labels, n) where labels is int32 [H, W], 0 = background/unlabelled,
        1..n = component indices.
    """
    try:
        from scipy.ndimage import label  # type: ignore
    except ImportError:
        raise RuntimeError("scipy required for connected_components. pip install scipy.")

    fg = (grid != bg).astype(np.int32)
    struct = np.ones((3, 3), dtype=np.int32) if connectivity == 8 else None
    labels, n = label(fg, structure=struct)
    return labels.astype(np.int32), int(n)


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return inclusive (y1, x1, y2, x2) of True region, or None if empty."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def canonical_object(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop grid to mask bounding box, zero everything outside mask."""
    bb = bounding_box(mask)
    if bb is None:
        return np.zeros((1, 1), dtype=grid.dtype)
    y1, x1, y2, x2 = bb
    crop = grid[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    crop[~crop_mask] = 0
    return crop


def objects_by_color(
    grid: np.ndarray,
    bg: int = 0,
    connectivity: int = 4,
) -> dict[int, list[ObjectInfo]]:
    """Return all objects grouped by colour.

    Each contiguous blob of a single colour (excluding bg) is one ObjectInfo.

    Example:
        objs = objects_by_color(np.array([[1,0],[0,2]]))
        # → {1: [ObjectInfo(...)], 2: [ObjectInfo(...)]}
    """
    try:
        from scipy.ndimage import label  # type: ignore
    except ImportError:
        raise RuntimeError("scipy required for objects_by_color. pip install scipy.")

    result: dict[int, list[ObjectInfo]] = {}
    struct = np.ones((3, 3), dtype=np.int32) if connectivity == 8 else None

    for color in range(10):
        if color == bg:
            continue
        mask_color = (grid == color)
        if not mask_color.any():
            continue

        labeled, n = label(mask_color, structure=struct)
        result[color] = []
        for i in range(1, n + 1):
            obj_mask = labeled == i
            bb = bounding_box(obj_mask)
            if bb is None:
                continue
            ys, xs = np.where(obj_mask)
            pixels = sorted(zip(ys.tolist(), xs.tolist()))
            result[color].append(
                ObjectInfo(
                    color=color,
                    mask=obj_mask,
                    bbox=bb,
                    pixels=pixels,
                    canonical=canonical_object(grid, obj_mask),
                )
            )

    return result


def objects_match(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two canonical object arrays have the same shape and values."""
    return a.shape == b.shape and bool(np.array_equal(a, b))


# ── Higher-level helpers used by synthesis detectors ─────────────────────────

def all_objects(
    grid: np.ndarray,
    bg: int = 0,
    connectivity: int = 4,
) -> list[ObjectInfo]:
    """Flat list of all objects across all colours, sorted top-left to bottom-right."""
    by_color = objects_by_color(grid, bg=bg, connectivity=connectivity)
    flat: list[ObjectInfo] = []
    for objs in by_color.values():
        flat.extend(objs)
    flat.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
    return flat


def grid_object_count(grid: np.ndarray, bg: int = 0, connectivity: int = 4) -> int:
    """Return total number of distinct connected objects (all colours)."""
    _, n = connected_components(grid, connectivity=connectivity, bg=bg)
    return n


def dominant_bg(grid: np.ndarray) -> int:
    """Return the most frequent colour in the grid (usually background)."""
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts.argmax())
