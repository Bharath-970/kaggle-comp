from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.data.encoding import grid_shape, grid_to_tensor, validate_grid


def _im2col_3x3(input_tensor: np.ndarray) -> np.ndarray:
    padded = np.pad(input_tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    patches = []
    for dy in range(3):
        for dx in range(3):
            patch = padded[0, :, dy : dy + 30, dx : dx + 30]
            patches.append(patch.reshape(10, -1))
    stacked = np.concatenate(patches, axis=0)
    return stacked.T


def fit_single_conv3(
    labeled_pairs: Sequence[dict],
    *,
    l2_lambda: float = 1e-2,
    max_samples: int | None = 5000,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not labeled_pairs:
        return None

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for pair in labeled_pairs:
        input_grid = pair.get("input")
        output_grid = pair.get("output")
        if input_grid is None or output_grid is None:
            continue
        try:
            validate_grid(input_grid)
            validate_grid(output_grid)
        except Exception:
            continue

        if grid_shape(input_grid) != grid_shape(output_grid):
            continue

        inputs.append(grid_to_tensor(input_grid).astype(np.float32))
        targets.append(grid_to_tensor(output_grid).astype(np.float32))

    if not inputs:
        return None

    X_blocks = [_im2col_3x3(inp) for inp in inputs]
    X = np.concatenate(X_blocks, axis=0)

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]

    weights = np.zeros((10, 10, 3, 3), dtype=np.float32)
    bias = np.zeros((10,), dtype=np.float32)

    if l2_lambda > 0.0:
        reg = np.sqrt(l2_lambda) * np.eye(X.shape[1], dtype=np.float32)

    for channel in range(10):
        y_blocks = [tgt[0, channel].reshape(-1) for tgt in targets]
        y = np.concatenate(y_blocks, axis=0)
        if max_samples is not None and y.shape[0] > X.shape[0]:
            y = y[idx]

        if l2_lambda > 0.0:
            X_aug = np.vstack([X, reg])
            y_aug = np.concatenate([y, np.zeros(reg.shape[0], dtype=np.float32)])
        else:
            X_aug = X
            y_aug = y

        coeffs, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
        weights[channel] = coeffs.reshape(10, 3, 3)

    return weights, bias
