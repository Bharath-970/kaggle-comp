from __future__ import annotations

import numpy as np


def generate_sparse_weight_matrices(
    *,
    trials: int,
    sparsity: float = 0.9,
    seed: int = 123,
) -> list[np.ndarray]:
    if trials <= 0:
        return []

    rng = np.random.default_rng(seed)
    sparsity = min(max(sparsity, 0.0), 0.99)

    matrices: list[np.ndarray] = []
    for _ in range(trials):
        matrix = rng.normal(loc=0.0, scale=1.0, size=(10, 10)).astype(np.float32)
        mask = rng.random((10, 10)) > sparsity
        sparse = np.where(mask, matrix, 0.0).astype(np.float32)
        matrices.append(sparse)
    return matrices


def mutate_weight_matrix(
    matrix: np.ndarray,
    *,
    mutation_rate: float = 0.05,
    mutation_scale: float = 0.25,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.asarray(matrix, dtype=np.float32).copy()

    mutation_rate = min(max(mutation_rate, 0.0), 1.0)
    mask = rng.random(result.shape) < mutation_rate
    delta = rng.normal(loc=0.0, scale=mutation_scale, size=result.shape).astype(np.float32)
    result[mask] += delta[mask]
    return result
