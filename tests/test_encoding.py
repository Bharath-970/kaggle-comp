from __future__ import annotations

import numpy as np

from src.data.encoding import decode_tensor_to_grid, exact_output_match, grid_to_tensor


def test_grid_roundtrip_decode() -> None:
    grid = [
        [1, 2, 3],
        [4, 0, 9],
    ]
    tensor = grid_to_tensor(grid)
    decoded = decode_tensor_to_grid(tensor, output_height=2, output_width=3)
    assert decoded == grid


def test_exact_output_match_rejects_border_activation() -> None:
    expected = [[7]]
    prediction = grid_to_tensor(expected)
    assert exact_output_match(prediction, expected)

    bad = prediction.copy()
    bad[0, 1, 29, 29] = 1.0
    assert not exact_output_match(bad, expected)
