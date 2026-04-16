from __future__ import annotations

import numpy as np

from src.pipeline.task_heuristics import (
    derive_global_color_mapping,
    derive_global_translation,
    mapping_to_weight_matrix,
)


def test_derive_global_color_mapping() -> None:
    payload = [
        {
            "input": [[1, 2], [0, 1]],
            "output": [[3, 4], [0, 3]],
        }
    ]

    mapping = derive_global_color_mapping(payload)

    assert mapping is not None
    assert mapping[1] == 3
    assert mapping[2] == 4
    assert mapping[0] == 0

    matrix = mapping_to_weight_matrix(mapping)
    assert matrix.shape == (10, 10)
    assert matrix[3, 1] == 1.0
    assert matrix[4, 2] == 1.0


def test_derive_global_translation_right_shift() -> None:
    payload = [
        {
            "input": [
                [2, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            "output": [
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        }
    ]

    shift = derive_global_translation(payload, max_shift=2)

    assert shift == (1, 0)
