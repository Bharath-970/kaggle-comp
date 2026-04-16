from __future__ import annotations

import numpy as np

from src.eval.cost import compute_cost
from src.onnx.native_builder import build_pointwise_color_map_model


def test_pointwise_color_map_cost_breakdown() -> None:
    weights = np.eye(10, dtype=np.float32)
    model = build_pointwise_color_map_model(weights)

    cost = compute_cost(model)

    assert cost.parameters > 0
    assert cost.memory_bytes > 0
    assert cost.macs > 0
    assert cost.total_cost > 0
    assert cost.score > 1.0
