from __future__ import annotations

import numpy as np

from src.eval.cost import compute_cost
from src.onnx.compliance_guard import validate_model
from src.onnx.native_builder import build_depthwise_shift_model, build_shift_then_color_map_model


def test_depthwise_shift_model_compliant() -> None:
    model = build_depthwise_shift_model(dx=1, dy=0, model_name="shift_right")

    report = validate_model(model)
    cost = compute_cost(model)

    assert report.ok
    assert "Conv" in report.op_types
    assert cost.parameters == 90
    assert cost.memory_bytes > 0


def test_shift_then_color_map_model_compliant() -> None:
    matrix = np.eye(10, dtype=np.float32)
    model = build_shift_then_color_map_model(dx=0, dy=1, weight_matrix=matrix)

    report = validate_model(model)
    cost = compute_cost(model)

    assert report.ok
    assert cost.parameters == 190
    assert cost.memory_bytes > 0
    assert cost.macs > 0
