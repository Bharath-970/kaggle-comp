from __future__ import annotations

import numpy as np

from src.eval.cost import compute_cost
from src.onnx.compliance_guard import validate_model
from src.onnx.native_builder import (
    build_maxpool_model,
    build_minpool_model,
    build_maxpool_then_color_map_model,
    build_minpool_then_color_map_model,
)


def test_pool_models_compliant() -> None:
    max_model = build_maxpool_model(3)
    min_model = build_minpool_model(3)

    max_report = validate_model(max_model)
    min_report = validate_model(min_model)

    assert max_report.ok
    assert min_report.ok
    assert "MaxPool" in max_report.op_types
    assert "MaxPool" in min_report.op_types


def test_pool_color_map_compliant() -> None:
    weights = np.eye(10, dtype=np.float32)

    max_model = build_maxpool_then_color_map_model(3, weights)
    min_model = build_minpool_then_color_map_model(3, weights)

    max_report = validate_model(max_model)
    min_report = validate_model(min_model)

    assert max_report.ok
    assert min_report.ok

    max_cost = compute_cost(max_model)
    min_cost = compute_cost(min_model)

    assert max_cost.total_cost > 0
    assert min_cost.total_cost > 0
