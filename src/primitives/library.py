from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx

from src.onnx.native_builder import build_identity_model, build_pointwise_color_map_model


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    capabilities: tuple[str, ...]
    min_nodes: int
    description: str


def default_primitive_specs() -> tuple[PrimitiveSpec, ...]:
    return (
        PrimitiveSpec(
            name="identity",
            capabilities=("shape_preserving", "color_preserving"),
            min_nodes=1,
            description="Pass-through baseline for shape-preserving tasks.",
        ),
        PrimitiveSpec(
            name="pointwise_color_map",
            capabilities=("shape_preserving", "color_mapping"),
            min_nodes=1,
            description="1x1 channel mixing for color remapping and channel logic.",
        ),
    )


def build_primitive_model(name: str, **kwargs: Any) -> onnx.ModelProto:
    if name == "identity":
        return build_identity_model(model_name=kwargs.get("model_name", "identity"))

    if name == "pointwise_color_map":
        weights = kwargs.get("weight_matrix")
        if weights is None:
            weights = np.eye(10, dtype=np.float32)
        bias = kwargs.get("bias")
        return build_pointwise_color_map_model(
            weights,
            bias=bias,
            model_name=kwargs.get("model_name", "pointwise_color_map"),
        )

    raise ValueError(f"Unknown primitive: {name}")
