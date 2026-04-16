from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any

import onnx

from src.primitives.library import build_primitive_model


@dataclass(frozen=True)
class CandidateBlueprint:
    candidate_id: str
    family: str
    primitive_name: str
    metadata: dict[str, Any]


_COUNTER = count(1)


def _next_candidate_id(task_id: str) -> str:
    return f"{task_id}-cand-{next(_COUNTER):05d}"


def propose_minimal_blueprints(
    task_id: str,
    ranked_families: list[tuple[str, float]],
    *,
    per_family: int = 2,
) -> list[CandidateBlueprint]:
    primitive_map = {
        "color_map": ("pointwise_color_map",),
        "symmetry": ("identity", "pointwise_color_map"),
        "object": ("pointwise_color_map",),
        "composition": ("identity", "pointwise_color_map"),
        "counting": ("pointwise_color_map",),
        "spatial_program": ("pointwise_color_map",),
    }

    blueprints: list[CandidateBlueprint] = []
    for family, confidence in ranked_families:
        primitives = primitive_map.get(family, ("identity",))
        for _ in range(max(1, per_family)):
            primitive_name = primitives[_ % len(primitives)]
            blueprints.append(
                CandidateBlueprint(
                    candidate_id=_next_candidate_id(task_id),
                    family=family,
                    primitive_name=primitive_name,
                    metadata={"family_confidence": confidence},
                )
            )
    return blueprints


def build_candidate_model(blueprint: CandidateBlueprint) -> onnx.ModelProto:
    model_name = blueprint.candidate_id.replace("_", "-")
    return build_primitive_model(blueprint.primitive_name, model_name=model_name)
