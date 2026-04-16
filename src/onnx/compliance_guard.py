from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import onnx

BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}
MAX_ONNX_BYTES = 1_440_000


@dataclass
class ComplianceReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    op_types: set[str] = field(default_factory=set)
    file_size_bytes: int | None = None


def _iter_graph_nodes(graph: onnx.GraphProto):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _iter_graph_nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _iter_graph_nodes(subgraph)


def _is_static_shape(value_info: onnx.ValueInfoProto) -> bool:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return False
    if not tensor_type.shape.dim:
        return False
    return all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in tensor_type.shape.dim)


def _shape_issues(model: onnx.ModelProto) -> list[str]:
    issues: list[str] = []
    for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if not _is_static_shape(item):
            issues.append(f"Dynamic or missing shape on tensor: {item.name}")
    for init in model.graph.initializer:
        if any(dim <= 0 for dim in init.dims):
            issues.append(f"Initializer has non-positive dimension: {init.name}")
    return issues


def validate_model(
    model_or_path: onnx.ModelProto | str | Path,
    *,
    max_bytes: int = MAX_ONNX_BYTES,
) -> ComplianceReport:
    if isinstance(model_or_path, onnx.ModelProto):
        model = model_or_path
        path: Path | None = None
    else:
        path = Path(model_or_path)
        model = onnx.load(path.as_posix())

    report = ComplianceReport(ok=True)

    op_types = {node.op_type for node in _iter_graph_nodes(model.graph)}
    report.op_types = op_types

    banned = sorted(op for op in op_types if op in BANNED_OPS)
    if banned:
        report.errors.append(f"Banned operators present: {', '.join(banned)}")

    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # pragma: no cover
        report.errors.append(f"onnx.checker failure: {exc}")

    report.errors.extend(_shape_issues(model))

    if path is not None and path.exists():
        size = path.stat().st_size
        report.file_size_bytes = size
        if size > max_bytes:
            report.errors.append(
                f"Model file size {size} exceeds max {max_bytes} bytes"
            )

    report.ok = len(report.errors) == 0
    return report
