"""ONNX static-shape and operator compliance checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import BANNED_ONNX_OPS, MAX_ONNX_FILE_BYTES

try:
    import onnx
except Exception:  # pragma: no cover - optional dependency path
    onnx = None


@dataclass(frozen=True)
class OnnxValidationReport:
    file_size_bytes: int
    max_file_size_bytes: int
    banned_ops: tuple[str, ...]
    dynamic_tensors: tuple[str, ...]
    external_data_files: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return (
            self.file_size_bytes <= self.max_file_size_bytes
            and len(self.banned_ops) == 0
            and len(self.dynamic_tensors) == 0
            and len(self.external_data_files) == 0
        )


def _require_onnx() -> None:
    if onnx is None:
        raise RuntimeError("onnx package is required for ONNX validation.")


def _iter_graph_value_infos(model: Any) -> list[Any]:
    graph = model.graph
    return list(graph.input) + list(graph.output) + list(graph.value_info)


def _has_dynamic_shape(value_info: Any) -> bool:
    value_type = value_info.type
    if not value_type.HasField("tensor_type"):
        return False
    tensor_type = value_type.tensor_type
    if not tensor_type.HasField("shape"):
        return True

    for dim in tensor_type.shape.dim:
        has_dim_value = dim.HasField("dim_value") and int(dim.dim_value) > 0
        has_dim_param = dim.HasField("dim_param") and bool(dim.dim_param)
        if has_dim_param or not has_dim_value:
            return True
    return False


def check_file_size(path: str | Path, max_file_size_bytes: int = MAX_ONNX_FILE_BYTES) -> tuple[int, bool]:
    file_size = Path(path).stat().st_size
    return file_size, file_size <= max_file_size_bytes


def find_external_data_files(path: str | Path) -> tuple[str, ...]:
    model_path = Path(path)
    sidecar_files = sorted(p.name for p in model_path.parent.glob(f"{model_path.name}.data*"))
    return tuple(sidecar_files)


def find_banned_ops(model: Any) -> tuple[str, ...]:
    used_ops = {node.op_type for node in model.graph.node}
    banned = sorted(used_ops.intersection(BANNED_ONNX_OPS))
    return tuple(banned)


def find_dynamic_tensors(model: Any) -> tuple[str, ...]:
    dynamic = []
    for value_info in _iter_graph_value_infos(model):
        if _has_dynamic_shape(value_info):
            dynamic.append(value_info.name)
    return tuple(sorted(dynamic))


def validate_onnx_file(path: str | Path) -> OnnxValidationReport:
    _require_onnx()

    model = onnx.load(str(path))
    file_size_bytes, _ = check_file_size(path, MAX_ONNX_FILE_BYTES)
    banned_ops = find_banned_ops(model)
    dynamic_tensors = find_dynamic_tensors(model)
    external_data_files = find_external_data_files(path)

    return OnnxValidationReport(
        file_size_bytes=file_size_bytes,
        max_file_size_bytes=MAX_ONNX_FILE_BYTES,
        banned_ops=banned_ops,
        dynamic_tensors=dynamic_tensors,
        external_data_files=external_data_files,
    )
