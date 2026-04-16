from __future__ import annotations

from dataclasses import dataclass
import contextlib
import math
import os
from pathlib import Path
import tempfile

import onnx
import onnx_tool

EXCLUDED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION"}

@dataclass(frozen=True)
class CostBreakdown:
    parameters: int
    memory_bytes: int
    macs: int

    @property
    def total_cost(self) -> int:
        return self.parameters + self.memory_bytes + self.macs

    @property
    def score(self) -> float:
        return max(1.0, 25.0 - math.log(max(self.total_cost, 1)))


def compute_cost(model_or_path: onnx.ModelProto | str | Path) -> CostBreakdown:
    temp_path: str | None = None
    if isinstance(model_or_path, onnx.ModelProto):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
            temp_path = handle.name
        onnx.save(model_or_path, temp_path)
        load_target: str | Path = temp_path
    else:
        load_target = Path(model_or_path)

    try:
        if isinstance(load_target, Path):
            load_target = load_target.as_posix()
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            profiler = onnx_tool.loadmodel(load_target, {"verbose": False})
            graph = profiler.graph
            graph.graph_reorder_nodes()
            graph.shape_infer(None)
            graph.profile()

        if not graph.valid_profile:
            raise RuntimeError("Invalid model profile; cannot compute cost")

        for node_key in graph.nodemap.keys():
            if graph.nodemap[node_key].op_type.upper() in EXCLUDED_OP_TYPES:
                raise RuntimeError("Model uses excluded operator")

        macs_value = graph.macs
        if isinstance(macs_value, (list, tuple)):
            macs = int(sum(macs_value))
        else:
            macs = int(macs_value)
        memory_bytes = int(graph.memory)
        params = int(graph.params)
        return CostBreakdown(parameters=params, memory_bytes=memory_bytes, macs=macs)
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
