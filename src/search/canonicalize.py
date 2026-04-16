from __future__ import annotations

import hashlib
import json
from typing import Any

import onnx
from onnx import numpy_helper


def _attribute_signature(attribute: onnx.AttributeProto) -> tuple[str, Any]:
    attr_type = attribute.type
    if attr_type == onnx.AttributeProto.INT:
        value: Any = int(attribute.i)
    elif attr_type == onnx.AttributeProto.FLOAT:
        value = float(attribute.f)
    elif attr_type == onnx.AttributeProto.STRING:
        value = attribute.s.decode("utf-8", errors="ignore")
    elif attr_type == onnx.AttributeProto.INTS:
        value = tuple(int(v) for v in attribute.ints)
    elif attr_type == onnx.AttributeProto.FLOATS:
        value = tuple(float(v) for v in attribute.floats)
    elif attr_type == onnx.AttributeProto.STRINGS:
        value = tuple(v.decode("utf-8", errors="ignore") for v in attribute.strings)
    elif attr_type == onnx.AttributeProto.TENSOR:
        tensor = numpy_helper.to_array(attribute.t)
        value = {
            "shape": tuple(int(v) for v in tensor.shape),
            "dtype": str(tensor.dtype),
            "hash": hashlib.sha256(tensor.tobytes()).hexdigest(),
        }
    else:
        value = f"unsupported_attr_type_{int(attr_type)}"
    return attribute.name, value


def canonical_graph_signature(model: onnx.ModelProto) -> dict[str, Any]:
    node_sigs: list[dict[str, Any]] = []
    for node in model.graph.node:
        attrs = sorted((_attribute_signature(attr) for attr in node.attribute), key=lambda kv: kv[0])
        node_sigs.append(
            {
                "op_type": node.op_type,
                "inputs": tuple(node.input),
                "outputs": tuple(node.output),
                "attributes": attrs,
            }
        )

    node_sigs.sort(key=lambda item: (item["op_type"], item["inputs"], item["outputs"]))

    initializers = []
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        initializers.append(
            {
                "name": init.name,
                "shape": tuple(int(v) for v in array.shape),
                "dtype": str(array.dtype),
                "hash": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    initializers.sort(key=lambda item: item["name"])

    return {
        "ir_version": int(model.ir_version),
        "nodes": node_sigs,
        "initializers": initializers,
        "inputs": sorted((inp.name, tuple(dim.dim_value for dim in inp.type.tensor_type.shape.dim)) for inp in model.graph.input),
        "outputs": sorted((out.name, tuple(dim.dim_value for dim in out.type.tensor_type.shape.dim)) for out in model.graph.output),
    }


def canonical_graph_json(model: onnx.ModelProto) -> str:
    return json.dumps(canonical_graph_signature(model), sort_keys=True, separators=(",", ":"))
