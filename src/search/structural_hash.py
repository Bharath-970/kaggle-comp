from __future__ import annotations

import hashlib

import onnx

from src.search.canonicalize import canonical_graph_json


def structural_hash(model: onnx.ModelProto) -> str:
    canonical = canonical_graph_json(model)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
