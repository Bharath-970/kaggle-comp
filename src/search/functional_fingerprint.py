from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


def generate_probe_tensors(count: int = 4, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    probes: list[np.ndarray] = []

    for _ in range(count):
        colors = rng.integers(0, 10, size=(30, 30), endpoint=False)
        probe = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for row in range(30):
            for col in range(30):
                probe[0, int(colors[row, col]), row, col] = 1.0
        probes.append(probe)

    return probes


def functional_fingerprint(
    model_path: str | Path,
    *,
    probes: Iterable[np.ndarray] | None = None,
) -> str:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required for functional fingerprints") from exc

    session = ort.InferenceSession(Path(model_path).as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    digest = hashlib.sha256()
    for probe in probes if probes is not None else generate_probe_tensors():
        output = session.run(None, {input_name: probe.astype(np.float32)})[0]
        rounded = np.round(output.astype(np.float32), decimals=6)
        digest.update(rounded.tobytes())

    return digest.hexdigest()
