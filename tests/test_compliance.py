from __future__ import annotations

from pathlib import Path

from src.onnx.compliance_guard import validate_model
from src.onnx.native_builder import build_identity_model, save_model


def test_identity_model_is_compliant(tmp_path: Path) -> None:
    model = build_identity_model()
    path = save_model(model, tmp_path / "identity.onnx")

    report = validate_model(path)

    assert report.ok
    assert "Identity" in report.op_types
    assert not report.errors
