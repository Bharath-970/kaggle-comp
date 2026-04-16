"""PyTorch -> ONNX export helpers with static-shape validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backbone import RegisterBackbone
from .constants import INPUT_SHAPE
from .onnx_rules import OnnxValidationReport, validate_onnx_file

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export.")


def export_static_onnx(
    model: Any,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
    opset: int = 18,
    run_validation: bool = True,
) -> OnnxValidationReport | None:
    _require_torch()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            external_data=False,
        )

    if not run_validation:
        return None

    return validate_onnx_file(path)


def build_and_export_register_backbone(
    output_path: str | Path,
    steps: int = 3,
    scratch_channels: int = 8,
    mask_channels: int = 2,
    phase_channels: int = 1,
    hidden_channels: int = 32,
    use_coords: bool = True,
    use_depthwise: bool = True,
    opset: int = 18,
) -> OnnxValidationReport | None:
    model = RegisterBackbone(
        steps=steps,
        scratch_channels=scratch_channels,
        mask_channels=mask_channels,
        phase_channels=phase_channels,
        hidden_channels=hidden_channels,
        use_coords=use_coords,
        use_depthwise=use_depthwise,
    )
    return export_static_onnx(model=model, output_path=output_path, opset=opset)
