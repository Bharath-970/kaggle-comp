"""PyTorch -> ONNX export helpers with static-shape validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from .backbone import RegisterBackbone
from .constants import COLOR_CHANNELS, GRID_SIZE, IDENTITY_CHANNELS, INPUT_SHAPE, STATE_CHANNELS
from .onnx_rules import OnnxValidationReport, validate_onnx_file

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export.")


class CompetitionIOWrapper(nn.Module):
    """Adapt internal 26x50x50 solver state to NeuroGolf competition I/O.

    Input:  [1, 10, 30, 30]
    Output: [1, 10, 30, 30]

    The wrapped solver still operates on the repo's internal state layout
    ([1, 26, 50, 50]). We pad the visible competition canvas into the top-left
    30x30 region and provide zeroed identity channels.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype

        colors = torch.zeros((batch, COLOR_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=dtype, device=device)
        colors[:, :, :30, :30] = x
        ids = torch.zeros((batch, IDENTITY_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=dtype, device=device)
        state = torch.cat([colors, ids], dim=1)

        out = self.model(state)
        return out[:, :COLOR_CHANNELS, :30, :30]


def export_static_onnx(
    model: Any,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
    opset: int = 18,
    run_validation: bool = True,
    competition_io: bool = False,
) -> OnnxValidationReport | None:
    _require_torch()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    export_model = CompetitionIOWrapper(model) if competition_io else model
    export_model.eval()
    dummy_shape = (1, COLOR_CHANNELS, 30, 30) if competition_io else input_shape
    dummy_input = torch.zeros(dummy_shape, dtype=torch.float32)

    with torch.no_grad():
        # Torch 2.6+ emits a noisy pytree deprecation warning inside export internals.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                category=FutureWarning,
            )
            torch.onnx.export(
                export_model,
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
