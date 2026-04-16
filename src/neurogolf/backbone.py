"""Channel-register backbone for ARC transformations."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None

from .constants import COLOR_CHANNELS

BaseModule = nn.Module if nn is not None else object


class RegisterBackbone(BaseModule):
    """Static-shape backbone using channels as writable memory registers."""

    def __init__(
        self,
        color_channels: int = COLOR_CHANNELS,
        scratch_channels: int = 8,
        mask_channels: int = 2,
        phase_channels: int = 1,
        hidden_channels: int = 32,
        steps: int = 3,
        use_coords: bool = True,
        use_depthwise: bool = True,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to instantiate RegisterBackbone.")

        super().__init__()
        if steps < 1:
            raise ValueError("steps must be >= 1")

        self.color_channels = color_channels
        self.scratch_channels = scratch_channels
        self.mask_channels = mask_channels
        self.phase_channels = phase_channels
        self.steps = steps
        self.use_coords = use_coords
        self.use_depthwise = use_depthwise

        # Coordinate channels (x, y)
        in_channels = color_channels + (2 if use_coords else 0)
        self.total_register_channels = color_channels + scratch_channels + mask_channels + phase_channels

        self.input_projection = nn.Conv2d(in_channels, self.total_register_channels, kernel_size=1)

        self.update_blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(steps):
            groups = self.total_register_channels if use_depthwise else 1
            self.update_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.total_register_channels,
                        self.total_register_channels,
                        kernel_size=3,
                        padding=1,
                        groups=groups,
                        bias=False,
                    ),
                    nn.Conv2d(self.total_register_channels, hidden_channels, kernel_size=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(hidden_channels, self.total_register_channels, kernel_size=1),
                )
            )
            self.gates.append(nn.Conv2d(self.total_register_channels, self.total_register_channels, kernel_size=1))

        self.readout = nn.Conv2d(self.total_register_channels, color_channels, kernel_size=1)

    def _get_coords(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        y_range = torch.linspace(0.0, 1.0, h, device=x.device)
        x_range = torch.linspace(0.0, 1.0, w, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return coords

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if torch is None:
            raise RuntimeError("PyTorch is required to run RegisterBackbone.forward().")
        if input_tensor.ndim != 4:
            raise ValueError("Expected [N,C,H,W] input tensor.")

        if self.use_coords:
            coords = self._get_coords(input_tensor)
            input_tensor = torch.cat([input_tensor, coords], dim=1)

        state = self.input_projection(input_tensor)
        for update_block, gate_layer in zip(self.update_blocks, self.gates):
            update = update_block(state)
            gate = torch.sigmoid(gate_layer(state))
            state = state + (gate * update)

        return self.readout(state)


def count_trainable_parameters(model: Any) -> int:
    if not hasattr(model, "parameters"):
        return 0
    return int(sum(param.numel() for param in model.parameters() if getattr(param, "requires_grad", False)))
