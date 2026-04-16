"""Comprehensive library of zero-parameter ARC solvers for parameter golf."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

from .constants import GRID_SIZE
from .grid_codec import encode_grid_to_tensor


class IdentitySolver(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RotationSolver(nn.Module):
    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=self.k, dims=(2, 3))


class FlipSolver(nn.Module):
    def __init__(self, horizontal: bool = True) -> None:
        super().__init__()
        self.dims = (2,) if horizontal else (3,)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=self.dims)


class TransposeSolver(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3)


class ShiftSolver(nn.Module):
    def __init__(self, dx: int, dy: int) -> None:
        super().__init__()
        self.dx = dx
        self.dy = dy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ARC shifts treat out-of-bounds pixels as clear, not wrapped.
        shifted = x

        if self.dy > 0:
            shifted = F.pad(shifted[:, :, :-self.dy, :], (0, 0, self.dy, 0))
        elif self.dy < 0:
            up = -self.dy
            shifted = F.pad(shifted[:, :, up:, :], (0, 0, 0, up))

        if self.dx > 0:
            shifted = F.pad(shifted[:, :, :, :-self.dx], (self.dx, 0, 0, 0))
        elif self.dx < 0:
            left = -self.dx
            shifted = F.pad(shifted[:, :, :, left:], (0, left, 0, 0))

        return shifted


class GeneralColorRemapSolver(nn.Module):
    """Input->output color remap, represented as channel permutation."""

    def __init__(self, input_to_output: list[int]) -> None:
        super().__init__()
        if len(input_to_output) != 10:
            raise ValueError("Expected 10-channel color map.")

        output_to_input = list(range(10))
        for in_color, out_color in enumerate(input_to_output):
            output_to_input[out_color] = in_color
        self.perm = output_to_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm, :, :]


class SubgridSolver(nn.Module):
    def __init__(self, y1: int, y2: int, x1: int, x2: int) -> None:
        super().__init__()
        self.bounds = (y1, y2, x1, x2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sub = x[:, :, self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]]
        h, w = sub.shape[2], sub.shape[3]
        return F.pad(sub, (0, GRID_SIZE - w, 0, GRID_SIZE - h))


class TilingSolver(nn.Module):
    """Repeats only the active (uh, uw) portion of the input."""
    def __init__(self, uh: int, uw: int, repeats_h: int, repeats_w: int) -> None:
        super().__init__()
        self.uh = uh
        self.uw = uw
        self.repeats = (repeats_h, repeats_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [1, 10, 30, 30]
        # Extract the unit
        unit = x[:, :, :self.uh, :self.uw]
        # Repeat it
        out = unit.repeat(1, 1, self.repeats[0], self.repeats[1])
        # Pad back to 30x30
        h, w = out.shape[2], out.shape[3]
        return F.pad(out, (0, GRID_SIZE - w, 0, GRID_SIZE - h))


class NearestNeighborScaleSolver(nn.Module):
    """Upscales the active top-left input region by integer factors."""

    def __init__(self, in_h: int, in_w: int, scale_h: int, scale_w: int) -> None:
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.scale_h = scale_h
        self.scale_w = scale_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unit = x[:, :, :self.in_h, :self.in_w]
        out = unit.repeat_interleave(self.scale_h, dim=2).repeat_interleave(self.scale_w, dim=3)
        h, w = out.shape[2], out.shape[3]
        return F.pad(out, (0, GRID_SIZE - w, 0, GRID_SIZE - h))


class KroneckerSolver(nn.Module):
    """
    Implements recursive ARC Kronecker product.
    Typically: Out = Kron(Unit, Unit) where Unit is a 3x3 subgrid of the input.
    """
    def __init__(self, uh: int = 3, uw: int = 3) -> None:
        super().__init__()
        self.uh = uh
        self.uw = uw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract the unit (template and kernel) from the top-left
        unit = x[:, :, :self.uh, :self.uw]
        
        # Binary mask from unit (1 where any color is present)
        mask = torch.sum(unit[:, 1:, :, :], dim=1, keepdim=True).clamp(0, 1)
        
        # Recursive Kronecker: Expand mask using unit
        # [1, 1, h, w] x [1, 10, h, w] -> [1, 10, h*h, w*w]
        # We use a nested loop or reshape trick for ONNX compatibility
        batch, channels, h, w = unit.shape
        # [1, 1, h, 1, w, 1] * [1, 10, 1, h, 1, w] -> [1, 10, h, h, w, w]
        out = mask.unsqueeze(3).unsqueeze(5) * unit.unsqueeze(2).unsqueeze(4)
        out = out.reshape(batch, channels, h*h, w*w)
        
        # Pad back to 30x30
        h2, w2 = out.shape[2], out.shape[3]
        return F.pad(out, (0, GRID_SIZE - w2, 0, GRID_SIZE - h2))


class ConstantGridSolver(nn.Module):
    """Always emits the same grid regardless of input."""

    def __init__(self, grid: list[list[int]]) -> None:
        super().__init__()
        template = torch.from_numpy(encode_grid_to_tensor(grid))
        self.register_buffer("template", template)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return self.template
        return self.template.expand(x.shape[0], -1, -1, -1)


class ColorNormalizedSolver(nn.Module):
    """
    Wraps a backbone to make it color-invariant.
    Performs normalization and de-normalization using a task-specific map.
    """
    def __init__(self, backbone: nn.Module, color_map: list[int]) -> None:
        super().__init__()
        self.backbone = backbone
        # color_map[old] = new.
        # For channel indexing, normalization needs read_perm[new] = old.
        inv_map = [0] * 10
        for old, new in enumerate(color_map):
            inv_map[new] = old
        
        self.norm_perm = inv_map
        self.denorm_perm = color_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [1, 10, 30, 30]
        # Normalize: Permute channels
        x_norm = x[:, self.norm_perm, :, :]
        # Backbone logic
        out_norm = self.backbone(x_norm)
        # Denormalize: Permute channels back
        return out_norm[:, self.denorm_perm, :, :]


class CompositionalSolver(nn.Sequential):
    """Sequence of primitives."""
    pass
