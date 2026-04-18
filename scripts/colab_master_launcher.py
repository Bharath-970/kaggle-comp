from __future__ import annotations
# @title NeuroGolf 2026: ULTIMATE HYBRID LAUNCHER (Target 5000+)
# @markdown Environment: A100 or T4 GPU recommended.
# Final verified version with Color-Invariance and Zero-Parameter Symbolic Sweeps.

import os, sys, json, time, random, zipfile, shutil, warnings
print("✅ NeuroGolf SOTA Engine Initializing...")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterable, Literal, Sequence

# --- 1. SETUP ---
os.makedirs("neurogolf", exist_ok=True)
os.makedirs("artifacts/submission", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Install dependencies
try: 
    import onnx, onnxruntime as ort, scipy.ndimage, onnxscript
except: 
    os.system(f"{sys.executable} -m pip install -q onnx onnxruntime scipy onnxscript")

# --- 2. THE CORE ENGINE ---

# --- constants.py ---
"""Project-wide constants for ARC tensor and ONNX validation."""

GRID_SIZE = 30
COLOR_CHANNELS = 10
BATCH_DIM = 1
INPUT_SHAPE = (BATCH_DIM, COLOR_CHANNELS, GRID_SIZE, GRID_SIZE)

# Competition states 1.44MB max ONNX file size.
MAX_ONNX_FILE_BYTES = 1_440_000

BANNED_ONNX_OPS = frozenset(
    {
        "Loop",
        "Scan",
        "NonZero",
        "Unique",
        "Script",
        "Function",
    }
)


# --- grid_codec.py ---
"""ARC grid <-> tensor conversion helpers."""



from typing import Sequence

import numpy as np




def _validate_grid(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not grid:
        raise ValueError("Grid cannot be empty.")

    height = len(grid)
    width = len(grid[0])
    if width == 0:
        raise ValueError("Grid rows cannot be empty.")

    if height > GRID_SIZE or width > GRID_SIZE:
        raise ValueError(f"Grid shape {height}x{width} exceeds {GRID_SIZE}x{GRID_SIZE}.")

    for row in grid:
        if len(row) != width:
            raise ValueError("Grid rows must have equal width.")
        for value in row:
            if value < 0 or value >= COLOR_CHANNELS:
                raise ValueError(f"Color index out of range: {value}")

    return height, width


def encode_grid_to_tensor(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Encode an ARC grid into [1, 10, 30, 30] one-hot tensor."""
    height, width = _validate_grid(grid)

    tensor = np.zeros((1, COLOR_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for row_idx in range(height):
        for col_idx in range(width):
            color = int(grid[row_idx][col_idx])
            tensor[0, color, row_idx, col_idx] = 1.0

    return tensor


def decode_tensor_to_grid(tensor: np.ndarray, output_height: int, output_width: int, strict: bool = False) -> list[list[int]]:
    """Decode tensor logits/probabilities back to color grid with fixed output shape."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected batch size 1.")
        channel_tensor = tensor[0]
    elif tensor.ndim == 3:
        channel_tensor = tensor
    else:
        raise ValueError("Tensor must have shape [1,C,H,W] or [C,H,W].")

    if channel_tensor.shape[0] != COLOR_CHANNELS:
        raise ValueError(f"Expected {COLOR_CHANNELS} channels, got {channel_tensor.shape[0]}.")

    if output_height < 1 or output_width < 1:
        raise ValueError("output_height and output_width must be positive.")
    if output_height > GRID_SIZE or output_width > GRID_SIZE:
        raise ValueError("Requested output shape exceeds 30x30.")

    output: list[list[int]] = []
    for row_idx in range(output_height):
        row: list[int] = []
        for col_idx in range(output_width):
            pixel_logits = channel_tensor[:, row_idx, col_idx]
            if strict:
                non_zero = int(np.count_nonzero(pixel_logits > 0.5))
                if non_zero != 1:
                    raise ValueError(
                        f"Strict decode expected one active channel at ({row_idx}, {col_idx}), got {non_zero}."
                    )
            row.append(int(np.argmax(pixel_logits)))
        output.append(row)

    return output


def get_color_normalization_map(grids: list[list[list[int]]]) -> list[int]:
    """Return a 10-color permutation mapping observed colors to small indices.

    Produces a bijection over the 10 ARC colors:
    - Color 0 always maps to 0.
    - Other colors are assigned in order of first appearance across `grids`.
    - Remaining unseen colors fill the remaining indices in ascending order.
    """
    seen: list[int] = [0]
    seen_set = {0}
    for grid in grids:
        for row in grid:
            for val in row:
                if val not in seen_set:
                    seen.append(val)
                    seen_set.add(val)

    mapping = [0] * 10
    # Assign observed colors to [0..len(seen)-1].
    for new, old in enumerate(seen):
        if 0 <= old < 10:
            mapping[old] = new

    next_new = len(seen)
    for old in range(10):
        if old in seen_set:
            continue
        mapping[old] = next_new
        next_new += 1

    return mapping


def apply_color_map(grid: list[list[int]], mapping: list[int]) -> list[list[int]]:
    """Applies a color mapping to a grid."""
    return [[mapping[val] for val in row] for row in grid]


# --- solvers.py ---
"""Comprehensive library of zero-parameter ARC solvers for parameter golf."""



try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None




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
            shifted = F.pad(shifted[:, :, : -self.dy, :], (0, 0, self.dy, 0))
        elif self.dy < 0:
            up = -self.dy
            shifted = F.pad(shifted[:, :, up:, :], (0, 0, 0, up))

        if self.dx > 0:
            shifted = F.pad(shifted[:, :, :, : -self.dx], (self.dx, 0, 0, 0))
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
        sub = x[:, :, self.bounds[0] : self.bounds[1], self.bounds[2] : self.bounds[3]]
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
        unit = x[:, :, : self.uh, : self.uw]
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
        unit = x[:, :, : self.in_h, : self.in_w]
        out = unit.repeat_interleave(self.scale_h, dim=2).repeat_interleave(
            self.scale_w, dim=3
        )
        # Pad back to 30x30
        h, w = out.shape[2], out.shape[3]
        return F.pad(out, (0, GRID_SIZE - w, 0, GRID_SIZE - h))

class GravitySolver(nn.Module):
    """Moves all non-zero pixels as far as possible in a direction until they hit a boundary."""
    def __init__(self, direction: str = "down") -> None:
        super().__init__()
        self.direction = direction.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        grid = torch.argmax(x, dim=1).squeeze(0).cpu().numpy()
        h, w = grid.shape
        new_grid = np.zeros_like(grid)
        if self.direction == "down":
            for c in range(w):
                particles = [grid[r, c] for r in range(h) if grid[r, c] != 0]
                for i, color in enumerate(reversed(particles)):
                    new_grid[h - 1 - i, c] = color
        elif self.direction == "up":
            for c in range(w):
                particles = [grid[r, c] for r in range(h) if grid[r, c] != 0]
                for i, color in enumerate(particles):
                    new_grid[i, c] = color
        elif self.direction == "right":
            for r in range(h):
                particles = [grid[r, c] for c in range(w) if grid[r, c] != 0]
                for i, color in enumerate(reversed(particles)):
                    new_grid[r, w - 1 - i] = color
        elif self.direction == "left":
            for r in range(h):
                particles = [grid[r, c] for c in range(w) if grid[r, c] != 0]
                for i, color in enumerate(particles):
                    new_grid[r, i] = color
        out_tensor = torch.zeros((1, 10, GRID_SIZE, GRID_SIZE), device=device)
        for r in range(h):
            for c in range(w):
                out_tensor[0, int(new_grid[r, c]), r, c] = 1.0
        return out_tensor

class CropToColorSolver(nn.Module):
    """Crops the grid to the bounding box of a specific color mask."""
    def __init__(self, target_color: int) -> None:
        super().__init__()
        self.target_color = target_color

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[0, self.target_color] > 0.5
        coords = torch.nonzero(mask)
        if coords.shape[0] == 0: return x
        y1, x1 = coords.min(dim=0).values
        y2, x2 = coords.max(dim=0).values
        sub = x[:, :, y1 : y2 + 1, x1 : x2 + 1]
        h, w = sub.shape[2], sub.shape[3]
        return F.pad(sub, (0, GRID_SIZE - w, 0, GRID_SIZE - h))
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
        unit = x[:, :, : self.uh, : self.uw]

        # Binary mask from unit (1 where any color is present)
        mask = torch.sum(unit[:, 1:, :, :], dim=1, keepdim=True).clamp(0, 1)

        # Recursive Kronecker: Expand mask using unit
        # [1, 1, h, w] x [1, 10, h, w] -> [1, 10, h*h, w*w]
        # We use a nested loop or reshape trick for ONNX compatibility
        batch, channels, h, w = unit.shape
        # [1, 1, h, 1, w, 1] * [1, 10, 1, h, 1, w] -> [1, 10, h, h, w, w]
        out = mask.unsqueeze(3).unsqueeze(5) * unit.unsqueeze(2).unsqueeze(4)
        out = out.reshape(batch, channels, h * h, w * w)

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


class CyclicSequenceSolver(nn.Module):
    """Fills grid using a cyclic sequence with optional row/column shifts."""
    def __init__(self, sequence, dx=0, dy=0):
        super().__init__()
        self.seq = nn.Parameter(torch.tensor(sequence, dtype=torch.long), requires_grad=False)
        self.dx, self.dy = dx, dy
    def forward(self, x):
        # x: [B, 10, H, W]
        b, _, h, w = x.shape
        device = x.device
        coords_h = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
        coords_w = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w)
        
        # pattern[i, j] = seq[(i * dy + j * dx) % len(seq)]
        idx = (coords_h * self.dy + coords_w * self.dx) % len(self.seq)
        out = self.seq[idx]
        return F.one_hot(out, 10).permute(0, 3, 1, 2).float()


class CompositionalSolver(nn.Sequential):
    """Sequence of primitives."""

    pass


class ErodeSolver(nn.Module):
    """Binary erosion via min-pool (negative max-pool)."""

    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.iterations):
            x = -F.max_pool2d(-x, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return x


class DilateSolver(nn.Module):
    """Binary dilation via max-pool."""

    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.iterations):
            x = F.max_pool2d(x, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return x


class ColorQuantizeSolver(nn.Module):
    """Reduce color palette via color remap."""

    def __init__(self, target_palette_size: int = 4):
        super().__init__()
        self.target_palette_size = target_palette_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape
        x_flat = x.view(batch, channels, -1)
        x_max = x_flat.max(dim=2, keepdim=True)[0]
        x_quantized = (x_flat / (x_max + 1e-8) * (self.target_palette_size - 1)).round()
        return x_quantized.view(batch, channels, h, w)


class PatternRepeatSolver(nn.Module):
    """Tile/mirror a pattern."""

    def __init__(self, tile_h: int = 2, tile_w: int = 2, mode: str = "tile"):
        super().__init__()
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.mode = mode  # "tile" or "mirror"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mirror":
            x = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)
            x = torch.cat([x, torch.flip(x, dims=[-2])], dim=-2)
        else:
            x = x.repeat(1, 1, self.tile_h, self.tile_w)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# New fundamental solver types (ONNX-safe, no Loop/Scan/NonZero/dynamic shapes)
# ─────────────────────────────────────────────────────────────────────────────


class FoldOverlaySolver(nn.Module):
    """Fold the active grid region onto itself and merge.

    Handles variable grid sizes dynamically via a baked (grid_h, grid_w),
    which is the MAXIMUM expected size across all task pairs. The flip is
    applied to the [0:grid_h, 0:grid_w] subgrid and embedded back.

    Different task pairs may have different sizes — we bake the per-task
    max so the ONNX graph is static.

    mode='or_overlay'  : flipped non-bg fills bg cells in original.
    mode='replace_bg'  : flipped value replaces wherever original is bg.
    flip_dim=2 → flip vertical (flipud), flip_dim=3 → flip horizontal (fliplr).
    """

    def __init__(self, flip_dim: int, mode: str, bg_color: int = 0,
                 grid_h: int = GRID_SIZE, grid_w: int = GRID_SIZE) -> None:
        super().__init__()
        if flip_dim not in (2, 3):
            raise ValueError("flip_dim must be 2 (H) or 3 (W)")
        if mode not in ("or_overlay", "replace_bg"):
            raise ValueError("mode must be 'or_overlay' or 'replace_bg'")
        self.flip_dim = flip_dim
        self.mode = mode
        self.bg_color = bg_color
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Build row & col "alive" vectors: 1 for indices within the grid, 0 outside
        # These are static masks for ONNX
        r_mask = torch.zeros(1, 1, GRID_SIZE, 1)
        r_mask[:, :, :grid_h, :] = 1.0
        self.register_buffer("row_alive", r_mask)   # [1,1,30,1]

        c_mask = torch.zeros(1, 1, 1, GRID_SIZE)
        c_mask[:, :, :, :grid_w] = 1.0
        self.register_buffer("col_alive", c_mask)   # [1,1,1,30]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build static active mask [1, 1, 30, 30] from baked row/col alive vectors
        active_mask = self.row_alive * self.col_alive  # [1, 1, 30, 30]

        # To flip only the active subgrid:
        #   1. Zero-out padding so flipping doesn't drag padding content inside
        x_active = x * active_mask                   # [1, 10, 30, 30], zeros outside

        #   2. Flip the whole tensor (active content flips within its region)
        if self.flip_dim == 2:
            # Vertical flip of active region: row i ↔ row (grid_h-1-i)
            # After flipping whole 30-row tensor: active content is at rows (30-grid_h)..(29)
            # We need to re-align it to rows 0..(grid_h-1) by rolling it up
            flipped_full = torch.flip(x_active, dims=[2])   # content at bottom
            # Roll up by (GRID_SIZE - grid_h) to bring content to row 0
            roll_amount = self.grid_h - GRID_SIZE           # negative = roll up
            flipped = torch.roll(flipped_full, shifts=roll_amount, dims=2)
        else:
            flipped_full = torch.flip(x_active, dims=[3])   # content at right
            roll_amount = self.grid_w - GRID_SIZE
            flipped = torch.roll(flipped_full, shifts=roll_amount, dims=3)

        # Zero out anything that rolled back into the padding zone
        flipped = flipped * active_mask

        bg_ch = x[:, self.bg_color : self.bg_color + 1, :, :]
        is_bg = (bg_ch > 0.5).float() * active_mask

        if self.mode == "or_overlay":
            return x + is_bg * (flipped - x * is_bg)
        else:  # replace_bg
            return x * (1.0 - is_bg) + flipped * is_bg




class DiagonalPeriodicTilingSolver(nn.Module):
    """Tile a diagonal color sequence periodically across the whole grid.

    output[r][c] = color at phase (r + c) % period derived from the input's diagonal.
    Stores (grid_h, grid_w) so it operates only within the active region.

    ONNX-safe: only ReduceSum, element-wise multiply, and buffer arithmetic.
    """

    def __init__(self, period: int,
                 grid_h: int = GRID_SIZE, grid_w: int = GRID_SIZE,
                 bg_color: int = 0) -> None:
        super().__init__()
        self.period = period
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.bg_color = bg_color

        # Pre-build phase masks: phase_masks[k, 0, r, c] = 1 iff (r+c)%period == k
        # Only within the active region
        masks = torch.zeros(period, 1, GRID_SIZE, GRID_SIZE)
        for r in range(grid_h):
            for c in range(grid_w):
                masks[(r + c) % period, 0, r, c] = 1.0
        self.register_buffer("phase_masks", masks)  # [period, 1, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 10, 30, 30]
        # We only look at the non-background channels for phase color detection
        # Build x_fg: zero out the bg channel so bg cells don't pollute phase sums
        x_fg = x.clone()
        x_fg[:, self.bg_color : self.bg_color + 1, :, :] = 0.0

        output = torch.zeros_like(x)
        for k in range(self.period):
            mask_k = self.phase_masks[k : k + 1]  # [1, 1, H, W]
            # Sum non-bg channels over phase-k positions → which color is active there?
            color_sum = (x_fg * mask_k).sum(dim=(2, 3), keepdim=True)  # [1, 10, 1, 1]
            # The winning color channel will have the largest sum
            # Broadcast back to fill all phase-k positions
            output = output + color_sum * mask_k
        return output


class GravitySolver(nn.Module):
    """Simulate pixel gravity: non-background pixels fall in a fixed direction.

    Uses an unrolled bubble-sort via adjacent comparisons (ONNX-safe, no loops in graph).
    direction: 'up', 'down', 'left', 'right'
    bg_color: the color treated as empty space (pixels fall through it).
    iterations: number of bubble-sort passes; GRID_SIZE is sufficient.
    """

    def __init__(self, direction: str, bg_color: int = 0, iterations: int = 30) -> None:
        super().__init__()
        if direction not in ("up", "down", "left", "right"):
            raise ValueError(f"Unknown direction: {direction}")
        self.direction = direction
        self.bg_color = bg_color
        self.iterations = iterations

    def _swap_step(self, x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        """One bubble-sort pass: swap adjacent channel stacks if out of order."""
        # For 'down': roll content upward in H dim (shift= -1) and check if swap improves
        # bg channel is self.bg_color; non-bg should be below bg
        bg = x[:, self.bg_color : self.bg_color + 1, :, :]

        # shifted: look at the cell in the direction of travel
        shifted_bg = torch.roll(bg, shifts=shift, dims=dim)
        # Crop the roll artefact at the border (set border to 0 so no swap there)
        if dim == 2:
            if shift == 1:
                shifted_bg[:, :, 0, :] = 0.0
            else:
                shifted_bg[:, :, -1, :] = 0.0
        else:
            if shift == 1:
                shifted_bg[:, :, :, 0] = 0.0
            else:
                shifted_bg[:, :, :, -1] = 0.0

        # swap_mask: 1 where (current cell is bg) AND (neighbour moving here is non-bg)
        swap_mask = ((bg > 0.5) & (shifted_bg < 0.5)).float()  # [1, 1, H, W]

        # Swap: exchange content at these positions with their neighbours
        # Cell that receives non-bg (swap_mask is 1): gets shifted_x
        shifted_x = torch.roll(x, shifts=shift, dims=dim)
        
        # Cell that gives non-bg (it will receive bg): that is the neighbour cell!
        # Find which cells gave their non-bg:
        reverse_swap = torch.roll(swap_mask, shifts=-shift, dims=dim)
        reverse_x = torch.roll(x, shifts=-shift, dims=dim)

        # Apply both sides of the swap
        # 1) If cell received non-bg, it gets shifted_x
        # 2) Else If cell gave non-bg, it gets reverse_x (the bg cell)
        # 3) Else keep original x
        new_x = torch.where(swap_mask > 0.5, shifted_x, 
                            torch.where(reverse_swap > 0.5, reverse_x, x))

        # Zero the edge artefact
        if dim == 2:
            if shift == 1:
                new_x[:, :, 0, :] = x[:, :, 0, :]
            else:
                new_x[:, :, -1, :] = x[:, :, -1, :]
        else:
            if shift == 1:
                new_x[:, :, :, 0] = x[:, :, :, 0]
            else:
                new_x[:, :, :, -1] = x[:, :, :, -1]

        return new_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direction → (shift, dim) for bubble-sort passes.
        # Negative shift means old index i moves to new index i-1 (moves UP/LEFT).
        # Positive shift means old index i moves to new index i+1 (moves DOWN/RIGHT).
        direction_params = {
            "down":  (1,  2),
            "up":    (-1, 2),
            "right": (1,  3),
            "left":  (-1, 3),
        }
        shift, dim = direction_params[self.direction]

        out = x
        for _ in range(self.iterations):
            out = self._swap_step(out, shift, dim)
        return out


class FloodFillSolver(nn.Module):
    """Fill enclosed background regions with a fixed colour.

    Algorithm (ONNX-safe, no Loop/NonZero):
      1. Dynamically identify active region using cumsum.
      2. Spread background-connectivity FROM the active border inward using max-pool dilations.
      3. Enclosed bg = total bg - border-reachable bg.
      4. Replace enclosed bg cells with `fill_color` channel active.

    Works for tasks where a single fixed colour fills ALL enclosed holes.
    """

    def __init__(
        self,
        fill_color: int,
        bg_color: int = 0,
        iterations: int = 60,
    ) -> None:
        super().__init__()
        self.fill_color = fill_color
        self.bg_color = bg_color
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bg = x[:, self.bg_color : self.bg_color + 1, :, :]  # [1,1,30,30]

        # Dynamically detect active region: cells that sum to >0 across all channels
        # Since encode_grid_to_tensor uses one-hot colors, sum of channels = 1 for active cells and 0 for padding.
        active_mask = x.sum(dim=1, keepdim=True)  # [1,1,30,30]

        # Build dynamic border: edge rows and cols of the active region
        # An active cell is on the border if any neighbour (or itself) is non-active
        outside_active = 1.0 - active_mask
        # Pad with 1.0 (non-active) so cells touching top/left of 30x30 are correctly marked as border
        padded_outside = F.pad(outside_active, (1, 1, 1, 1), value=1.0)
        reached_outside = F.max_pool2d(padded_outside, kernel_size=3, stride=1, padding=0)
        border_mask = active_mask * (reached_outside > 0.5).float()

        # Seed: bg cells on the border of the active region
        external = bg * border_mask

        # Flood-fill through bg cells using repeated 3×3 max-pool
        for _ in range(self.iterations):
            spread = F.max_pool2d(external, kernel_size=3, stride=1, padding=1)
            external = bg * spread  # can only spread through bg

        # Enclosed bg = in active region, is bg, NOT externally reachable
        enclosed = bg * (1.0 - external) * active_mask

        # Rebuild output: ONNX-safe (no clone): reconstruct each channel
        not_enclosed = 1.0 - enclosed
        bg_out = bg * not_enclosed  # remove bg from enclosed positions
        fill_ch = x[:, self.fill_color : self.fill_color + 1, :, :]
        fill_out = fill_ch * not_enclosed + enclosed  # add fill at enclosed

        # Reconstruct full 10-channel tensor
        channels = []
        for c in range(10):
            if c == self.bg_color:
                channels.append(bg_out)
            elif c == self.fill_color:
                channels.append(fill_out)
            else:
                channels.append(x[:, c : c + 1, :, :])
        return torch.cat(channels, dim=1)


class IsolatedPixelCrossSolver(nn.Module):
    """Each pixel that is the sole non-bg occupant of its row OR column
    extends to fill that entire row AND column with its colour.

    Detection rule used during synthesis (not in forward):
      - If a non-bg pixel is isolated in its row (row count == 1): fill the row.
      - If a non-bg pixel is isolated in its column (col count == 1): fill the col.
      - A pixel may fill both if isolated on both axes simultaneously.

    ONNX-safe: only linear algebra + broadcast (no NonZero/gather on dynamic index).

    The architecture:
      For each color channel c != bg:
        row_active[c, r]  = 1  if the row r has exactly one non-bg pixel of color c
        col_active[c, c_] = 1  if the col c_ has exactly one non-bg pixel of color c
      Then: out[c, r, col] = max(original, row_active[c,r] * isolated_in_row,
                                            col_active[c,col] * isolated_in_col)
    """

    def __init__(self, bg_color: int = 0) -> None:
        super().__init__()
        self.bg_color = bg_color

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 10, 30, 30]
        out = x.clone()

        for c in range(10):
            if c == self.bg_color:
                continue
            ch = x[:, c : c + 1, :, :]  # [1, 1, 30, 30]

            row_sum = ch.sum(dim=3, keepdim=True)   # [1, 1, 30, 1] — count per row
            col_sum = ch.sum(dim=2, keepdim=True)   # [1, 1, 1, 30] — count per col

            # A row is "isolated" for this colour if exactly 1 pixel present
            row_isolated = (row_sum > 0.5) * (row_sum < 1.5)  # [1,1,30,1]
            col_isolated = (col_sum > 0.5) * (col_sum < 1.5)  # [1,1,1,30]

            # Extend: any isolated row gets that channel set to 1 across full row
            row_fill = row_isolated.expand_as(ch)   # broadcast across col axis
            col_fill = col_isolated.expand_as(ch)   # broadcast across row axis

            extended = (ch + row_fill + col_fill).clamp(0.0, 1.0)

            # Only clear bg where we've filled
            filled_mask = (extended - ch).clamp(0.0, 1.0)
            out[:, self.bg_color : self.bg_color + 1, :, :] = (
                out[:, self.bg_color : self.bg_color + 1, :, :] * (1.0 - filled_mask)
            )
            out[:, c : c + 1, :, :] = extended

        return out


# --- backbone.py ---
"""Channel-register backbone for ARC transformations."""



from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None



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


# --- ensemble.py ---
"""ONNX-friendly ensembling of multiple model branches."""



try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


class EnsembleSolver(nn.Module):
    """
    Combines multiple models into a single ONNX graph.
    Performs 'Mean' ensembling on probabilities (logits).
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        return out

class ConsensusEnsembleSolver(nn.Module):
    """
    Performs pixel-wise majority voting across branches.
    More robust against noise than Mean ensembling.
    """
    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get argmax for each branch
        branch_outputs = []
        for b in self.branches:
            branch_outputs.append(torch.argmax(b(x), dim=1))
        
        # Stack indices: [N_BRANCH, B, 30, 30]
        stacked = torch.stack(branch_outputs, dim=0)
        
        # One-hot vote aggregation
        one_hots = torch.nn.functional.one_hot(stacked, 10) # [N, B, 30, 30, 10]
        votes = one_hots.sum(dim=0).float() # [B, 30, 30, 10]
        
        # Re-permute to output standard [B, 10, 30, 30] logits
        return votes.permute(0, 3, 1, 2)


# --- task_io.py ---
"""Load ARC task JSON files with lightweight validation."""



from dataclasses import dataclass
from pathlib import Path
import json




@dataclass(frozen=True)
class GridPair:
    input_grid: list[list[int]]
    output_grid: list[list[int]]


@dataclass(frozen=True)
class TaskData:
    train: tuple[GridPair, ...]
    test: tuple[GridPair, ...]
    arc_gen: tuple[GridPair, ...]


def _parse_pairs(raw_pairs: list[dict]) -> tuple[GridPair, ...]:
    parsed: list[GridPair] = []
    for raw in raw_pairs:
        input_grid = raw["input"]
        output_grid = raw["output"]

        # Validate by attempting encoding against fixed competition tensor shape.
        encode_grid_to_tensor(input_grid)
        encode_grid_to_tensor(output_grid)

        parsed.append(GridPair(input_grid=input_grid, output_grid=output_grid))

    return tuple(parsed)


def load_task_json(task_path: str | Path) -> TaskData:
    path = Path(task_path)
    payload = json.loads(path.read_text())

    train_pairs = _parse_pairs(payload.get("train", []))
    test_pairs = _parse_pairs(payload.get("test", []))
    arc_gen_pairs = _parse_pairs(payload.get("arc-gen", []))

    return TaskData(train=train_pairs, test=test_pairs, arc_gen=arc_gen_pairs)


# --- search_solvers.py ---
#!/usr/bin/env python3
"""Hybrid symbolic + neural fallback solver search for NeuroGolf tasks."""



import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Literal

import torch
import numpy as np

# Paths set relative to working directory for Colab
ROOT = Path.cwd()
SRC = ROOT






def _grid_is_supported(grid: list[list[int]]) -> bool:
    if not grid or not grid[0]:
        return False
    return len(grid) <= GRID_SIZE and len(grid[0]) <= GRID_SIZE


def load_task_json_relaxed(task_path: str | Path) -> tuple[TaskData, int]:
    """Load task, dropping oversize arc-gen examples while keeping train/test strict."""
    try:
        return load_task_json(task_path), 0
    except ValueError:
        payload = json.loads(Path(task_path).read_text())

    dropped = 0

    def parse_split(split: str, strict: bool) -> tuple[GridPair, ...]:
        nonlocal dropped
        parsed: list[GridPair] = []
        for raw in payload.get(split, []):
            input_grid = raw["input"]
            output_grid = raw["output"]
            if _grid_is_supported(input_grid) and _grid_is_supported(output_grid):
                parsed.append(GridPair(input_grid=input_grid, output_grid=output_grid))
            else:
                dropped += 1
                if strict:
                    raise ValueError(f"{split} contains unsupported grid size > {GRID_SIZE}x{GRID_SIZE}.")
        return tuple(parsed)

    task = TaskData(
        train=parse_split("train", strict=True),
        test=parse_split("test", strict=True),
        arc_gen=parse_split("arc-gen", strict=False),
    )
    return task, dropped


def _iter_all_pairs(task: TaskData, *, include_arc_gen: bool = True) -> Iterable[GridPair]:
    for pair in task.train:
        yield pair
    for pair in task.test:
        yield pair
    if include_arc_gen:
        for pair in task.arc_gen:
            yield pair


def _iter_search_pairs(task: TaskData) -> Iterable[GridPair]:
    # Cheap candidate generation: ARC tasks have tiny train/test, huge arc-gen.
    for pair in task.train:
        yield pair
    for pair in task.test:
        yield pair


def check_solve(model: torch.nn.Module, task: TaskData, *, include_arc_gen: bool = True, train_only: bool = False) -> bool:
    model.eval()
    with torch.no_grad():
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        
        pairs_to_check = list(task.train)
        if not train_only:
            if include_arc_gen:
                pairs_to_check.extend(task.arc_gen)
            pairs_to_check.extend(task.test)
            
        for pair in pairs_to_check:
            in_tensor = torch.from_numpy(encode_grid_to_tensor(pair.input_grid)).to(device)
            pred_tensor = model(in_tensor).detach().cpu().numpy()
            expected = pair.output_grid
            pred = decode_tensor_to_grid(pred_tensor, len(expected), len(expected[0]))
            if pred != expected:
                return False
    return True


@dataclass(frozen=True)
class TaskFitScore:
    exact_pairs: int
    total_pairs: int
    pixel_accuracy: float


@dataclass(frozen=True)
class FallbackProfile:
    tier: Literal["easy", "medium", "hard"]
    n_models: int
    epochs: int
    hidden_channels: int
    steps: int
    batch_size: int
    promotion_n_models: int
    promotion_epochs: int
    promotion_hidden_channels: int
    promotion_steps: int
    promotion_pixel_threshold: float


def _grid_entropy_norm(grid: list[list[int]]) -> float:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr.ravel(), minlength=COLOR_CHANNELS).astype(np.float64)
    probs = counts[counts > 0] / float(arr.size)
    entropy = -np.sum(probs * np.log2(probs)) if probs.size else 0.0
    return float(entropy / np.log2(COLOR_CHANNELS))


def _grid_symmetry_score(grid: list[list[int]]) -> float:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size == 0:
        return 0.0
    checks = [
        float(np.array_equal(arr, np.flipud(arr))),
        float(np.array_equal(arr, np.fliplr(arr))),
    ]
    if arr.shape[0] == arr.shape[1]:
        checks.append(float(np.array_equal(arr, arr.T)))
    return float(sum(checks) / len(checks))


def _task_complexity_score(task: TaskData) -> float:
    grids = [pair.input_grid for pair in task.train]
    if not grids:
        return 0.5

    areas = []
    entropies = []
    symmetries = []
    for grid in grids:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        areas.append(min(1.0, float(h * w) / float(GRID_SIZE * GRID_SIZE)))
        entropies.append(_grid_entropy_norm(grid))
        symmetries.append(_grid_symmetry_score(grid))

    area_score = float(np.mean(areas))
    entropy_score = float(np.mean(entropies))
    asymmetry_score = 1.0 - float(np.mean(symmetries))
    few_shot_score = 1.0 - min(len(task.train), 4) / 4.0

    # Weighted complexity proxy in [0,1]: larger, noisier, asymmetric, low-shot tasks are harder.
    return max(
        0.0,
        min(
            1.0,
            0.35 * area_score
            + 0.35 * entropy_score
            + 0.20 * asymmetry_score
            + 0.10 * few_shot_score,
        ),
    )


def _build_fallback_profile(task: TaskData) -> FallbackProfile:
    complexity = _task_complexity_score(task)
    if complexity < 0.34:
        return FallbackProfile(
            tier="easy",
            n_models=1,
            epochs=180,
            hidden_channels=12,
            steps=4,
            batch_size=16,
            promotion_n_models=2,
            promotion_epochs=320,
            promotion_hidden_channels=18,
            promotion_steps=5,
            promotion_pixel_threshold=0.74,
        )
    if complexity < 0.62:
        return FallbackProfile(
            tier="medium",
            n_models=1,
            epochs=280,
            hidden_channels=16,
            steps=5,
            batch_size=12,
            promotion_n_models=2,
            promotion_epochs=480,
            promotion_hidden_channels=22,
            promotion_steps=6,
            promotion_pixel_threshold=0.70,
        )
    return FallbackProfile(
        tier="hard",
        n_models=2,
        epochs=520,
        hidden_channels=24,
        steps=6,
        batch_size=8,
        promotion_n_models=3,
        promotion_epochs=760,
        promotion_hidden_channels=30,
        promotion_steps=7,
        promotion_pixel_threshold=0.66,
    )


def _poe_rank_score(name: str, candidate: TaskFitScore, identity: TaskFitScore) -> float:
    exact_ratio = candidate.exact_pairs / max(1, candidate.total_pairs)
    fit = max(0.0, min(1.0, candidate.pixel_accuracy))
    improvement = max(0.0, fit - identity.pixel_accuracy)
    family_confidence = 1.0 if "mean_ensemble" in name else 0.96 if "consensus" in name else 0.92

    # Product-of-experts style ranking favors exactness first, then fit, then robust improvement.
    return (
        max(exact_ratio, 0.01) ** 0.55
        * max(fit, 0.01) ** 0.35
        * max(improvement + 0.05 * family_confidence, 0.01) ** 0.10
    )


def score_task_fit(
    model: torch.nn.Module,
    task: TaskData,
    *,
    include_arc_gen: bool = False,
    train_only: bool = True,
) -> TaskFitScore:
    model.eval()
    with torch.no_grad():
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        pairs_to_check = list(task.train)
        if not train_only:
            if include_arc_gen:
                pairs_to_check.extend(task.arc_gen)
            pairs_to_check.extend(task.test)

        exact_pairs = 0
        pixel_correct = 0
        pixel_total = 0
        for pair in pairs_to_check:
            expected = pair.output_grid
            in_tensor = torch.from_numpy(encode_grid_to_tensor(pair.input_grid)).to(device)
            pred_tensor = model(in_tensor).detach().cpu().numpy()
            pred = decode_tensor_to_grid(pred_tensor, len(expected), len(expected[0]))

            expected_arr = np.array(expected, dtype=np.int64)
            pred_arr = np.array(pred, dtype=np.int64)
            same = pred_arr == expected_arr
            if bool(np.all(same)):
                exact_pairs += 1
            pixel_correct += int(np.count_nonzero(same))
            pixel_total += int(same.size)

    pixel_accuracy = (pixel_correct / pixel_total) if pixel_total else 0.0
    return TaskFitScore(
        exact_pairs=exact_pairs,
        total_pairs=len(pairs_to_check),
        pixel_accuracy=pixel_accuracy,
    )


def _all_pairs_same_shape(task: TaskData) -> tuple[tuple[int, int], tuple[int, int]] | None:
    pairs = list(_iter_all_pairs(task))
    if not pairs:
        return None
    in_h, in_w = len(pairs[0].input_grid), len(pairs[0].input_grid[0])
    out_h, out_w = len(pairs[0].output_grid), len(pairs[0].output_grid[0])
    for pair in pairs[1:]:
        if (len(pair.input_grid), len(pair.input_grid[0])) != (in_h, in_w):
            return None
        if (len(pair.output_grid), len(pair.output_grid[0])) != (out_h, out_w):
            return None
    return (in_h, in_w), (out_h, out_w)


def _derive_color_map_constraints(
    src_grid: list[list[int]],
    dst_grid: list[list[int]],
    input_to_output: dict[int, int],
    output_to_input: dict[int, int],
) -> bool:
    if len(src_grid) != len(dst_grid) or len(src_grid[0]) != len(dst_grid[0]):
        return False
    for r in range(len(src_grid)):
        for c in range(len(src_grid[0])):
            in_color = src_grid[r][c]
            out_color = dst_grid[r][c]

            mapped_out = input_to_output.get(in_color)
            if mapped_out is None:
                input_to_output[in_color] = out_color
            elif mapped_out != out_color:
                return False

            mapped_in = output_to_input.get(out_color)
            if mapped_in is None:
                output_to_input[out_color] = in_color
            elif mapped_in != in_color:
                return False
    return True


def _finalize_color_map(input_to_output: dict[int, int]) -> list[int]:
    color_map = list(range(10))
    for in_color, out_color in input_to_output.items():
        if 0 <= in_color < 10 and 0 <= out_color < 10:
            color_map[in_color] = out_color
    return color_map


def _apply_shift_zero_padded(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = arr.shape
    out = np.zeros((h, w), dtype=arr.dtype)
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)  # exclusive
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)  # exclusive
    dst_y1 = max(0, dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    if src_y2 > src_y1 and src_x2 > src_x1:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return out


def _grid_transform(
    grid: list[list[int]],
    op: Literal["identity", "transpose", "rot90", "rot180", "rot270", "flip_h", "flip_v", "shift"],
    *,
    dx: int = 0,
    dy: int = 0,
) -> list[list[int]]:
    arr = np.array(grid, dtype=np.int64)
    if op == "identity":
        out = arr
    elif op == "transpose":
        out = arr.T
    elif op == "rot90":
        out = np.rot90(arr, k=1)
    elif op == "rot180":
        out = np.rot90(arr, k=2)
    elif op == "rot270":
        out = np.rot90(arr, k=3)
    elif op == "flip_h":
        out = np.fliplr(arr)
    elif op == "flip_v":
        out = np.flipud(arr)
    elif op == "shift":
        out = _apply_shift_zero_padded(arr, dy=dy, dx=dx)
    else:
        raise ValueError(f"Unknown op: {op}")
    return out.tolist()


def _derive_global_color_map_for_grid_transform(
    pairs: Iterable[GridPair],
    op: Literal["identity", "transpose", "rot90", "rot180", "rot270", "flip_h", "flip_v", "shift"],
    *,
    dx: int = 0,
    dy: int = 0,
) -> list[int] | None:
    input_to_output: dict[int, int] = {}
    output_to_input: dict[int, int] = {}
    for pair in pairs:
        transformed = _grid_transform(pair.input_grid, op, dx=dx, dy=dy)
        ok = _derive_color_map_constraints(transformed, pair.output_grid, input_to_output, output_to_input)
        if not ok:
            return None
    return _finalize_color_map(input_to_output)


def _wrap_with_color_map_if_match(
    base_model: torch.nn.Module,
    task: TaskData,
) -> torch.nn.Module | None:
    # Fallback path: derive map by running the model. This is slower, but keeps compatibility
    # for transforms we don't have a pure-grid implementation for.
    input_to_output: dict[int, int] = {}
    output_to_input: dict[int, int] = {}
    with torch.no_grad():
        try:
            device = next(base_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        for pair in _iter_all_pairs(task):
            expected = pair.output_grid
            in_tensor = torch.from_numpy(encode_grid_to_tensor(pair.input_grid)).to(device)
            pred_tensor = base_model(in_tensor).detach().cpu().numpy()
            pred_grid = decode_tensor_to_grid(pred_tensor, len(expected), len(expected[0]))
            ok = _derive_color_map_constraints(pred_grid, expected, input_to_output, output_to_input)
            if not ok:
                return None
    return CompositionalSolver(base_model, GeneralColorRemapSolver(_finalize_color_map(input_to_output)))


def _candidate_geometries(max_shift: int) -> list[torch.nn.Module]:
    geoms: list[torch.nn.Module] = [IdentitySolver(), TransposeSolver()]
    geoms.extend(RotationSolver(k=k) for k in (1, 2, 3))
    geoms.append(FlipSolver(horizontal=True))
    geoms.append(FlipSolver(horizontal=False))
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            geoms.append(ShiftSolver(dx=dx, dy=dy))
    return geoms


def _all_outputs_identical(task: TaskData) -> bool:
    all_pairs = list(_iter_all_pairs(task))
    if not all_pairs:
        return False
    first = all_pairs[0].output_grid
    return all(pair.output_grid == first for pair in all_pairs[1:])


def _try_enclosed_fill(task: TaskData) -> torch.nn.Module | None:
    """Attempts to solve by filling enclosed regions with a color.
    Uses color-invariance to handle example-specific palettes.
    """
    from scipy import ndimage
    
    # Primitives for filling
    class EnclosedFill(torch.nn.Module):
        def __init__(self, b, f):
            super().__init__()
            self.b, self.f = b, f
        def forward(self, x):
            grid = torch.argmax(x, dim=1)
            border = (grid == self.b).float().unsqueeze(1)
            outside = torch.ones_like(border)
            outside[:, :, 1:-1, 1:-1] = 0
            outside = outside * (1 - border)
            for _ in range(64): 
                dilated = torch.nn.functional.max_pool2d(outside, 3, stride=1, padding=1)
                outside = dilated * (1 - border)
            inside = (1 - border) * (1 - outside)
            out = grid.clone()
            out[inside.squeeze(1).bool()] = self.f
            return torch.nn.functional.one_hot(out.long(), 10).permute(0, 3, 1, 2).float()

    first_in = np.array(task.train[0].input_grid)
    first_out = np.array(task.train[0].output_grid)
    in_colors = np.unique(first_in)
    out_colors = np.unique(first_out)
    
    for b in in_colors:
        for f in out_colors:
            if b == f: continue
            candidate = EnclosedFill(int(b), int(f))
            model = _wrap_with_color_map_if_match(candidate, task)
            if model and check_solve(model, task):
                return model
    return None

def _try_periodic_tiling(task: TaskData) -> torch.nn.Module | None:
    """Detects if input follows a repeating row/column sequence and extends it."""
    
    # 1. Discover the mapping and the period
    best_ph, best_pw = None, None
    global_c_map = None

    for pair in task.train:
        in_g = np.array(pair.input_grid)
        out_g = np.array(pair.output_grid)
        ih, iw = in_g.shape
        oh, ow = out_g.shape
        
        # We need to find a period T and a color map M such that:
        # out_g[i, j] = M(in_g[i % T_h, j % T_w]) -- wait, that's not quite right for Task 003.
        # In Task 003: out_g[i, j] = M(pattern[i % T_h, j % T_w])
        # And the input is just the first few rows of that pattern.
        
        # Let's find T_h such that in_g[i] == in_g[i + T_h]
        ph = None
        for t in range(1, ih):
            match = True
            for i in range(ih - t):
                if not np.array_equal(in_g[i], in_g[i + t]):
                    match = False; break
            if match:
                ph = t; break
        
        pw = None
        for t in range(1, iw):
            match = True
            for j in range(iw - t):
                if not np.array_equal(in_g[:, j], in_g[:, j + t]):
                    match = False; break
            if match:
                pw = t; break
        
        # If no period found, we assume the whole input is the period
        if ph is None: ph = ih
        if pw is None: pw = iw
        
        # Check if output matches this period extended
        # Create a candidate by tiling the 'period' from the input
        unit = in_g[:ph, :pw]
        
        # Find color map M
        map_candidate = {}
        possible = True
        for i in range(min(ih, oh)):
            for j in range(min(iw, ow)):
                ic, oc = in_g[i, j], out_g[i, j]
                if ic in map_candidate and map_candidate[ic] != oc:
                    # Color swap must be consistent globally
                    possible = False; break
                map_candidate[ic] = oc
            if not possible: break
        
        if possible:
            # Check if this map and period explain the WHOLE output
            # Fill missing colors with identity
            full_map = list(range(10))
            for k, v in map_candidate.items(): full_map[k] = v
            
            mapped_unit = np.array(full_map)[unit]
            reps_h = oh // ph + 1
            reps_w = ow // pw + 1
            candidate_out = np.tile(mapped_unit, (reps_h, reps_w))[:oh, :ow]
            
            if np.array_equal(candidate_out, out_g):
                global_c_map = full_map
                best_ph, best_pw = ph, pw
                # Success for this pair. Continue to verify next pairs.
            else:
                # Try one more thing: shifted tiling if first column is background
                possible = False
        else:
            possible = False
            
        if not possible: return None
    
    if global_c_map is None: return None

    class TilingModel(torch.nn.Module):
        def __init__(self, ph, pw, th, tw, c_map):
            super().__init__()
            self.ph, self.pw = ph, pw
            self.th, self.tw = th, tw
            self.register_buffer("c_map", torch.tensor(c_map).long())

        def forward(self, x):
            # x: [B, 10, H, W]
            grid = torch.argmax(x, dim=1) # [B, H, W]
            # Take the prefix period
            unit = grid[:, :self.ph, :self.pw]
            # Apply color map
            mapped_unit = self.c_map[unit]
            # Tile
            # Torch doesn't have a simple tile-to-size, so we use repeat and crop
            reps_h = (self.th // self.ph) + 1
            reps_w = (self.tw // self.pw) + 1
            tiled = mapped_unit.repeat(1, reps_h, reps_w)
            out = tiled[:, :self.th, :self.tw]
            return torch.nn.functional.one_hot(out, 10).permute(0, 3, 1, 2).float()

    return TilingModel(best_ph, best_pw, task.train[0].output_grid.shape[0], task.train[0].output_grid.shape[1], global_c_map)

def _try_global_shift(task: TaskData) -> torch.nn.Module | None:
    """Checks if output is a pixel shift of input (padded with 0)."""
    from scipy.ndimage import shift
    for dy in range(-10, 11):
        for dx in range(-10, 11):
            if dx == 0 and dy == 0: continue
            match = True
            for pair in task.train:
                in_g = np.array(pair.input_grid)
                out_g = np.array(pair.output_grid)
                if in_g.shape != out_g.shape:
                    match = False; break
                
                # mode='constant' with cval=0 is standard ARC behavior
                shifted = shift(in_g, (dy, dx), mode='constant', cval=0)
                if not np.array_equal(shifted, out_g):
                    match = False; break
            
            if match:
                return ShiftSolver(dx=dx, dy=dy)
    return None

def _try_gravity(task: TaskData) -> torch.nn.Module | None:
    """Checks if objects fall toward a boundary by simulating gravity in all 4 directions."""
    for direction in ["down", "up", "left", "right"]:
        solver = GravitySolver(direction=direction)
        if check_solve(solver, task, train_only=True):
            return solver
    return None

def _try_subgrid_cropping(task: TaskData) -> torch.nn.Module | None:
    """Checks if output matches a subgrid defined by a specific input color's bounding box."""
    # Try all colors as potential 'mask colors'
    all_colors = set()
    for p in task.train:
        all_colors.update(np.unique(p.input_grid))
    
    for color in all_colors:
        if color == 0: continue
        solver = CropToColorSolver(target_color=int(color))
        # This solver might return correct shape but wrong content if used alone
        # We check if it matches the train outputs after a possible color remap
        candidate = _wrap_with_color_map_if_match(solver, task)
        if candidate is not None and check_solve(candidate, task):
            return candidate
    return None

def _try_scaling(task: TaskData) -> torch.nn.Module | None:
    ref = task.train[0]
    ih, iw = np.array(ref.input_grid).shape
    oh, ow = np.array(ref.output_grid).shape
    if oh % ih == 0 and ow % iw == 0 and oh > ih:
        model = ScalingModel(oh // ih, ow // iw, oh, ow)
        if check_solve(model, task): return model
    return None

class KroneckerModel(torch.nn.Module):
    """Out = Kron(Kernel, Template). Used for fractal patterns."""
    def __init__(self, kh, kw, th, tw):
        super().__init__()
        self.kh, self.kw = kh, kw
        self.th, self.tw = th, tw

    def forward(self, x):
        grid = torch.argmax(x, dim=1) # [B, H, W]
        kernel = grid[:, :self.kh, :self.kw]
        b, h, w = grid.shape
        # Simplified Kronecker logic for ARC
        out = kernel.unsqueeze(1).unsqueeze(2) * (grid.unsqueeze(3).unsqueeze(4) > 0).float()
        out = out.permute(0, 1, 3, 2, 4).reshape(b, h*self.kh, w*self.kw)
        return torch.nn.functional.one_hot(out.long(), 10).permute(0, 3, 1, 2).float()

def _try_kronecker(task: TaskData) -> torch.nn.Module | None:
    for kh in [2, 3, 4, 5]:
        for kw in [kh]: 
            ref = task.train[0]
            ih, iw = np.array(ref.input_grid).shape
            oh, ow = np.array(ref.output_grid).shape
            if oh == ih * kh and ow == iw * kw:
                model = KroneckerModel(kh, kw, oh, ow) 
                if check_solve(model, task): return model
    return None

def _try_boundary(task: TaskData) -> torch.nn.Module | None:
    """Checks if output is input with an added frame/boundary."""
    for c in range(10):
        model = BoundaryModel(c, mode="grid")
        if check_solve(model, task): return model
    return None

def _try_symmetry(task: TaskData) -> torch.nn.Module | None:
    """Checks if output is a flip or rotation of input."""
    ops = [
        (lambda x: x, "identity"),
        (lambda x: np.flipud(x), "flip_v"),
        (lambda x: np.fliplr(x), "flip_h"),
        (lambda x: np.rot90(x, 1), "rot90"),
        (lambda x: np.rot90(x, 2), "rot180"),
        (lambda x: np.rot90(x, 3), "rot270"),
        (lambda x: x.T, "transpose"),
    ]
    
    for op, name in ops:
        match = True
        for pair in task.train:
            try:
                res = op(np.array(pair.input_grid))
                if not np.array_equal(res, np.array(pair.output_grid)):
                    match = False; break
            except:
                match = False; break
        if match:
            class SymModel(torch.nn.Module):
                def __init__(self, mode):
                    super().__init__()
                    self.mode = mode
                def forward(self, x):
                    if self.mode == "flip_v": return torch.flip(x, [2])
                    if self.mode == "flip_h": return torch.flip(x, [3])
                    if self.mode == "rot90": return torch.rot90(x, 1, [2, 3])
                    if self.mode == "rot180": return torch.rot90(x, 2, [2, 3])
                    if self.mode == "rot270": return torch.rot90(x, 3, [2, 3])
                    if self.mode == "transpose": return x.transpose(2, 3)
                    return x
            return SymModel(name)
    return None

class IdentitySolver(torch.nn.Module):
    """Fallback solver that returns the input grid as-is."""
    def forward(self, input): # Rename x to input to match ARC standard feed
        return input

class KroneckerModel(torch.nn.Module):
    """Out = Kron(Kernel, Template). Used for fractal patterns."""
    def __init__(self, kh, kw, th, tw):
        super().__init__()
        self.kh, self.kw = kh, kw
        self.th, self.tw = th, tw

    def forward(self, x):
        grid = torch.argmax(x, dim=1) # [B, H, W]
        # Kernel is the top-left kh x kw
        kernel = grid[:, :self.kh, :self.kw]
        # Template is the following th x tw
        # (This is a simplified Kronecker heuristic, often kernel == template in ARC)
        # We'll use the WHOLE input as a template if small enough
        mask = (kernel > 0).float().unsqueeze(3).unsqueeze(5) # [B, kh, kw, 1, 1, 1]
        # For ARC: Output[i*kh:i*kh+kh, j*kw:j*kw+kw] = Kernel if Template[i,j] > 0
        # Actually simplified: each non-zero pixel in input becomes a 'kernel' in output
        # [B, 1, 1, kh, kw] * [B, in_h, in_w, 1, 1]
        b, h, w = grid.shape
        # Result grid size [b, h*kh, w*kw]
        out = kernel.unsqueeze(1).unsqueeze(2) * (grid.unsqueeze(3).unsqueeze(4) > 0).float()
        out = out.permute(0, 1, 3, 2, 4).reshape(b, h*self.kh, w*self.kw)
        return torch.nn.functional.one_hot(out.long(), 10).permute(0, 3, 1, 2).float()

class ScalingModel(torch.nn.Module):
    """Integer factor upscaling (Nearest Neighbor)."""
    def __init__(self, sh, sw, target_h, target_w):
        super().__init__()
        self.sh, self.sw = sh, sw
        self.th, self.tw = target_h, target_w
    def forward(self, x):
        grid = torch.argmax(x, dim=1)
        out = grid.repeat_interleave(self.sh, dim=1).repeat_interleave(self.sw, dim=2)
        out = out[:, :self.th, :self.tw]
        return torch.nn.functional.one_hot(out.long(), 10).permute(0, 3, 1, 2).float()

class BoundaryModel(torch.nn.Module):
    """Draws a boundary/frame around objects or the whole grid."""
    def __init__(self, color, mode="grid"):
        super().__init__()
        self.color = color
        self.mode = mode
    def forward(self, x):
        grid = torch.argmax(x, dim=1) # [B, H, W]
        b, h, w = grid.shape
        out = grid.clone()
        if self.mode == "grid":
            out[:, 0, :] = self.color
            out[:, -1, :] = self.color
            out[:, :, 0] = self.color
            out[:, :, -1] = self.color
        return torch.nn.functional.one_hot(out.long(), 10).permute(0, 3, 1, 2).float()

def get_grid_symmetries(grid: np.ndarray):
    """Returns all 8 D4 symmetries of a grid."""
    syms = []
    curr = grid
    for _ in range(4):
        syms.append(curr)
        syms.append(np.fliplr(curr))
        curr = np.rot90(curr)
    return syms

def get_task_dna(task: TaskData):
    """Extracts a structural fingerprint (DNA) of the task."""
    ref = task.train[0]
    in_g, out_g = np.array(ref.input_grid), np.array(ref.output_grid)
    ih, iw = in_g.shape
    oh, ow = out_g.shape
    
    ratio = (oh / ih, ow / iw)
    color_delta = len(np.unique(out_g)) - len(np.unique(in_g))
    
    # Check for simple scaling
    is_scaling = (ratio[0] == int(ratio[0]) and ratio[1] == int(ratio[1]) and ratio[0] > 1)
    
    # Check for periodicity
    has_period = False
    for t in range(1, ih // 2 + 1):
        if ih % t == 0 and np.array_equal(np.tile(in_g[:t], (ih//t, 1)), in_g):
            has_period = True; break
            
    return {
        "ratio": ratio,
        "color_delta": color_delta,
        "is_scaling": is_scaling,
        "has_period": has_period
    }

def find_master_synthesis(task: TaskData, max_shift: int = 2) -> torch.nn.Module | None:
    if not task.train:
        return None

    # 1. SCAN TASK DNA
    dna = get_task_dna(task)
    
    # 2. PRIORITIZED SOLVER DISPATCH
    solvers = [
        _try_gravity,           # Physics-based
        _try_subgrid_cropping,  # Focus-based
    ]
    
    # Family: Tiling/Periodic
    if dna['has_period'] or dna['ratio'][0] > 1.05:
        solvers.append(_try_periodic_tiling)
        
    # Family: Scaling/Kronecker
    if dna['is_scaling']:
        solvers.append(_try_scaling)
        solvers.append(_try_kronecker)
        
    # Family: Symmetry/Identity
    if dna['ratio'] == (1.0, 1.0):
        solvers.append(_try_symmetry)
        solvers.append(_try_global_shift)
        solvers.append(_try_enclosed_fill)
    
    # General pool (Fallback search)
    all_solvers = [
        _try_periodic_tiling, _try_symmetry, _try_global_shift, 
        _try_enclosed_fill, _try_scaling, _try_kronecker, _try_boundary, _try_cyclic_sequence
    ]
    for s in all_solvers:
        if s not in solvers: solvers.append(s)

    # Log the prioritized family search
    family_names = [s.__name__.replace('_try_', '').upper() for s in solvers]
    print(f"DNA Match: {family_names} ", end="", flush=True)

    for solver_fn in solvers:
        try:
            model = solver_fn(task)
            if model:
                if check_solve(model, task, train_only=True):
                    print(f"SOLVED ({solver_fn.__name__.replace('_try_', '').upper()})")
                    return model
        except Exception:
            continue
    # End of Tier 1 dispatch

    first_pair = task.train[0]
    in_grid = first_pair.input_grid
    out_grid = first_pair.output_grid

    in_h, in_w = len(in_grid), len(in_grid[0])
    out_h, out_w = len(out_grid), len(out_grid[0])

    if _all_outputs_identical(task):
        constant = ConstantGridSolver(out_grid)
        if check_solve(constant, task, train_only=True):
            return constant

    # Fast lane: if all pairs share input/output shapes, do pure-grid detection
    # and only then instantiate the corresponding Torch module.
    shapes = _all_pairs_same_shape(task)
    if shapes is not None:
        (in_h_all, in_w_all), (out_h_all, out_w_all) = shapes
        search_pairs = list(_iter_search_pairs(task))

        # Geometry ops that preserve shape (or swap for transpose/rotations).
        grid_ops: list[tuple[Literal["identity", "transpose", "rot90", "rot180", "rot270", "flip_h", "flip_v"], torch.nn.Module]] = [
            ("identity", IdentitySolver()),
            ("transpose", TransposeSolver()),
            ("rot90", RotationSolver(k=1)),
            ("rot180", RotationSolver(k=2)),
            ("rot270", RotationSolver(k=3)),
            ("flip_h", FlipSolver(horizontal=True)),
            ("flip_v", FlipSolver(horizontal=False)),
        ]

        # Try base geometric ops.
        for op, module in grid_ops:
            cmap = _derive_global_color_map_for_grid_transform(search_pairs, op)
            if cmap is None:
                continue
            candidate = CompositionalSolver(module, GeneralColorRemapSolver(cmap))
            if check_solve(candidate, task, train_only=True):
                return candidate

        # Try shifts with wider range cheaply.
        if (in_h_all, in_w_all) == (out_h_all, out_w_all):
            for dy in range(-max_shift, max_shift + 1):
                for dx in range(-max_shift, max_shift + 1):
                    if dx == 0 and dy == 0:
                        continue
                    cmap = _derive_global_color_map_for_grid_transform(search_pairs, "shift", dx=dx, dy=dy)
                    if cmap is None:
                        continue
                    candidate = CompositionalSolver(ShiftSolver(dx=dx, dy=dy), GeneralColorRemapSolver(cmap))
                    if check_solve(candidate, task, train_only=True):
                        return candidate

    # Lane A: full-grid geometry + color map.
    if in_h == out_h and in_w == out_w:
        for geom in _candidate_geometries(max_shift=max_shift):
            candidate = _wrap_with_color_map_if_match(geom, task)
            if candidate is not None and check_solve(candidate, task):
                return candidate

    # Lane A: integer nearest-neighbor upscale + color map.
    if out_h % in_h == 0 and out_w % in_w == 0:
        scale_h = out_h // in_h
        scale_w = out_w // in_w
        if scale_h >= 1 and scale_w >= 1:
            upscale = NearestNeighborScaleSolver(in_h=in_h, in_w=in_w, scale_h=scale_h, scale_w=scale_w)
            candidate = _wrap_with_color_map_if_match(upscale, task)
            if candidate is not None and check_solve(candidate, task):
                return candidate

    # Lane B: crop/transform/tile synthesis.
    geoms_no_shift: list[torch.nn.Module] = [
        IdentitySolver(),
        TransposeSolver(),
        RotationSolver(k=1),
        RotationSolver(k=2),
        RotationSolver(k=3),
        FlipSolver(horizontal=True),
        FlipSolver(horizontal=False),
    ]
    for uh in range(1, in_h + 1):
        for uw in range(1, in_w + 1):
            for y in range(in_h - uh + 1):
                for x in range(in_w - uw + 1):
                    sub = SubgridSolver(y, y + uh, x, x + uw)

                    if uh == out_h and uw == out_w:
                        for geom in geoms_no_shift:
                            base = CompositionalSolver(sub, geom)
                            candidate = _wrap_with_color_map_if_match(base, task)
                            if candidate is not None and check_solve(candidate, task):
                                return candidate

                    if out_h % uh == 0 and out_w % uw == 0:
                        rh, rw = out_h // uh, out_w // uw
                        tile = TilingSolver(uh, uw, rh, rw)
                        for geom in geoms_no_shift:
                            base = CompositionalSolver(sub, geom, tile)
                            candidate = _wrap_with_color_map_if_match(base, task)
                            if candidate is not None and check_solve(candidate, task):
                                return candidate

    # Lane B: Kronecker-style structure.
    max_unit_h = min(6, in_h)
    max_unit_w = min(6, in_w)
    for uh in range(2, max_unit_h + 1):
        for uw in range(2, max_unit_w + 1):
            if uh * uh != out_h or uw * uw != out_w:
                continue
            kron = KroneckerSolver(uh=uh, uw=uw)
            candidate = _wrap_with_color_map_if_match(kron, task)
            if candidate is not None and check_solve(candidate, task):
                return candidate

    # Lane M: Morphological ops (erode, dilate, color quantize, pattern repeat)


    morph_ops = [
        ("erode_1x", ErodeSolver(kernel_size=3, iterations=1)),
        ("erode_2x", ErodeSolver(kernel_size=3, iterations=2)),
        ("dilate_1x", DilateSolver(kernel_size=3, iterations=1)),
        ("dilate_2x", DilateSolver(kernel_size=3, iterations=2)),
        ("quantize_4", ColorQuantizeSolver(target_palette_size=4)),
        ("quantize_8", ColorQuantizeSolver(target_palette_size=8)),
        ("pattern_tile", PatternRepeatSolver(tile_h=2, tile_w=2, mode="tile")),
        ("pattern_mirror", PatternRepeatSolver(tile_h=1, tile_w=1, mode="mirror")),
    ]

    for op_name, morph_module in morph_ops:
        candidate = _wrap_with_color_map_if_match(morph_module, task)
        if candidate is not None and check_solve(candidate, task):
            return candidate

    # ── Lane N: New fundamental solvers ───────────────────────────────────────


    # N1: Fold + Overlay (flip grid and merge with original)
    candidate = _try_fold_overlay(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N2: Diagonal Periodic Tiling
    candidate = _try_diagonal_tiling(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N3: Gravity (pixels fall in one direction)
    candidate = _try_gravity(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N4: Subgrid Cropping (Crop to color focus)
    candidate = _try_subgrid_cropping(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N5: Flood Fill (enclosed background regions with fixed colour)
    candidate = _try_flood_fill(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N5: Isolated Pixel Cross-Line extension
    candidate = IsolatedPixelCrossSolver(bg_color=_detect_bg(task))
    if check_solve(candidate, task, train_only=True):
        return candidate

    return None


# ─── Helper detectors for new solver lanes ───────────────────────────────────

def _detect_bg(task: "TaskData") -> int:
    """Return the most common colour across all train inputs (background)."""
    counts = [0] * 10
    for pair in task.train:
        for row in pair.input_grid:
            for v in row:
                if 0 <= v < 10:
                    counts[v] += 1
    return int(np.argmax(counts))


def _try_fold_overlay(task: "TaskData"):
    """Detect FoldOverlaySolver parameters from training pairs."""
    pass # from .solvers import FoldOverlaySolver

    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    inp0 = np.array(pairs[0].input_grid, dtype=np.int32)
    out0 = np.array(pairs[0].output_grid, dtype=np.int32)
    if inp0.shape != out0.shape:
        return None

    bg = _detect_bg(task)

    for flip_dim, tfn in [(2, np.flipud), (3, np.fliplr)]:
        for mode in ("or_overlay", "replace_bg"):
            all_ok = True
            for pair in pairs:
                inp = np.array(pair.input_grid, dtype=np.int32)
                out = np.array(pair.output_grid, dtype=np.int32)
                if inp.shape != out.shape:
                    all_ok = False
                    break
                try:
                    fl = tfn(inp)
                except Exception:
                    all_ok = False
                    break
                if fl.shape != inp.shape:
                    all_ok = False
                    break
                if mode == "or_overlay":
                    cand = np.where((fl != bg) & (inp == bg), fl, inp)
                else:
                    cand = np.where(inp == bg, fl, inp)
                if not np.array_equal(cand, out):
                    all_ok = False
                    break
            if all_ok:
                max_h = max(np.array(p.input_grid).shape[0] for p in pairs)
                max_w = max(np.array(p.input_grid).shape[1] for p in pairs)
                return FoldOverlaySolver(flip_dim=flip_dim, mode=mode, bg_color=bg,
                                        grid_h=max_h, grid_w=max_w)
    return None


def _try_diagonal_tiling(task: "TaskData"):
    """Detect DiagonalPeriodicTilingSolver period from training pairs."""
    pass # from .solvers import DiagonalPeriodicTilingSolver

    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    inp0 = np.array(pairs[0].input_grid, dtype=np.int32)
    out0 = np.array(pairs[0].output_grid, dtype=np.int32)
    if inp0.shape != out0.shape:
        return None

    bg = _detect_bg(task)

    for period in range(1, min(20, inp0.shape[0] + inp0.shape[1])):
        all_ok = True
        for pair in pairs:
            inp = np.array(pair.input_grid, dtype=np.int32)
            out = np.array(pair.output_grid, dtype=np.int32)
            if inp.shape != out.shape:
                all_ok = False
                break

            # Collect diagonal colours
            h, w = inp.shape
            diag_map: dict[int, int] = {}
            consistent = True
            for r in range(h):
                for c in range(w):
                    if inp[r, c] != bg:
                        phase = (r + c) % period
                        color = inp[r, c]
                        if phase in diag_map:
                            if diag_map[phase] != color:
                                consistent = False
                                break
                        else:
                            diag_map[phase] = color
                if not consistent:
                    break
            if not consistent or len(diag_map) != period:
                all_ok = False
                break

            # Build expected output
            cand = np.zeros_like(inp)
            for r in range(h):
                for c in range(w):
                    phase = (r + c) % period
                    cand[r, c] = diag_map.get(phase, bg)
            if not np.array_equal(cand, out):
                all_ok = False
                break

        if all_ok:
            max_h = max(np.array(p.input_grid).shape[0] for p in pairs)
            max_w = max(np.array(p.input_grid).shape[1] for p in pairs)
            return DiagonalPeriodicTilingSolver(period=period, grid_h=max_h, grid_w=max_w,
                                               bg_color=bg)
    return None


def _try_gravity(task: "TaskData"):
    """Detect GravitySolver direction from training pairs."""
    pass # from .solvers import GravitySolver

    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    inp0 = np.array(pairs[0].input_grid, dtype=np.int32)
    out0 = np.array(pairs[0].output_grid, dtype=np.int32)
    if inp0.shape != out0.shape:
        return None

    bg = _detect_bg(task)

    for direction in ("down", "up", "left", "right"):
        all_ok = True
        for pair in pairs:
            inp = np.array(pair.input_grid, dtype=np.int32)
            out = np.array(pair.output_grid, dtype=np.int32)
            if inp.shape != out.shape:
                all_ok = False
                break
            h, w = inp.shape
            cand = np.full_like(inp, bg)
            if direction == "down":
                for c in range(w):
                    col = inp[:, c]
                    nz = col[col != bg]
                    if len(nz):
                        cand[h - len(nz):, c] = nz
            elif direction == "up":
                for c in range(w):
                    col = inp[:, c]
                    nz = col[col != bg]
                    if len(nz):
                        cand[: len(nz), c] = nz
            elif direction == "right":
                for r in range(h):
                    row = inp[r, :]
                    nz = row[row != bg]
                    if len(nz):
                        cand[r, w - len(nz):] = nz
            elif direction == "left":
                for r in range(h):
                    row = inp[r, :]
                    nz = row[row != bg]
                    if len(nz):
                        cand[r, : len(nz)] = nz
            if not np.array_equal(cand, out):
                all_ok = False
                break
        if all_ok:
            return GravitySolver(direction=direction, bg_color=bg)
    return None


def _try_cyclic_sequence(task: TaskData) -> torch.nn.Module | None:
    """Detects if grid is filled with a cyclic sequence of colors."""
    pairs = list(task.train)
    first_in = np.array(pairs[0].input_grid)
    first_out = np.array(pairs[0].output_grid)
    ih, iw = first_in.shape
    oh, ow = first_out.shape
    
    # Heuristic: try sequences of lengths 2-6
    for seq_len in range(2, 7):
        # Sample sequence from first row/column of output
        # Case A: Horizontal sequence
        for dx in range(seq_len):
            for dy in range(seq_len):
                # We need to find sequence S such that out[i, j] = S[(i*dy + j*dx)%L]
                # To find S, we collect (val, pos % L)
                s_map = {}
                possible = True
                for i in range(min(oh, 10)):
                    for j in range(min(ow, 10)):
                        val = first_out[i, j]
                        pos = (i * dy + j * dx) % seq_len
                        if pos in s_map and s_map[pos] != val:
                            possible = False; break
                        s_map[pos] = val
                    if not possible: break
                
                if possible and len(s_map) == seq_len:
                    seq = [s_map[p] for p in range(seq_len)]
                    solver = CyclicSequenceSolver(seq, dx=dx, dy=dy)
                    if check_solve(solver, task, train_only=True):
                        return solver
    return None


def _try_flood_fill(task: "TaskData"):
    """Detect FloodFillSolver(fill_color, bg_color) from training pairs.

    Matches tasks where every enclosed background region is filled with
    a single fixed colour (consistent across all pairs).
    """
    pass # from .solvers import FloodFillSolver
    try:
        from scipy import ndimage  # type: ignore
    except ImportError:
        return None

    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    inp0 = np.array(pairs[0].input_grid, dtype=np.int32)
    out0 = np.array(pairs[0].output_grid, dtype=np.int32)
    if inp0.shape != out0.shape:
        return None

    bg = _detect_bg(task)

    # Check all pairs: only bg cells change, enclosed bg → same fixed fill colour
    fixed_fill: int | None = None
    all_ok = True
    for pair in pairs:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        if inp.shape != out.shape:
            all_ok = False
            break
        # Non-bg cells must not change
        if np.any((inp != out) & (inp != bg)):
            all_ok = False
            break
        # Find enclosed bg regions
        bg_mask = (inp == bg).astype(np.int32)
        labeled, n_regions = ndimage.label(bg_mask)
        h, w = inp.shape
        border_labels: set[int] = set()
        for r in range(h):
            for c in [0, w - 1]:
                if labeled[r, c] > 0:
                    border_labels.add(labeled[r, c])
        for c in range(w):
            for r in [0, h - 1]:
                if labeled[r, c] > 0:
                    border_labels.add(labeled[r, c])

        enclosed = [i for i in range(1, n_regions + 1) if i not in border_labels]
        if not enclosed:
            all_ok = False
            break

        for rid in enclosed:
            reg_mask = labeled == rid
            fill_colors = set(out[reg_mask].tolist())
            if len(fill_colors) != 1:
                all_ok = False
                break
            fc = fill_colors.pop()
            if fc == bg:
                all_ok = False
                break
            if fixed_fill is None:
                fixed_fill = fc
            elif fixed_fill != fc:
                all_ok = False
                break
        if not all_ok:
            break

    if all_ok and fixed_fill is not None:
        return FloodFillSolver(fill_color=fixed_fill, bg_color=bg, iterations=60)
    return None


def _train_fallback(task: TaskData, task_id: str) -> torch.nn.Module | None:
    if not task.train:
        print(f"[{task_id}] Warning: No training pairs found. Skipping neural tier.")
        return None

    profile = _build_fallback_profile(task)
    print(
        f"[{task_id}] Adaptive fallback route: tier={profile.tier}, "
        f"branches={profile.n_models}, epochs={profile.epochs}, "
        f"hidden={profile.hidden_channels}, steps={profile.steps}"
    )

    def _train_backbone(
        *,
        n_models: int,
        epochs: int,
        hidden_channels: int,
        steps: int,
        batch_size: int,
        seed: int,
    ) -> EnsembleSolver:
        return train_ensemble_for_task(
            task=task,
            task_id=task_id,
            n_models=n_models,
            config=TrainConfig(
                epochs=epochs,
                learning_rate=4e-3,
                weight_decay=1e-5,
                arcgen_train_sample=32,
                batch_size=batch_size,
                seed=seed,
                use_augmentation=True,
                min_epochs=max(16, int(epochs * 0.2)),
                eval_interval=8,
                early_stop_patience=20,
                early_stop_delta=5e-4,
                entropy_patience_bonus=4,
                enable_dynamic_early_stop=True,
            ),
            backbone_kwargs={
                "hidden_channels": hidden_channels,
                "steps": steps,
                "scratch_channels": 8,
                "mask_channels": 2,
                "phase_channels": 1,
                "use_coords": True,
                "use_depthwise": True,
            },
        )

    backbone = _train_backbone(
        n_models=profile.n_models,
        epochs=profile.epochs,
        hidden_channels=profile.hidden_channels,
        steps=profile.steps,
        batch_size=profile.batch_size,
        seed=42,
    )

    color_map = None
    if hasattr(backbone, "branches") and len(backbone.branches) > 0:
        color_map = getattr(backbone.branches[0], "neurogolf_color_map", None)
    if color_map is None:
        all_inputs = [pair.input_grid for pair in task.train]
        all_inputs.extend(pair.input_grid for pair in task.test)
        color_map = get_color_normalization_map(all_inputs)

    def _make_candidates(
        source_backbone: EnsembleSolver,
        *,
        prefix: str = "",
    ) -> list[tuple[str, torch.nn.Module]]:
        stage_candidates: list[tuple[str, torch.nn.Module]] = [
            (f"{prefix}mean_ensemble", ColorNormalizedSolver(backbone=source_backbone, color_map=color_map))
        ]

        if hasattr(source_backbone, "branches"):
            stage_candidates.append(
                (
                    f"{prefix}consensus",
                    ColorNormalizedSolver(
                        backbone=ConsensusEnsembleSolver(source_backbone.branches),
                        color_map=color_map,
                    ),
                )
            )
            for i, branch in enumerate(source_backbone.branches):
                stage_candidates.append(
                    (f"{prefix}branch_{i + 1}", ColorNormalizedSolver(backbone=branch, color_map=color_map))
                )

        return stage_candidates

    def _try_exact(candidates: list[tuple[str, torch.nn.Module]]) -> torch.nn.Module | None:
        for name, candidate in candidates:
            if not check_solve(candidate, task, train_only=True):
                continue
            if "consensus" in name:
                print("SOLVED (Neural Consensus Recovery)")
            elif "branch_" in name:
                print(f"SOLVED (Neural Recovery - {name} Exact Match)")
            else:
                print("SOLVED (Neural Mean Ensemble)")
            setattr(candidate, "neurogolf_exact_fit", True)
            return candidate
        return None

    candidates = _make_candidates(backbone)
    exact_candidate = _try_exact(candidates)
    if exact_candidate is not None:
        return exact_candidate

    identity_score = score_task_fit(IdentitySolver(), task, train_only=True)

    def _select_best(
        candidate_pool: list[tuple[str, torch.nn.Module]],
    ) -> tuple[str, torch.nn.Module | None, TaskFitScore, float]:
        best_name = ""
        best_model: torch.nn.Module | None = None
        best_score = TaskFitScore(exact_pairs=-1, total_pairs=0, pixel_accuracy=-1.0)
        best_rank = -1.0
        for name, candidate in candidate_pool:
            candidate_score = score_task_fit(candidate, task, train_only=True)
            rank = _poe_rank_score(name, candidate_score, identity_score)
            if (rank, candidate_score.exact_pairs, candidate_score.pixel_accuracy) > (
                best_rank,
                best_score.exact_pairs,
                best_score.pixel_accuracy,
            ):
                best_name = name
                best_model = candidate
                best_score = candidate_score
                best_rank = rank
        return best_name, best_model, best_score, best_rank

    best_name, best_model, best_score, best_rank = _select_best(candidates)
    improved_over_identity = (
        best_score.exact_pairs > identity_score.exact_pairs
        or (
            best_score.exact_pairs == identity_score.exact_pairs
            and best_score.pixel_accuracy >= identity_score.pixel_accuracy + 0.01
        )
    )

    best_exact_ratio = best_score.exact_pairs / max(1, best_score.total_pairs)
    low_promise = best_exact_ratio < 0.10 and best_score.pixel_accuracy < 0.55 and not improved_over_identity

    should_promote = (
        not low_promise
        and improved_over_identity
        and best_score.pixel_accuracy >= profile.promotion_pixel_threshold
        and profile.promotion_epochs > profile.epochs
    )

    if should_promote:
        print(
            "PROMOTING (Neural Stage-2) "
            f"trigger={best_name}, pixel={best_score.pixel_accuracy:.3f}, "
            f"exact={best_score.exact_pairs}/{best_score.total_pairs}"
        )
        promoted_backbone = _train_backbone(
            n_models=profile.promotion_n_models,
            epochs=profile.promotion_epochs,
            hidden_channels=profile.promotion_hidden_channels,
            steps=profile.promotion_steps,
            batch_size=max(6, profile.batch_size - 2),
            seed=1337,
        )
        promoted_candidates = _make_candidates(promoted_backbone, prefix="promoted_")
        exact_candidate = _try_exact(promoted_candidates)
        if exact_candidate is not None:
            return exact_candidate
        candidates.extend(promoted_candidates)
        best_name, best_model, best_score, best_rank = _select_best(candidates)
        improved_over_identity = (
            best_score.exact_pairs > identity_score.exact_pairs
            or (
                best_score.exact_pairs == identity_score.exact_pairs
                and best_score.pixel_accuracy >= identity_score.pixel_accuracy + 0.01
            )
        )
        best_exact_ratio = best_score.exact_pairs / max(1, best_score.total_pairs)
        low_promise = best_exact_ratio < 0.10 and best_score.pixel_accuracy < 0.55 and not improved_over_identity

    if low_promise:
        print(
            "REJECTED (Low-Promise Neural) "
            f"exact={best_score.exact_pairs}/{best_score.total_pairs}, "
            f"pixel={best_score.pixel_accuracy:.3f}"
        )
        return None

    if best_model is not None and improved_over_identity and best_score.pixel_accuracy >= 0.65:
        print(
            "RECOVERED (Neural Best-Effort "
            f"{best_name}: exact={best_score.exact_pairs}/{best_score.total_pairs}, "
            f"pixel={best_score.pixel_accuracy:.3f}, rank={best_rank:.4f})"
        )
        setattr(best_model, "neurogolf_exact_fit", False)
        return best_model

    return None





# --- train.py ---
"""Local training helpers for RegisterBackbone on ARC tasks."""



from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Sequence

import numpy as np



try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    F = None


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    arcgen_train_sample: int = 32
    batch_size: int = 16
    seed: int = 42
    use_augmentation: bool = True
    min_epochs: int = 24
    eval_interval: int = 8
    early_stop_patience: int = 20
    early_stop_delta: float = 5e-4
    entropy_patience_bonus: int = 4
    enable_dynamic_early_stop: bool = True


@dataclass(frozen=True)
class TaskTrainSummary:
    task_id: str
    train_samples: int
    final_loss: float
    best_loss: float
    epochs: int


@dataclass(frozen=True)
class SliceRunSummary:
    total_task_files: int
    selected_task_files: int
    trained_tasks: int
    skipped_tasks: int



def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for training.")


def _pick_device() -> "torch.device":
    _require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _augment_grid(grid: list[list[int]], k: int, flip: bool) -> list[list[int]]:
    """Rotate k*90 degrees and optionally flip."""
    arr = np.array(grid)
    if k > 0:
        arr = np.rot90(arr, k=k)
    if flip:
        arr = np.fliplr(arr)
    return arr.tolist()


def _build_train_pairs(task: TaskData, config: TrainConfig) -> list[GridPair]:
    pairs = list(task.train)
    
    if config.use_augmentation:
        augmented = []
        for pair in pairs:
            # Add 4 rotations x 2 flips = 8 symmetries
            for k in range(4):
                for flip in [False, True]:
                    if k == 0 and not flip:
                        continue # Original is already in 'pairs'
                    augmented.append(GridPair(
                        input_grid=_augment_grid(pair.input_grid, k, flip),
                        output_grid=_augment_grid(pair.output_grid, k, flip)
                    ))
        pairs.extend(augmented)

    if config.arcgen_train_sample > 0 and len(task.arc_gen) > 0:
        rnd = random.Random(config.seed)
        sample_size = min(config.arcgen_train_sample, len(task.arc_gen))
        pairs.extend(rnd.sample(list(task.arc_gen), k=sample_size))
    return pairs


def _pairs_to_tensors(pairs: Sequence[GridPair], device: "torch.device") -> tuple["torch.Tensor", "torch.Tensor"]:
    inputs = []
    targets = []
    for pair in pairs:
        in_tensor = encode_grid_to_tensor(pair.input_grid)
        out_tensor = encode_grid_to_tensor(pair.output_grid)

        inputs.append(in_tensor[0])
        # Cross-entropy expects integer class labels per pixel.
        targets.append(np.argmax(out_tensor[0], axis=0).astype(np.int64))

    x = torch.from_numpy(np.stack(inputs)).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(np.stack(targets)).to(device=device, dtype=torch.long)
    return x, y




def _apply_augmentation(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies random symmetry on the fly. Color is handled by normalization."""
    # Symmetries (Rot 90s + Flip)
    k = random.randint(0, 3)
    flip = random.random() > 0.5
    
    x = torch.rot90(x, k=k, dims=(2, 3))
    y = torch.rot90(y, k=k, dims=(1, 2))
    if flip:
        x = torch.flip(x, dims=(3,))
        y = torch.flip(y, dims=(2,))
        
    return x, y


def train_task_model(
    model: "torch.nn.Module",
    task: TaskData,
    task_id: str,
    config: TrainConfig,
    device: "torch.device" | None = None,
) -> TaskTrainSummary:
    _require_torch()
    _seed_everything(config.seed)
    if device is None:
        device = _pick_device()

    # Get GLOBAL map for this task
    global_cmap = getattr(model, "neurogolf_color_map", None)
    if global_cmap is None:
        all_inputs = [pair.input_grid for pair in task.train]
        all_inputs.extend(pair.input_grid for pair in task.test)
        global_cmap = get_color_normalization_map(all_inputs)
        setattr(model, "neurogolf_color_map", global_cmap)

    norm_pairs = []
    for pair in task.train:
        norm_in = apply_color_map(pair.input_grid, global_cmap)
        norm_out = apply_color_map(pair.output_grid, global_cmap)
        norm_pairs.append(GridPair(input_grid=norm_in, output_grid=norm_out))

    model.to(device)
    base_x, base_y = _pairs_to_tensors(norm_pairs, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

    best_loss = float("inf")
    best_exact = 0.0
    best_entropy = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    final_loss = float("inf")
    patience_left = max(1, int(config.early_stop_patience))
    recent_losses: list[float] = []
    completed_epochs = 0

    for epoch in range(config.epochs):
        completed_epochs = epoch + 1
        model.train()
        xb, yb = _apply_augmentation(base_x, base_y)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        
        # Dynamic L1: Disable regularization for final precision fit
        l1_coeff = 1e-6 if loss > 0.01 else 0.0
        l1_reg = torch.tensor(0.0, device=device)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        
        total_loss = loss + l1_coeff * l1_reg
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(base_x)
            eval_loss = F.cross_entropy(eval_logits, base_y)
            pred = torch.argmax(eval_logits, dim=1)
            pair_exact = float((pred == base_y).view(pred.shape[0], -1).all(dim=1).float().mean().item())
            probs = torch.softmax(eval_logits, dim=1).clamp_min(1e-8)
            entropy = float((-(probs * probs.log()).sum(dim=1).mean()).item())

        epoch_loss = float(eval_loss.item())
        final_loss = epoch_loss
        recent_losses.append(epoch_loss)

        improved_loss = epoch_loss + config.early_stop_delta < best_loss
        improved_exact = pair_exact > best_exact + 1e-6
        entropy_not_worse = entropy <= best_entropy + 1e-3

        if improved_loss or improved_exact:
            best_loss = min(best_loss, epoch_loss)
            best_exact = max(best_exact, pair_exact)
            if entropy_not_worse:
                best_entropy = min(best_entropy, entropy)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            if config.enable_dynamic_early_stop:
                patience_left = config.early_stop_patience
                if pair_exact >= 0.75 and entropy_not_worse:
                    patience_left += config.entropy_patience_bonus
        elif config.enable_dynamic_early_stop and epoch + 1 >= config.min_epochs:
            patience_left -= 1

        if pair_exact >= 1.0 and epoch + 1 >= max(8, config.min_epochs // 2):
            break

        if (
            config.enable_dynamic_early_stop
            and epoch + 1 >= config.min_epochs
            and (epoch + 1) % max(1, config.eval_interval) == 0
        ):
            plateau = len(recent_losses) >= 3 and (
                max(recent_losses[-3:]) - min(recent_losses[-3:]) < config.early_stop_delta
            )
            if patience_left <= 0 or (plateau and pair_exact >= 0.95):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if not np.isfinite(best_loss):
        best_loss = final_loss

    return TaskTrainSummary(
        task_id=task_id,
        train_samples=len(norm_pairs),
        final_loss=final_loss,
        best_loss=best_loss,
        epochs=completed_epochs,
    )



def iter_task_files(dataset_root: str | Path) -> list[Path]:
    return sorted(Path(dataset_root).glob("task*.json"))


def select_task_files(task_files: list[Path], start_index: int, end_index: int | None) -> list[Path]:
    if start_index < 1:
        raise ValueError("start_index must be >= 1")

    start = start_index - 1
    if end_index is None:
        return task_files[start:]
    if end_index < start_index:
        raise ValueError("end_index must be >= start_index")
    return task_files[start:end_index]


def save_slice_training_report(
    report_path: str | Path,
    dataset_root: str | Path,
    summary: SliceRunSummary,
    train_summaries: list[TaskTrainSummary],
    eval_report_path: str | Path,
    skipped: list[dict[str, str]],
    config: TrainConfig,
) -> None:
    payload = {
        "dataset_root": str(dataset_root),
        "summary": asdict(summary),
        "train_config": asdict(config),
        "eval_report_path": str(eval_report_path),
        "train_summaries": [asdict(item) for item in train_summaries],
        "skipped": skipped,
    }

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_task_for_training(task_path: str | Path) -> TaskData:
    return load_task_json(task_path)


# --- train_ensemble.py ---
"""Train an ensemble of models for a single ARC task."""



import torch
from typing import List


def train_ensemble_for_task(
    task: TaskData,
    task_id: str,
    n_models: int = 3,
    config: TrainConfig = TrainConfig(),
    backbone_kwargs: dict | None = None,
) -> EnsembleSolver:
    """ Trains N models with different seeds and returns an ensemble. """
    models = []
    if backbone_kwargs is None:
        backbone_kwargs = {}
    
    for i in range(n_models):
        print(f"  Training ensemble branch {i+1}/{n_models}...", end=" ", flush=True)
        # Use a different seed for each branch
        branch_config = TrainConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            arcgen_train_sample=config.arcgen_train_sample,
            seed=config.seed + i,
            use_augmentation=config.use_augmentation,
            batch_size=config.batch_size,
            min_epochs=config.min_epochs,
            eval_interval=config.eval_interval,
            early_stop_patience=config.early_stop_patience,
            early_stop_delta=config.early_stop_delta,
            entropy_patience_bonus=config.entropy_patience_bonus,
            enable_dynamic_early_stop=config.enable_dynamic_early_stop,
        )
        
        model = RegisterBackbone(**backbone_kwargs)
        train_summary = train_task_model(model, task, task_id, branch_config)
        print(f"loss={train_summary.final_loss:.4f}")
        models.append(model)
        
    return EnsembleSolver(models)


# --- export.py ---
"""PyTorch -> ONNX export helpers with static-shape validation."""



from pathlib import Path
from typing import Any



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
        # Torch 2.6+ emits a noisy pytree deprecation warning inside export internals.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                category=FutureWarning,
            )
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


# --- onnx_rules.py ---
"""ONNX static-shape and operator compliance checks."""



from dataclasses import dataclass
from pathlib import Path
from typing import Any



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


# --- STATICS REFINEMENT V4 ---
def refine_statics_v4(onnx_path, output_path):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto
    try:
        model = onnx.load(str(onnx_path))
        all_tensors = set()
        for node in model.graph.node:
            for output in node.output:
                if output: all_tensors.add(output)
        for inp in model.graph.input: all_tensors.add(inp.name)

        temp_model = onnx.load(str(onnx_path))
        temp_model.graph.ClearField('output')
        for t_name in all_tensors:
            new_out = temp_model.graph.output.add()
            new_out.name = t_name
            
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(temp_model.SerializeToString(), sess_options=so, providers=['CPUExecutionProvider'])
        
        # Dynamically satisfy ALL required inputs
        input_feed = {}
        for inp in sess.get_inputs():
            shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
            # Special case for standard ARC input
            if inp.name == 'input': shape = [1, 10, 30, 30]
            input_feed[inp.name] = np.zeros(shape, dtype=np.float32)
            
        run_outputs = sess.run(None, input_feed)
        name_to_shape = {name: out.shape for name, out in zip([idx.name for idx in sess.get_outputs()], run_outputs)}
        name_to_type = {name: out.dtype for name, out in zip([idx.name for idx in sess.get_outputs()], run_outputs)}

        final_model = onnx.load(str(onnx_path))
        final_model.graph.ClearField('value_info')
        for name, shape in name_to_shape.items():
            vi = final_model.graph.value_info.add()
            vi.name = name
            tt = vi.type.tensor_type
            if name_to_type[name] == np.float32: tt.elem_type = TensorProto.FLOAT
            elif name_to_type[name] == np.int64: tt.elem_type = TensorProto.INT64
            elif name_to_type[name] == np.int32: tt.elem_type = TensorProto.INT32
            elif name_to_type[name] == np.bool_: tt.elem_type = TensorProto.BOOL
            else: tt.elem_type = TensorProto.FLOAT
            tt.shape.MergeFrom(onnx.TensorShapeProto())
            for s in shape:
                dim = tt.shape.dim.add()
                dim.dim_value = max(1, int(s))

        for tensor in list(final_model.graph.input) + list(final_model.graph.output):
            if tensor.type.HasField('tensor_type'):
                tt = tensor.type.tensor_type
                tt.shape.MergeFrom(onnx.TensorShapeProto())
                tt.shape.ClearField('dim')
                actual_shape = name_to_shape.get(tensor.name, [1, 10, 30, 30])
                for d in actual_shape:
                    dim = tt.shape.dim.add()
                    dim.dim_value = max(1, int(d))
        onnx.save(final_model, str(output_path))
        return True
    except Exception as e:
        print(f"Error refining {onnx_path}: {e}")
        return False


def _is_arc_task_json(path: Path) -> bool:
    """Returns True when JSON contains ARC-style train/test lists."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("train"), list) and isinstance(payload.get("test"), list)


def _discover_arc_task_files(root: Path, recursive: bool = True) -> list[Path]:
    iterator = root.rglob("task*.json") if recursive else root.glob("task*.json")
    return sorted(path for path in iterator if path.is_file() and _is_arc_task_json(path))

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=0)
    args, _ = parser.parse_known_args()

    print("🚀 Starting NeuroGolf SOTA Synthesis...")
    
    # --- Robust Data Discovery ---
    # We search in multiple candidate roots to avoid "Data directory empty" failures.
    user_dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    home_downloads = Path.home() / "Downloads"
    candidate_roots = []
    if user_dataset_root is not None:
        candidate_roots.append(user_dataset_root)
    candidate_roots.extend(
        [
            Path("data"),
            Path("."),
            Path("/content"),
            Path("neurogolf-2026"),
            home_downloads / "neurogolf-2026",
            home_downloads / "neurogolf-2026-main",
        ]
    )

    # De-duplicate while preserving order.
    deduped_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in candidate_roots:
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        deduped_roots.append(root)
    candidate_roots = deduped_roots
    
    tasks = []
    dataset_found_in = None
    
    # Priority search: find ARC-like task JSON files.
    for root in candidate_roots:
        if root.exists() and root.is_dir():
            # Avoid deep recursive scans under '.'; recurse for explicit dataset dirs.
            recursive_scan = root != Path(".")
            found = _discover_arc_task_files(root, recursive=recursive_scan)
            if found:
                tasks = found
                dataset_found_in = root
                break
    
    # Fallback to general JSONs only if in a dedicated data directory
    if not tasks:
        for root in candidate_roots:
            if root.exists() and root.is_dir() and "data" in str(root).lower():
                iterator = root.rglob("*.json") if root != Path(".") else root.glob("*.json")
                found = sorted(path for path in iterator if path.is_file() and _is_arc_task_json(path))
                if found:
                    tasks = found
                    dataset_found_in = root
                    break

    if not tasks:
        print("\n[!] No tasks found in common candidate directories.")
        # Fallback to Colab uploader if available
        try:
            from google.colab import files
            print("Opening uploader... Please select your task files.")
            uploaded = files.upload()
            if uploaded:
                dataset_root = Path("data")
                dataset_root.mkdir(exist_ok=True)
                for filename in uploaded.keys():
                    if filename.endswith(".zip"):
                        print(f"Unzipping {filename}...")
                        import zipfile
                        with zipfile.ZipFile(filename, 'r') as zip_ref:
                            zip_ref.extractall(dataset_root)
                    else:
                        shutil.move(filename, dataset_root / filename)
                tasks = sorted(path for path in dataset_root.rglob("*.json") if path.is_file() and _is_arc_task_json(path))
                dataset_found_in = dataset_root
        except ImportError:
            print("Error: Manual data placement required. Searched in:")
            for c in candidate_roots: print(f"  - {c}")
            return

    if not tasks:
        print("Error: No tasks identified. Synthesis cannot proceed.")
        return

    discovered_count = len(tasks)
    start_index = max(1, int(args.start_index))
    end_index = int(args.end_index) if int(args.end_index) > 0 else None
    tasks = select_task_files(tasks, start_index, end_index)
    if args.task_limit and args.task_limit > 0:
        tasks = tasks[: args.task_limit]

    if not tasks:
        print(
            "Error: Selected task slice is empty. "
            f"start_index={start_index}, end_index={end_index}, task_limit={args.task_limit}"
        )
        return

    print(
        f"✅ Dataset discovered in: {dataset_found_in} "
        f"({len(tasks)}/{discovered_count} tasks selected)"
    )

    out_dir = Path("artifacts/submission")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    solved_ids = []
    best_effort_ids = []
    failed_ids = []
    
    for t_file in tasks:
        task_id = t_file.stem
        try:
            task, dropped = load_task_json_relaxed(t_file)
        except:
            print(f"[{task_id}] Error loading task. Skipping.")
            continue
            
        print(f"[{task_id}] Searching... ", end="", flush=True)
        # 1. Try Symbolic
        model = find_master_synthesis(task, max_shift=4)
        if model:
            print("SOLVED (Symbolic)")
            solved_ids.append(task_id)
        else:
            # 2. Neural Fallback (Color Normalized)
            print("Neural fallback... ", end="", flush=True)
            model = _train_fallback(task, task_id)
            if model:
                if bool(getattr(model, "neurogolf_exact_fit", False)):
                    print("SOLVED (Neural)")
                    solved_ids.append(task_id)
                else:
                    print("⚠️ PARTIAL (Neural Best-Effort)")
                    best_effort_ids.append(task_id)
            else:
                print("⚠️ FAILED (Using Identity Fallback)")
                model = IdentitySolver()
                failed_ids.append(task_id)
        
        raw_path = out_dir / f"raw_{task_id}.onnx"
        final_path = out_dir / f"{task_id}.onnx"
        export_static_onnx(model, raw_path)
        refine_statics_v4(raw_path, final_path)

    # Zip and finish
    with zipfile.ZipFile("neurogolf_submission_v4_hybrid_sota.zip", "w") as z:
        for f in out_dir.glob("task*.onnx"):
            if not f.name.startswith("raw_"):
                z.write(f, f.name)
                
    print(f"\n" + "="*40)
    print(f"✅ CAMPAIGN COMPLETE.")
    print(f"   Successfully Solved: {len(solved_ids)}/{len(tasks)}")
    print(f"   Neural Best-Effort:   {len(best_effort_ids)}/{len(tasks)}")
    print(f"   Identity Fallbacks:  {len(failed_ids)}/{len(tasks)}")
    
    if failed_ids:
        print("\n❌ Tasks for Manual Review (failed both tiers):")
        for fid in failed_ids:
            print(f" - {fid}")
    print("="*40)

if __name__ == "__main__":
    main()
