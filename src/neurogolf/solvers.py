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

from .constants import GRID_SIZE, STATE_CHANNELS, COLOR_CHANNELS, IDENTITY_CHANNELS
from .grid_codec import encode_grid_to_tensor


class DilateSolver(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == STATE_CHANNELS, (
            f"State corruption: expected {STATE_CHANNELS} channels, got {x.shape[1]}"
        )
        colors = x[:, :COLOR_CHANNELS, :, :]
        ids = x[:, COLOR_CHANNELS:, :, :]
        dilated_colors = F.max_pool2d(colors, 3, stride=1, padding=1)
        return torch.cat([dilated_colors, ids], dim=1)


class ErodeSolver(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == STATE_CHANNELS, (
            f"State corruption: expected {STATE_CHANNELS} channels, got {x.shape[1]}"
        )
        colors = x[:, :COLOR_CHANNELS, :, :]
        eroded = 1.0 - F.max_pool2d(1.0 - colors, 3, stride=1, padding=1)
        return torch.cat([eroded, x[:, COLOR_CHANNELS:, :, :]], dim=1)


# State Integrity Guard
def validate_state(x: torch.Tensor, name: str = "Primitive") -> torch.Tensor:
    if x.shape[1] == 0:
        raise RuntimeError(f"ZERO CHANNEL STATE DETECTED in {name}")
    return x


class MultiCriteriaObjectSelector(nn.Module):
    """Selects objects based on criteria like 'largest', 'smallest', 'topmost'."""

    def __init__(self, criterion: str = "largest") -> None:
        super().__init__()
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > COLOR_CHANNELS:
            ids = x[:, COLOR_CHANNELS:, :, :]
        else:
            # It's a mask or raw color grid. Use the color channels as IDs if needed,
            # or treat the whole thing as one object if it's a 1-ch mask.
            ids = (
                x
                if x.shape[1] == 1
                else (
                    x[:, 1:COLOR_CHANNELS, :, :].sum(dim=1, keepdim=True) > 0.5
                ).float()
            )

        B = ids.shape[0]
        areas = ids.sum(dim=(2, 3))  # [B, K]

        # Guard against zero-channel ids
        if ids.shape[1] == 0:
            return x

        if self.criterion == "largest":
            idx = areas.argmax(dim=1)  # [1]
        elif self.criterion == "smallest":
            m_areas = torch.where(areas > 0, areas, torch.full_like(areas, 1e6))
            idx = m_areas.argmin(dim=1)  # [1]
        else:
            idx = torch.tensor([0], device=x.device)

        # Gather-style indexing for batch awareness [B, 1, H, W]
        selected_id_mask = ids[torch.arange(B), idx].unsqueeze(1)

        # Colors extraction
        if x.shape[1] >= COLOR_CHANNELS:
            colors = x[:, :COLOR_CHANNELS, :, :]
        else:
            colors = x  # Placeholder

        selected_colors = colors * selected_id_mask

        # Reconstruct identity stack (Standard K=16)
        new_ids = torch.zeros(
            B, IDENTITY_CHANNELS, GRID_SIZE, GRID_SIZE, device=x.device
        )
        # Place the selected mask back into its original slot (mod K)
        # We use scatter-based assignment for batch-safe reconstructions
        target_idx = (
            (idx % IDENTITY_CHANNELS)
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, GRID_SIZE, GRID_SIZE)
        )
        new_ids.scatter_(1, target_idx, selected_id_mask)

        # Ensure we return 10 color channels
        if selected_colors.shape[1] < COLOR_CHANNELS:
            # Promote mask to color (e.g. assume color 1)
            bg = 1.0 - selected_id_mask
            res_colors = torch.cat(
                [
                    bg,
                    selected_id_mask,
                    torch.zeros(x.shape[0], 8, GRID_SIZE, GRID_SIZE, device=x.device),
                ],
                dim=1,
            )
        else:
            res_colors = selected_colors

        out = torch.cat([res_colors, new_ids], dim=1)
        return validate_state(out, "ObjectSelector")


class RelativeMoveSolver(nn.Module):
    """Move source object to target object's position.

    source: criterion to select source object ("largest", "smallest", "color_1", "color_2", etc.)
    target: criterion to select target object ("largest", "smallest", "color_1", "color_2", etc.)
    mode: "center" - move source center to target center

    This is the minimal relative placement solver needed for tasks like "move A next to B".
    """

    def __init__(
        self, source: str = "largest", target: str = "smallest", mode: str = "center"
    ) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.mode = mode

    def _get_object_mask(self, x: torch.Tensor, criterion: str) -> torch.Tensor:
        """Get mask for object matching criterion."""
        colors = x[:, :COLOR_CHANNELS, :, :]

        # If criterion is like "color_1", extract that specific color
        if criterion.startswith("color_"):
            color_idx = int(criterion.split("_")[1])
            if 0 <= color_idx < COLOR_CHANNELS:
                return colors[:, color_idx : color_idx + 1, :, :]

        # Otherwise use size-based selection
        # Create individual object masks using connected components logic via channel splitting
        # For now, treat each color channel > 0 as a separate "object"
        if criterion == "largest":
            areas = colors[:, 1:, :, :].sum(dim=(2, 3))  # Skip bg
            idx = areas.argmax(dim=1, keepdim=True)
            mask = (
                colors[:, 1:, :, :]
                .gather(
                    1, idx.unsqueeze(2).unsqueeze(3).expand(-1, 1, GRID_SIZE, GRID_SIZE)
                )
                .clamp(0, 1)
            )
            return mask
        elif criterion == "smallest":
            areas = colors[:, 1:, :, :].sum(dim=(2, 3))
            areas = torch.where(areas > 0, areas, torch.full_like(areas, 1e6))
            idx = areas.argmin(dim=1, keepdim=True)
            mask = (
                colors[:, 1:, :, :]
                .gather(
                    1, idx.unsqueeze(2).unsqueeze(3).expand(-1, 1, GRID_SIZE, GRID_SIZE)
                )
                .clamp(0, 1)
            )
            return mask

        # Default: return first non-bg object
        return colors[:, 1:2, :, :].clamp(0, 1)

    def _get_center(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get center coordinates of mask."""
        h_indices = (
            torch.arange(GRID_SIZE, device=mask.device).view(1, 1, GRID_SIZE, 1).float()
        )
        w_indices = (
            torch.arange(GRID_SIZE, device=mask.device).view(1, 1, 1, GRID_SIZE).float()
        )

        count = mask.sum() + 1e-6
        mean_h = (mask * h_indices).sum() / count
        mean_w = (mask * w_indices).sum() / count

        return mean_h, mean_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()

        # Get source and target masks
        src_mask = self._get_object_mask(x, self.source)
        tgt_mask = self._get_object_mask(x, self.target)

        # Get centers
        src_h, src_w = self._get_center(src_mask)
        tgt_h, tgt_w = self._get_center(tgt_mask)

        # Calculate shift
        dy = int(tgt_h.item() - src_h.item())
        dx = int(tgt_w.item() - src_w.item())

        if dy == 0 and dx == 0:
            return validate_state(out, "RelativeMove")

        # Shift the source object
        shifted_src = torch.roll(src_mask, shifts=(dy, dx), dims=(2, 3))

        # Clear original source position
        out[:, 1:COLOR_CHANNELS, :, :] = out[:, 1:COLOR_CHANNELS, :, :] * (1 - src_mask)

        # Place at new position (add to existing)
        out[:, 1:COLOR_CHANNELS, :, :] = out[:, 1:COLOR_CHANNELS, :, :] + shifted_src

        return validate_state(out, "RelativeMove")


class IdentitySolver(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return validate_state(x, "Identity")


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
    """
    Fixed cropping to a bounding box.
    """
    def __init__(self, y1: int, y2: int, x1: int, x2: int) -> None:
        super().__init__()
        self.bounds = (y1, y2, x1, x2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.bounds[0] : self.bounds[1], self.bounds[2] : self.bounds[3]]


class DynamicSliceSolver(nn.Module):
    """
    More complex cropping (central, rows-after-color, etc).
    """
    def __init__(self, mode: str, size: tuple[int, int] = (5, 5), bounds: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        super().__init__()
        self.mode = mode
        self.size = size
        self.bounds = bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.mode == "central":
            sh, sw = self.size
            y1 = (H - sh) // 2
            x1 = (W - sw) // 2
            return x[:, :, y1 : y1 + sh, x1 : x1 + sw]
        elif self.mode == "fixed":
            y1, y2, x1, x2 = self.bounds
            return x[:, :, y1:y2, x1:x2]
        return x



class TilingSolver(nn.Module):
    """Repeats only the active (uh, uw) portion of the input."""

    def __init__(self, uh: int, uw: int, repeats_h: int, repeats_w: int) -> None:
        super().__init__()
        self.uh = uh
        self.uw = uw
        self.repeats = (repeats_h, repeats_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [1, 10, GRID_SIZE, GRID_SIZE]
        # Extract the unit
        unit = x[:, :, : self.uh, : self.uw]
        # Repeat it
        out = unit.repeat(1, 1, self.repeats[0], self.repeats[1])
        # Pad back to GRID_SIZE
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

        # Pad back to GRID_SIZE
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
        # x is [1, 10, GRID_SIZE, GRID_SIZE]
        # Normalize: Permute channels
        x_norm = x[:, self.norm_perm, :, :]
        # Backbone logic
        out_norm = self.backbone(x_norm)
        # Denormalize: Permute channels back
        return out_norm[:, self.denorm_perm, :, :]


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
            x = -F.max_pool2d(
                -x, self.kernel_size, stride=1, padding=self.kernel_size // 2
            )
        return x


class DilateSolver(nn.Module):
    """Binary dilation via max-pool."""

    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.iterations):
            x = F.max_pool2d(
                x, self.kernel_size, stride=1, padding=self.kernel_size // 2
            )
        return x


class ProjectSolver(nn.Module):
    """Projects colors in a straight line until blocked by another color."""

    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        for _ in range(GRID_SIZE):
            if self.direction == "right":
                shifted = F.pad(out[:, :, :, :-1], (1, 0, 0, 0))
            elif self.direction == "left":
                shifted = F.pad(out[:, :, :, 1:], (0, 1, 0, 0))
            elif self.direction == "down":
                shifted = F.pad(out[:, :, :-1, :], (0, 0, 1, 0))
            elif self.direction == "up":
                shifted = F.pad(out[:, :, 1:, :], (0, 0, 0, 1))
            else:
                shifted = out

            # Only background cells can be overwritten
            bg_mask = (out[:, 0:1, :, :] > 0.5).float()

            # The foreground flowing in (Colors + IDs)
            fill_fg = shifted[:, 1:, :, :]
            has_color = (
                fill_fg[:, : COLOR_CHANNELS - 1, :, :].sum(dim=1, keepdim=True) > 0.5
            ).float()

            # Only update if the cell was bg AND there's a color flowing into it
            update_mask = bg_mask * has_color

            out_bg = torch.where(
                update_mask > 0.5, torch.zeros_like(out[:, 0:1]), out[:, 0:1]
            )
            out_fg = torch.where(
                update_mask.expand(-1, STATE_CHANNELS - 1, -1, -1) > 0.5,
                fill_fg,
                out[:, 1:],
            )
            out = torch.cat([out_bg, out_fg], dim=1)

        return out


class AdjacencyMaskSolver(nn.Module):
    """Returns the outer 1-pixel boundary of objects (newly adjacent pixels)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = F.max_pool2d(x[:, :COLOR_CHANNELS, :, :], 3, stride=1, padding=1)
        # New pixels are where dilated has color but original didn't
        adj_c = torch.clamp(d[:, 1:COLOR_CHANNELS] - x[:, 1:COLOR_CHANNELS], 0.0, 1.0)
        # Reconstruct background
        out_bg = 1.0 - adj_c.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        # IDs stay as they are (AdjacencyMask usually used for color generation)
        res = torch.cat([out_bg, adj_c, x[:, COLOR_CHANNELS:]], dim=1)
        return validate_state(res, "AdjacencyMask")


class BorderMaskSolver(nn.Module):
    """Returns only the internal 1-pixel boundary of existing objects."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = -F.max_pool2d(-x, 3, stride=1, padding=1)
        # Border is original minus eroded
        border_c = torch.clamp(x[:, 1:] - e[:, 1:], 0.0, 1.0)
        # Reconstruct background
        out_bg = 1.0 - border_c.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        res = torch.cat([out_bg, border_c], dim=1)
        return validate_state(res, "BorderMask")


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

    def __init__(
        self,
        flip_dim: int,
        mode: str,
        bg_color: int = 0,
        grid_h: int = GRID_SIZE,
        grid_w: int = GRID_SIZE,
    ) -> None:
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
        self.register_buffer("row_alive", r_mask)  # [1,1,GRID_SIZE,1]

        c_mask = torch.zeros(1, 1, 1, GRID_SIZE)
        c_mask[:, :, :, :grid_w] = 1.0
        self.register_buffer("col_alive", c_mask)  # [1,1,1,GRID_SIZE]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build static active mask [1, 1, GRID_SIZE, GRID_SIZE] from baked row/col alive vectors
        active_mask = self.row_alive * self.col_alive  # [1, 1, GRID_SIZE, GRID_SIZE]

        # To flip only the active subgrid:
        #   1. Zero-out padding so flipping doesn't drag padding content inside
        x_active = x * active_mask  # [1, 10, GRID_SIZE, GRID_SIZE], zeros outside

        #   2. Flip the whole tensor (active content flips within its region)
        if self.flip_dim == 2:
            # Vertical flip of active region: row i ↔ row (grid_h-1-i)
            # After flipping whole tensor: active content is at rows (GRID_SIZE-grid_h)..(GRID_SIZE-1)
            # We need to re-align it to rows 0..(grid_h-1) by rolling it up
            flipped_full = torch.flip(x_active, dims=[2])  # content at bottom
            # Roll up by (GRID_SIZE - grid_h) to bring content to row 0
            roll_amount = self.grid_h - GRID_SIZE  # negative = roll up
            flipped = torch.roll(flipped_full, shifts=roll_amount, dims=2)
        else:
            flipped_full = torch.flip(x_active, dims=[3])  # content at right
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

    def __init__(
        self,
        period: int,
        grid_h: int = GRID_SIZE,
        grid_w: int = GRID_SIZE,
        bg_color: int = 0,
    ) -> None:
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
        # x: [1, 10, GRID_SIZE, GRID_SIZE]
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

    def __init__(
        self, direction: str, bg_color: int = 0, iterations: int = GRID_SIZE
    ) -> None:
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
        new_x = torch.where(
            swap_mask > 0.5, shifted_x, torch.where(reverse_swap > 0.5, reverse_x, x)
        )

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
            "down": (1, 2),
            "up": (-1, 2),
            "right": (1, 3),
            "left": (-1, 3),
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
        colors = x[:, :COLOR_CHANNELS, :, :]
        bg = colors[:, self.bg_color : self.bg_color + 1, :, :]
        fg = (colors[:, 1:].sum(dim=1, keepdim=True) > 0.5).float()

        # Build the active bbox mask with tensor ops only.
        row_has_fg = fg.amax(dim=3)  # [B, 1, H]
        col_has_fg = fg.amax(dim=2)  # [B, 1, W]

        row_seen_from_top = (torch.cumsum(row_has_fg, dim=2) > 0).float()
        row_seen_from_bottom = torch.flip(
            (torch.cumsum(torch.flip(row_has_fg, dims=[2]), dim=2) > 0).float(),
            dims=[2],
        )
        col_seen_from_left = (torch.cumsum(col_has_fg, dim=2) > 0).float()
        col_seen_from_right = torch.flip(
            (torch.cumsum(torch.flip(col_has_fg, dims=[2]), dim=2) > 0).float(),
            dims=[2],
        )

        active_rows = row_seen_from_top * row_seen_from_bottom
        active_cols = col_seen_from_left * col_seen_from_right
        active_mask = active_rows.unsqueeze(3) * active_cols.unsqueeze(2)

        top_edge = active_rows * (
            1.0 - F.pad(active_rows[:, :, :-1], (1, 0), value=0.0)
        )
        bottom_edge = active_rows * (
            1.0 - F.pad(active_rows[:, :, 1:], (0, 1), value=0.0)
        )
        left_edge = active_cols * (
            1.0 - F.pad(active_cols[:, :, :-1], (1, 0), value=0.0)
        )
        right_edge = active_cols * (
            1.0 - F.pad(active_cols[:, :, 1:], (0, 1), value=0.0)
        )

        border_mask = (
            top_edge.unsqueeze(3)
            + bottom_edge.unsqueeze(3)
            + left_edge.unsqueeze(2)
            + right_edge.unsqueeze(2)
        ).clamp(0.0, 1.0)

        # Seed external reachability from background cells on the active bbox border.
        external = bg * border_mask

        for _ in range(self.iterations):
            prev = external
            spread_up = torch.cat(
                [torch.zeros_like(external[:, :, :1, :]), external[:, :, :-1, :]], dim=2
            )
            spread_down = torch.cat(
                [external[:, :, 1:, :], torch.zeros_like(external[:, :, :1, :])], dim=2
            )
            spread_left = torch.cat(
                [torch.zeros_like(external[:, :, :, :1]), external[:, :, :, :-1]], dim=3
            )
            spread_right = torch.cat(
                [external[:, :, :, 1:], torch.zeros_like(external[:, :, :, :1])], dim=3
            )
            spread = (spread_up + spread_down + spread_left + spread_right).clamp(0.0, 1.0)
            external = torch.max(external, bg * spread * active_mask)

        enclosed = (bg * active_mask) * (1.0 - external)

        out = x.clone()
        out[:, self.bg_color : self.bg_color + 1, :, :] = (
            out[:, self.bg_color : self.bg_color + 1, :, :] * (1.0 - enclosed)
        )
        out[:, self.fill_color : self.fill_color + 1, :, :] = (
            out[:, self.fill_color : self.fill_color + 1, :, :] * (1.0 - enclosed)
            + enclosed
        ).clamp(0.0, 1.0)

        return out


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
        # x: [1, 10+K, GRID_SIZE, GRID_SIZE]
        out = x.clone()

        for c in range(COLOR_CHANNELS):
            if c == self.bg_color:
                continue
            ch = x[:, c : c + 1, :, :]  # [1, 1, GRID_SIZE, GRID_SIZE]

            row_sum = ch.sum(dim=3, keepdim=True)
            col_sum = ch.sum(dim=2, keepdim=True)

            row_isolated = (row_sum > 0.5) * (row_sum < 1.5)
            col_isolated = (col_sum > 0.5) * (col_sum < 1.5)

            row_fill = row_isolated.expand_as(ch)
            col_fill = col_isolated.expand_as(ch)

            extended = (ch + row_fill + col_fill).clamp(0.0, 1.0)

            # Only clear bg where we've filled
            filled_mask = (extended - ch).clamp(0.0, 1.0)
            out[:, self.bg_color : self.bg_color + 1, :, :] = out[
                :, self.bg_color : self.bg_color + 1, :, :
            ] * (1.0 - filled_mask)
            out[:, c : c + 1, :, :] = extended

            # IDs? Since this is a fill, the ID channel for the source object should also fill.
            # (Simplification: We only care about color filling here).

        return out


class AntiTransposeSolver(nn.Module):
    """Flip over anti-diagonal: transpose then flip both axes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3).flip(2).flip(3)


class PerColorShiftSolver(nn.Module):
    """Translate each color channel independently by per-color integer offsets.

    offsets: {color: (dy, dx)} where dy=row-shift (+ = down), dx=col-shift (+ = right).
    """

    def __init__(self, offsets: dict[int, tuple[int, int]]) -> None:
        super().__init__()
        # Store as list[(dy, dx)] indexed by color
        self.offsets: list[tuple[int, int]] = [
            offsets.get(c, (0, 0)) for c in range(10)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == STATE_CHANNELS, (
            f"State corruption: expected {STATE_CHANNELS} channels, got {x.shape[1]}"
        )
        colors = x[:, :COLOR_CHANNELS, :, :]
        ids = x[:, COLOR_CHANNELS:, :, :]

        shifted_colors = []
        for c in range(COLOR_CHANNELS):
            dy, dx = self.offsets[c]
            shifted = self._shift_channel(colors[:, c : c + 1, :, :], dy, dx)
            shifted_colors.append(shifted)

        shifted_colors_tensor = torch.cat(shifted_colors, dim=1)
        return torch.cat([shifted_colors_tensor, ids], dim=1)

    @staticmethod
    def _shift_channel(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        shifted = x

        if dy > 0:
            shifted = F.pad(shifted[:, :, :-dy, :], (0, 0, dy, 0))
        elif dy < 0:
            up = -dy
            shifted = F.pad(shifted[:, :, up:, :], (0, 0, 0, up))

        if dx > 0:
            shifted = F.pad(shifted[:, :, :, :-dx], (dx, 0, 0, 0))
        elif dx < 0:
            left = -dx
            shifted = F.pad(shifted[:, :, :, left:], (0, left, 0, 0))

        return shifted


class PerLineageShiftSolver(nn.Module):
    """
    Independently shifts specific identity lineages (channels 10+).
    offsets: {lineage_idx: (dy, dx)} where lineage_idx is relative to COLOR_CHANNELS.
    """

    def __init__(self, offsets: dict[int, tuple[int, int]]) -> None:
        super().__init__()
        # Store as list in buffer for ONNX
        self.register_buffer("dy", torch.zeros(IDENTITY_CHANNELS))
        self.register_buffer("dx", torch.zeros(IDENTITY_CHANNELS))
        for idx, (y, x) in offsets.items():
            if 0 <= idx < IDENTITY_CHANNELS:
                self.dy[idx] = float(y)
                self.dx[idx] = float(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Separate Colors and IDs
        colors = x[:, :COLOR_CHANNELS, :, :]
        ids = x[:, COLOR_CHANNELS:, :, :]

        # We need to shift the colors WITH the IDs.
        # This is tricky: we only want to shift the color pixels that BELONG to this ID.
        new_colors = colors.clone()
        new_ids = ids.clone()

        for i in range(IDENTITY_CHANNELS):
            dy_val = int(self.dy[i].item())
            dx_val = int(self.dx[i].item())
            if dy_val == 0 and dx_val == 0:
                continue

            # 1. Mask for this lineage
            mask = ids[:, i : i + 1, :, :]

            # 2. Extract color pixels belonging to ONLY this lineage
            # (Caution: overlapping lineages might exist)
            lin_colors = colors * mask

            # 3. Shift both
            shifted_colors = ShiftSolver(dx_val, dy_val)(lin_colors)
            shifted_mask = ShiftSolver(dx_val, dy_val)(mask)

            # 4. Update state (Subtract old, Add new)
            # This is a bit lossy if we don't handle collisions but good for finisher.
            new_colors = (new_colors - lin_colors + shifted_colors).clamp(0, 1)
            new_ids[:, i : i + 1, :, :] = shifted_mask

        return torch.cat([new_colors, new_ids], dim=1)


class ExtremeObjectSelector(nn.Module):
    """
    Selects a specific colored object channel based on extreme properties.
    Criteria: 'largest', 'smallest', 'topmost', 'bottommost', 'leftmost', 'rightmost'
    """

    def __init__(self, criterion: str = "largest") -> None:
        super().__init__()
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fg: [1, 9, H, W] (colors 1-9)
        fg = x[:, 1:, :, :]

        if self.criterion == "largest":
            scores = fg.sum(dim=(2, 3))  # [1, 9] (areas)
        elif self.criterion == "smallest":
            # area + large constant for empty channels
            areas = fg.sum(dim=(2, 3))
            scores = -(areas + (1.0 - (areas > 0).float()) * 1000.0)
        elif self.criterion == "topmost":
            # Mask * row_index, then find min
            rows = (
                torch.arange(GRID_SIZE, device=x.device)
                .view(1, 1, GRID_SIZE, 1)
                .float()
            )
            # We want topmost color, so we look for min row index that has 1
            # scores = - min_row
            scores = -(fg * rows).sum(dim=(2, 3)) / (fg.sum(dim=(2, 3)) + 1e-6)
        else:
            scores = fg.sum(dim=(2, 3))

        # Find max score channel
        max_scores = scores.max(dim=1, keepdim=True)[0]
        target_mask = (scores == max_scores).float()

        # In case of ties, pick first
        # (This is a bit loose for ONNX but usually one-hot works)
        # v9: Return full state (filtered by choice)
        mask = (target_mask.unsqueeze(2).unsqueeze(3) * fg).sum(dim=1, keepdim=True)
        res = x * mask.expand_as(x)
        return validate_state(res.clamp(0, 1), "ExtremeObjectSelector")


class RankPermutationSolver(nn.Module):
    """
    Reassigns colors to object channels based on their size rank.
    permutation: [9] indices. e.g. [2, 0, 1...] means largest gets color of 2nd channel, etc.
    """

    def __init__(self, permutation: list[int]) -> None:
        super().__init__()
        self.register_buffer("p", torch.tensor(permutation, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Identify ranks
        counts = x[:, 1:COLOR_CHANNELS, :, :].sum(dim=(2, 3))  # [1, 9]
        c1 = counts.unsqueeze(2)
        c2 = counts.unsqueeze(1)
        ranks = (c2 > c1).float().sum(dim=2).long()  # [1, 9]

        batch_size = x.shape[0]
        # 2. Map channels to their new target colors (Batch-aware)
        p_batched = self.p.unsqueeze(0).expand(batch_size, -1)
        new_colors_idx = torch.gather(p_batched, 1, ranks)  # [B, 9]

        # 3. Scatter back into [1, 10+K, H, W]
        batch, _, h, w = x.shape
        target_one_hot = F.one_hot(new_colors_idx, num_classes=9).float()

        x_fg = x[:, 1:COLOR_CHANNELS, :, :].reshape(batch, 9, -1)
        res_fg = torch.matmul(target_one_hot.transpose(1, 2), x_fg)
        res_fg = res_fg.view(batch, 9, h, w)

        bg = (1.0 - res_fg.sum(dim=1, keepdim=True)).clamp(0, 1)

        # Preserve IDs
        ids = x[:, COLOR_CHANNELS:, :, :]
        return torch.cat([bg, res_fg, ids], dim=1)


class BorderAwareMaskSolver(nn.Module):
    """Mask showing objects that touch the border."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 10, GRID_SIZE, GRID_SIZE]
        fg = x[:, 1:, :, :]
        # Check edges
        top = fg[:, :, 0, :].sum(dim=2)
        bottom = fg[:, :, -1, :].sum(dim=2)
        left = fg[:, :, :, 0].sum(dim=2)
        right = fg[:, :, :, -1].sum(dim=2)

        touches = ((top + bottom + left + right) > 0.5).float()  # [B, 9]
        mask = (touches.unsqueeze(2).unsqueeze(3) * fg).sum(dim=1, keepdim=True)

        # v9: Return full state
        res = x * mask.expand_as(x)
        return validate_state(res.clamp(0, 1), "BorderAwareMask")


class CollisionProjectSolver(nn.Module):
    """Directional projection that stops before overlapping non-zero background."""

    def __init__(self, direction: str) -> None:
        super().__init__()
        self.direction = direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        # Original occupancy (cannot be written into)
        occupied = (x.sum(dim=1, keepdim=True) > 0.5).float()

        for _ in range(GRID_SIZE):
            if self.direction == "right":
                shifted = F.pad(out[:, :, :, :-1], (1, 0, 0, 0))
            elif self.direction == "left":
                shifted = F.pad(out[:, :, :, 1:], (0, 1, 0, 0))
            elif self.direction == "down":
                shifted = F.pad(out[:, :, :-1, :], (0, 0, 1, 0))
            elif self.direction == "up":
                shifted = F.pad(out[:, :, 1:, :], (0, 0, 0, 1))
            else:
                shifted = out

            # shifted pixels that land on non-empty space must be blocked
            # We use shifted_background_channel as the indicator
            shifted_fg = shifted[:, 1:, :, :]
            # Land mask: where shifted pixels TRY to go
            try_mask = (shifted_fg.sum(dim=1, keepdim=True) > 0.5).float()

            # Blocked if target cell is ALREADY occupied in the ORIGINAL input
            blocked = try_mask * occupied

            # For each pixel, if blocked, we don't bring the color over.
            # (In reality, for ProjectSolver, we want to stop the WHOLE line, but
            # local blocking is a good enough primitive for composition).
            allowed_shift = shifted_fg * (1.0 - blocked)

            # Update background and colors
            out_colors = (out[:, 1:, :, :] + allowed_shift).clamp(0, 1)
            out_bg = (1.0 - out_colors.sum(dim=1, keepdim=True)).clamp(0, 1)
            out = torch.cat([out_bg, out_colors], dim=1)

        return out


class CenterObjectSolver(nn.Module):
    """Crops the object in the mask and places it in the center of the grid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: Move center of mass to center of grid
        # [1, 10, H, W]
        fg = (x.sum(dim=1, keepdim=True) > 0.5).float()
        indices_h = (
            torch.arange(GRID_SIZE, device=x.device).view(1, 1, GRID_SIZE, 1).float()
        )
        indices_w = (
            torch.arange(GRID_SIZE, device=x.device).view(1, 1, 1, GRID_SIZE).float()
        )

        count = fg.sum() + 1e-6
        mean_h = (fg * indices_h).sum() / count
        mean_w = (fg * indices_w).sum() / count

        dy = int(GRID_SIZE // 2 - mean_h.item())
        dx = int(GRID_SIZE // 2 - mean_w.item())

        # Use existing shift logic
        shifted = x
        if dy > 0:
            shifted = F.pad(shifted[:, :, :-dy, :], (0, 0, dy, 0))
        elif dy < 0:
            shifted = F.pad(shifted[:, :, -dy:, :], (0, 0, 0, -dy))
        if dx > 0:
            shifted = F.pad(shifted[:, :, :, :-dx], (dx, 0, 0, 0))
        elif dx < 0:
            shifted = F.pad(shifted[:, :, :, -dx:], (0, -dx, 0, 0))

        return shifted


class PropertyColorizer(nn.Module):
    """Sets a specific color to a channel based on its count/height."""

    def __init__(self, property_name: str, target_color: int) -> None:
        super().__init__()
        self.property_name = property_name
        self.target_color = target_color

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logic: If property matches (detected during search), return recolored mask
        # For simplicity in synthesis, we'll make this a direct recolor of the active mask
        mask = (x.sum(dim=1, keepdim=True) > 0.5).float()
        out_bg = 1.0 - mask
        channels = [torch.zeros_like(mask) for _ in range(10)]
        channels[0] = out_bg
        channels[self.target_color] = mask
        return torch.cat(channels, dim=1)


class OverlaySolver(nn.Module):
    """
    Combines two 10+K states using Foreground > Background rules.
    Color Rule: Highest color index wins if both are foreground.
    Identity Rule: Cumulative Union (pixels inherit IDs from both objects).
    """

    def forward(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        # Separate components
        colors_a = state_a[:, :COLOR_CHANNELS, :, :]
        ids_a = state_a[:, COLOR_CHANNELS:, :, :]

        colors_b = state_b[:, :COLOR_CHANNELS, :, :]
        ids_b = state_b[:, COLOR_CHANNELS:, :, :]

        # Foreground masks (anything not color 0)
        fg_a = (colors_a[:, 1:, :, :].sum(dim=1, keepdim=True) > 0.5).float()
        fg_b = (colors_b[:, 1:, :, :].sum(dim=1, keepdim=True) > 0.5).float()

        # Collision Logic: b over a
        # 1. New Colors
        # Use b where b is foreground, else use a
        new_colors = colors_b * fg_b + colors_a * (1.0 - fg_b)

        # 2. Cumulative Identities (Lineage)
        new_ids_raw = torch.max(ids_a, ids_b)  # Lineage union

        # Identity Entropy Control (Hardening Phase)
        # Rule: If an object absorbs a 'tiny' contributor (< 5% territory),
        # we drop the tiny contributor's ID from the union to prevent 'Mega-object' drift.
        areas_a = ids_a.sum(dim=(2, 3), keepdim=True)
        areas_b = ids_b.sum(dim=(2, 3), keepdim=True)

        # If area_b is insignificant compared to area_a, we use a's IDs only
        # This is a bit extreme, let's do it per channel.
        # Smarter pruning: Drop bits that have very small area relative to the whole union area.
        union_area = new_ids_raw.sum(dim=(2, 3), keepdim=True)
        significance = areas_b / (union_area + 1e-6)

        # Pruning mask: only keep ids_b bits if they are > 0.1 significant
        pruned_ids_b = torch.where(significance > 0.1, ids_b, torch.zeros_like(ids_b))
        new_ids = torch.max(ids_a, pruned_ids_b)

        # Ensure we don't have stray ID bits where there is NO color in the end
        new_fg = (new_colors[:, 1:, :, :].sum(dim=1, keepdim=True) > 0.5).float()
        new_ids = new_ids * new_fg

        return torch.cat([new_colors, new_ids], dim=1)


class ConditionalCompositionSolver(nn.Module):
    """Branching logic: mask * A(x) + (1-mask) * B(x)."""

    def __init__(
        self, mask_solver: nn.Module, solver_a: nn.Module, solver_b: nn.Module
    ) -> None:
        super().__init__()
        self.mask_solver = mask_solver
        self.solver_a = solver_a
        self.solver_b = solver_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == STATE_CHANNELS, (
            f"State corruption: expected {STATE_CHANNELS} channels, got {x.shape[1]}"
        )
        m_state = self.mask_solver(x)  # [B, 26, H, W]
        # v9: Extract mask from any active identity channel
        m = (m_state[:, COLOR_CHANNELS:, :, :].sum(dim=1, keepdim=True) > 0.5).float()

        a = self.solver_a(x)
        b = self.solver_b(x)

        # Explicitly expand mask to match full state channels
        m_exp = m.expand_as(a)
        res = m_exp * a + (1.0 - m_exp) * b
        return validate_state(res, "ConditionalComposition")


class RelationMatrixSolver(nn.Module):
    """
    Computes a [1, K, K, R] relation matrix between identity channels.
    Relations: 0:Touching, 1:Inside
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ids = x[:, COLOR_CHANNELS:, :, :]
        K = ids.shape[1]

        # 1. Areas [1, K, 1, 1]
        areas = ids.sum(dim=(2, 3), keepdim=True)

        # 2. Vectorized Touching (using Dilation + Batch Dot Product)
        # [1, K, H, W] -> dilate -> [1, K, H, W]
        dilated = F.max_pool2d(ids, 3, stride=1, padding=1)
        # Using bmm (batch matrix multiplication) or einsum to get [1, K, K] interactions
        # (B, K, HW) x (B, HW, K) -> (B, K, K)
        ids_flat = ids.reshape(B, K, -1)
        dilated_flat = dilated.reshape(B, K, -1)

        touch_matrix = torch.bmm(dilated_flat, ids_flat.transpose(1, 2))  # [1, K, K]
        touch_matrix = (touch_matrix > 0.5).float()

        # 3. Vectorized Inside (Intersection / Area)
        intersect_matrix = torch.bmm(ids_flat, ids_flat.transpose(1, 2))  # [B, K, K]

        # areas_row: [B, K, 1], repeat to match [B, K, K]
        areas_vec = areas.reshape(B, K, 1)  # Use dynamic B, K
        inside_matrix = (intersect_matrix >= (areas_vec - 0.1)).float()

        # Mask out diagonals (identity relation is boring)
        eye = torch.eye(K, device=x.device).reshape(1, K, K).expand(B, -1, -1)
        touch_matrix = touch_matrix * (1.0 - eye)
        inside_matrix = inside_matrix * (1.0 - eye)

        # Pack into [B, 2, K, K] for the SearchBrain
        return torch.stack([touch_matrix, inside_matrix], dim=1)


class AnchorFactory(nn.Module):
    """
    Generates single-pixel masks for various spatial interest points.
    """

    def __init__(self, mode: str = "object_center") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor, target_id: int) -> torch.Tensor:
        mask = x[:, COLOR_CHANNELS + target_id : COLOR_CHANNELS + target_id + 1, :, :]

        if self.mode == "object_center":
            h_indices = (
                torch.arange(GRID_SIZE, device=x.device)
                .view(1, 1, GRID_SIZE, 1)
                .float()
            )
            w_indices = (
                torch.arange(GRID_SIZE, device=x.device)
                .view(1, 1, 1, GRID_SIZE)
                .float()
            )

            count = mask.sum() + 1e-6
            mean_h = (mask * h_indices).sum() / count
            mean_w = (mask * w_indices).sum() / count

            dist_h = torch.abs(h_indices - mean_h)
            dist_w = torch.abs(w_indices - mean_w)
            anchor = ((dist_h < 0.5) & (dist_w < 0.5)).float()
            return anchor

        elif self.mode == "grid_center":
            anchor = torch.zeros_like(mask)
            anchor[:, :, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0
            return anchor

        return torch.zeros_like(mask)


class RoleAssigner(nn.Module):
    """
    Heuristic pass to label Identity channels as 'Anchor' vs 'Payload'.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        ids = x[:, COLOR_CHANNELS:, :, :]
        areas = ids.sum(dim=(2, 3))  # [B, K]
        max_area = areas.max(dim=1, keepdim=True)[0]
        return (areas == max_area).float()


class LargestObjectSelector(nn.Module):
    """Select the largest object (by pixel count) and return it as output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        colors = x[:, :COLOR_CHANNELS, :, :]

        # Find areas for each color
        areas = colors[:, 1:, :, :].sum(dim=(2, 3))  # Skip bg

        # Get largest
        max_area, max_idx = areas.max(dim=1, keepdim=True)

        # Create mask for largest color
        mask = colors[:, 1:, :, :].gather(
            1, max_idx.unsqueeze(2).unsqueeze(3).expand(-1, 1, GRID_SIZE, GRID_SIZE)
        )

        # Reconstruct output with just the largest object
        out = torch.zeros_like(x)
        out[:, 0:1, :, :] = 1.0 - mask  # bg
        out[:, 1:2, :, :] = mask.clamp(0, 1)

        return out


class SmallestObjectSelector(nn.Module):
    """Select the smallest object (by pixel count) and return it as output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        colors = x[:, :COLOR_CHANNELS, :, :]

        areas = colors[:, 1:, :, :].sum(dim=(2, 3))
        areas = torch.where(areas > 0, areas, torch.full_like(areas, 1e6))

        min_area, min_idx = areas.min(dim=1, keepdim=True)

        mask = colors[:, 1:, :, :].gather(
            1, min_idx.unsqueeze(2).unsqueeze(3).expand(-1, 1, GRID_SIZE, GRID_SIZE)
        )

        out = torch.zeros_like(x)
        out[:, 0:1, :, :] = 1.0 - mask
        out[:, 1:2, :, :] = mask.clamp(0, 1)

        return out


class ProjectUntilCollision(nn.Module):
    """Project each object in a direction until it hits another object or boundary."""

    def __init__(self, direction: str = "right") -> None:
        super().__init__()
        self.direction = direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()

        # Get foreground (non-background)
        fg = x[:, 1:, :, :].sum(dim=1, keepdim=True).clamp(0, 1)

        # Project in direction using repeated shifts
        for _ in range(GRID_SIZE):
            if self.direction == "right":
                shifted = torch.cat(
                    [torch.zeros_like(fg[:, :, :, :1]), fg[:, :, :, :-1]], dim=3
                )
            elif self.direction == "left":
                shifted = torch.cat(
                    [fg[:, :, :, 1:], torch.zeros_like(fg[:, :, :, :1])], dim=3
                )
            elif self.direction == "down":
                shifted = torch.cat(
                    [torch.zeros_like(fg[:, :, :1, :]), fg[:, :, :-1, :]], dim=2
                )
            elif self.direction == "up":
                shifted = torch.cat(
                    [fg[:, :, 1:, :], torch.zeros_like(fg[:, :, :1, :])], dim=2
                )
            else:
                shifted = fg

            # Only move into empty space
            can_move = (fg * (1 - shifted)).clamp(0, 1)
            fg = fg + can_move * shifted
            fg = fg.clamp(0, 1)

        # Apply to all color channels
        for c in range(1, COLOR_CHANNELS):
            color_mask = x[:, c : c + 1, :, :]
            moved = color_mask * fg
            out[:, c : c + 1, :, :] = out[:, c : c + 1, :, :] * (1 - fg) + moved

        return validate_state(out, "ProjectCollision")


class ConditionalCompositionSolver(nn.Module):
    """Conditionally apply one solver or another based on mask.

    out = mask * solver_a(x) + (1 - mask) * solver_b(x)
    """

    def __init__(self, mask_solver, solver_a, solver_b) -> None:
        super().__init__()
        self.mask_solver = mask_solver
        self.solver_a = solver_a
        self.solver_b = solver_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.mask_solver(x)
        out_a = self.solver_a(x)
        out_b = self.solver_b(x)

        # Apply mask
        result = mask * out_a + (1 - mask) * out_b
        return validate_state(result, "Conditional")


class SnapToGridCenter(nn.Module):
    """Snap all objects to the center of the grid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        colors = x[:, :COLOR_CHANNELS, :, :]

        out = torch.zeros_like(x)

        for c in range(1, COLOR_CHANNELS):
            color_mask = colors[:, c : c + 1, :, :]

            # Get centroid
            h_idx = (
                torch.arange(GRID_SIZE, device=x.device)
                .float()
                .view(1, 1, GRID_SIZE, 1)
            )
            w_idx = (
                torch.arange(GRID_SIZE, device=x.device)
                .float()
                .view(1, 1, 1, GRID_SIZE)
            )

            count = color_mask.sum() + 1e-6
            center_h = (color_mask * h_idx).sum() / count
            center_w = (color_mask * w_idx).sum() / count

            # Calculate shift to center
            target_h, target_w = GRID_SIZE / 2, GRID_SIZE / 2
            dy = int(target_h - center_h.item())
            dx = int(target_w - center_w.item())

            # Shift
            shifted = torch.roll(color_mask, shifts=(dy, dx), dims=(2, 3))
            out[:, c : c + 1, :, :] = shifted

        # Background
        out[:, 0:1, :, :] = 1.0 - out[:, 1:, :, :].sum(dim=1, keepdim=True).clamp(0, 1)

        return validate_state(out, "SnapToCenter")


class ExtractCenter(nn.Module):
    """Extract the center portion of the grid."""

    def __init__(self, size: int = 3) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = GRID_SIZE, GRID_SIZE
        start_h = (h - self.size) // 2
        start_w = (w - self.size) // 2

        center = x[:, :, start_h : start_h + self.size, start_w : start_w + self.size]

        # Pad back
        out = torch.zeros_like(x)
        out[:, :, start_h : start_h + self.size, start_w : start_w + self.size] = center

        return validate_state(out, "ExtractCenter")
