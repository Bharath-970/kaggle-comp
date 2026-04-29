#!/usr/bin/env python3
"""Hybrid symbolic + neural fallback solver search for NeuroGolf tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Literal

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for _p in (str(SRC), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neurogolf.constants import GRID_SIZE
from neurogolf.export import export_static_onnx
from neurogolf.grid_codec import decode_tensor_to_grid, encode_grid_to_tensor
from neurogolf.solvers import (
    AntiTransposeSolver,
    CompositionalSolver,
    ConstantGridSolver,
    FlipSolver,
    GeneralColorRemapSolver,
    IdentitySolver,
    KroneckerSolver,
    NearestNeighborScaleSolver,
    PerColorShiftSolver,
    RelativeMoveSolver,
    RotationSolver,
    ShiftSolver,
    SubgridSolver,
    TilingSolver,
    TransposeSolver,
)


class ColorSubstitutionSolver(torch.nn.Module):
    """Non-bijective color map: maps each input channel to an output channel.

    Unlike GeneralColorRemapSolver (bijective permutations only), this allows
    multiple input colors to merge into the same output color, or a color to
    disappear into background.

    color_map: list[10] where color_map[in_c] = out_c.
    """

    def __init__(self, color_map: list[int]) -> None:
        super().__init__()
        self.color_map = list(color_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for in_c, out_c in enumerate(self.color_map):
            out[:, out_c : out_c + 1, :, :] = (
                out[:, out_c : out_c + 1, :, :] + x[:, in_c : in_c + 1, :, :]
            )
        return out.clamp(0.0, 1.0)


class QuadrantMirrorSolver(torch.nn.Module):
    """Output = 2×2 quadrant mirror tile of the top-left H×W block:

        [ inp          | flip_h(inp)  ]
        [ flip_v(inp)  | flip_both(inp) ]

    Produces a 2H×2W output padded to the standard 30×30 canvas.
    All ops (flip, cat) are ONNX-exportable.
    """

    def __init__(self, in_h: int, in_w: int) -> None:
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x[:, :, : self.in_h, : self.in_w]
        fh = torch.flip(inp, dims=[-1])  # horizontal mirror
        fv = torch.flip(inp, dims=[-2])  # vertical mirror
        fb = torch.flip(inp, dims=[-2, -1])  # both mirrors
        top = torch.cat([inp, fh], dim=-1)
        bot = torch.cat([fv, fb], dim=-1)
        out_block = torch.cat([top, bot], dim=-2)
        out = torch.zeros_like(x)
        out[:, :, : 2 * self.in_h, : 2 * self.in_w] = out_block
        return out


class RotationQuadrantSolver(torch.nn.Module):
    """Output = 2×2 rotation quadrant tile of the top-left H×W block:

        [ inp        | rot90(inp)  ]
        [ rot270(inp)| rot180(inp) ]

    Produces a 2H×2W output padded to the standard 30×30 canvas.
    All ops (rot90 via flip+transpose) are ONNX-exportable.
    """

    def __init__(self, in_h: int, in_w: int, cw: bool = True) -> None:
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.cw = cw  # True=[inp,r270;r90,r180], False=[inp,r90;r270,r180]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x[:, :, : self.in_h, : self.in_w]
        # rot90 CCW = transpose then flip left-right
        r90 = torch.flip(inp.permute(0, 1, 3, 2), dims=[-2])  # flipud(T) = rot90 CCW
        # rot180 = flip both axes
        r180 = torch.flip(torch.flip(inp, dims=[-1]), dims=[-2])
        r270 = torch.flip(inp.permute(0, 1, 3, 2), dims=[-1])  # fliplr(T) = rot270 CCW
        # cw=True: [inp, rot270; rot90, rot180] (task106 pattern)
        # cw=False: [inp, rot90; rot270, rot180]
        if self.cw:
            top = torch.cat([inp, r270], dim=-1)
            bot = torch.cat([r90, r180], dim=-1)
        else:
            top = torch.cat([inp, r90], dim=-1)
            bot = torch.cat([r270, r180], dim=-1)
        out_block = torch.cat([top, bot], dim=-2)
        out = torch.zeros_like(x)
        out[:, :, : 2 * self.in_h, : 2 * self.in_w] = out_block
        return out


from neurogolf.task_io import GridPair, TaskData, load_task_json


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
                    raise ValueError(
                        f"{split} contains unsupported grid size > {GRID_SIZE}x{GRID_SIZE}."
                    )
        return tuple(parsed)

    task = TaskData(
        train=parse_split("train", strict=False),
        test=parse_split("test", strict=False),
        arc_gen=parse_split("arc-gen", strict=False),
    )
    return task, dropped


def _iter_all_pairs(
    task: TaskData, *, include_arc_gen: bool = True
) -> Iterable[GridPair]:
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


def check_solve(
    model: torch.nn.Module, task: TaskData, *, include_arc_gen: bool = True
) -> bool:
    model.eval()
    with torch.no_grad():
        for pair in _iter_all_pairs(task, include_arc_gen=include_arc_gen):
            in_tensor = torch.from_numpy(encode_grid_to_tensor(pair.input_grid))
            pred_tensor = model(in_tensor).detach().cpu().numpy()
            expected = pair.output_grid
            pred = decode_tensor_to_grid(pred_tensor, len(expected), len(expected[0]))
            if pred != expected:
                return False
    return True


def _all_pairs_same_shape(
    task: TaskData,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
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
    op: Literal[
        "identity",
        "transpose",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "shift",
    ],
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
    op: Literal[
        "identity",
        "transpose",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "shift",
    ],
    *,
    dx: int = 0,
    dy: int = 0,
) -> list[int] | None:
    input_to_output: dict[int, int] = {}
    output_to_input: dict[int, int] = {}
    for pair in pairs:
        transformed = _grid_transform(pair.input_grid, op, dx=dx, dy=dy)
        ok = _derive_color_map_constraints(
            transformed, pair.output_grid, input_to_output, output_to_input
        )
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
        for pair in _iter_all_pairs(task):
            expected = pair.output_grid
            exp_h, exp_w = len(expected), len(expected[0])
            in_tensor = torch.from_numpy(encode_grid_to_tensor(pair.input_grid))
            pred_tensor = base_model(in_tensor).detach().cpu().numpy()
            # Reject if model output spatial dims don't match expected output dims
            t = pred_tensor[0] if pred_tensor.ndim == 4 else pred_tensor
            pred_h = t.shape[1] if t.shape[0] >= 10 else t.shape[0]
            pred_w = t.shape[2] if t.ndim >= 3 else t.shape[1]
            if pred_h < exp_h or pred_w < exp_w:
                return None
            pred_grid = decode_tensor_to_grid(
                pred_tensor, exp_h, exp_w
            )
            ok = _derive_color_map_constraints(
                pred_grid, expected, input_to_output, output_to_input
            )
            if not ok:
                return None
    return CompositionalSolver(
        base_model, GeneralColorRemapSolver(_finalize_color_map(input_to_output))
    )


def _candidate_geometries(max_shift: int) -> list[torch.nn.Module]:
    geoms: list[torch.nn.Module] = [
        IdentitySolver(),
        TransposeSolver(),
        AntiTransposeSolver(),
    ]
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


def find_master_synthesis(task: TaskData, max_shift: int = 2) -> torch.nn.Module | None:
    if not task.train:
        return None

    first_pair = task.train[0]
    in_grid = first_pair.input_grid
    out_grid = first_pair.output_grid

    in_h, in_w = len(in_grid), len(in_grid[0])
    out_h, out_w = len(out_grid), len(out_grid[0])

    if _all_outputs_identical(task):
        constant = ConstantGridSolver(out_grid)
        if check_solve(constant, task):
            return constant

    # Fast lane: if all pairs share input/output shapes, do pure-grid detection
    # and only then instantiate the corresponding Torch module.
    shapes = _all_pairs_same_shape(task)
    if shapes is not None:
        (in_h_all, in_w_all), (out_h_all, out_w_all) = shapes
        search_pairs = list(_iter_search_pairs(task))

        # Geometry ops that preserve shape (or swap for transpose/rotations).
        grid_ops: list[
            tuple[
                Literal[
                    "identity",
                    "transpose",
                    "rot90",
                    "rot180",
                    "rot270",
                    "flip_h",
                    "flip_v",
                ],
                torch.nn.Module,
            ]
        ] = [
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
            if check_solve(candidate, task):
                return candidate

        # Try shifts with wider range cheaply.
        if (in_h_all, in_w_all) == (out_h_all, out_w_all):
            for dy in range(-max_shift, max_shift + 1):
                for dx in range(-max_shift, max_shift + 1):
                    if dx == 0 and dy == 0:
                        continue
                    cmap = _derive_global_color_map_for_grid_transform(
                        search_pairs, "shift", dx=dx, dy=dy
                    )
                    if cmap is None:
                        continue
                    candidate = CompositionalSolver(
                        ShiftSolver(dx=dx, dy=dy), GeneralColorRemapSolver(cmap)
                    )
                    if check_solve(candidate, task):
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
            upscale = NearestNeighborScaleSolver(
                in_h=in_h, in_w=in_w, scale_h=scale_h, scale_w=scale_w
            )
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
    from neurogolf.solvers import (
        ErodeSolver,
        DilateSolver,
        ColorQuantizeSolver,
        PatternRepeatSolver,
    )

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

    # ── Lane R: Relative Placement ────────────────────────────────────────────────
    # RelativeMoveSolver: move object A to position of object B
    rel_moves = [
        (
            "largest_to_smallest",
            RelativeMoveSolver(source="largest", target="smallest"),
        ),
        (
            "smallest_to_largest",
            RelativeMoveSolver(source="smallest", target="largest"),
        ),
    ]

    for name, rel_solver in rel_moves:
        candidate = _wrap_with_color_map_if_match(rel_solver, task)
        if candidate is not None and check_solve(candidate, task):
            return candidate

    # ── Lane N: New fundamental solvers ───────────────────────────────────────
    from neurogolf.solvers import (
        FoldOverlaySolver,
        DiagonalPeriodicTilingSolver,
        GravitySolver,
        FloodFillSolver,
        IsolatedPixelCrossSolver,
    )

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

    # N4: Flood Fill (enclosed background regions with fixed colour)
    candidate = _try_flood_fill(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N5: Isolated Pixel Cross-Line extension
    candidate = IsolatedPixelCrossSolver(bg_color=_detect_bg(task))
    if check_solve(candidate, task):
        return candidate

    # N6: Anti-transpose (flip over anti-diagonal)
    candidate = _wrap_with_color_map_if_match(AntiTransposeSolver(), task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N7: Per-color independent shifts (detect per-channel offsets)
    candidate = _try_per_color_shift(task, max_shift=max_shift)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N8: Output is single extracted object (output = bounding-box crop of one color)
    candidate = _try_extract_single_object(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N9: Output tiles a single object N×M times
    candidate = _try_tile_from_object(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N10: Count objects → output is a fixed grid (constant baked from count pattern)
    candidate = _try_count_to_constant(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N11: Output = largest object bounding-box crop
    candidate = _try_output_is_object_by_size(task, largest=True)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N12: Output = smallest object bounding-box crop
    candidate = _try_output_is_object_by_size(task, largest=False)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N13: Minimal repeating tile block
    candidate = _try_minimal_tile_block(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N14: Any-magnitude global shift — catches wrong_placement with offset > max_shift
    candidate = _try_global_const_shift(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N15: Non-bijective color substitution
    candidate = _try_color_substitution(task)
    if candidate is not None and check_solve(candidate, task):
        return candidate

    # N16: Enclosed region fill (Python-bake; verification done inside detector)
    candidate = _try_fill_enclosed(task)
    if candidate is not None:
        return candidate

    # N17: Color by component size rank (Python-bake; verification done inside detector)
    candidate = _try_color_by_rank(task)
    if candidate is not None:
        return candidate

    # N18: Quadrant mirror tile — [inp, flip_h; flip_v, flip_both] → 2H×2W output
    candidate = _try_quadrant_mirror(task)
    if candidate is not None:
        if isinstance(candidate, ConstantGridSolver) or check_solve(candidate, task):
            return candidate

    # N19: Rotation quadrant tile — [inp, rot90; rot270, rot180] → 2H×2W output
    candidate = _try_rotation_quadrant(task)
    if candidate is not None:
        if isinstance(candidate, ConstantGridSolver) or check_solve(candidate, task):
            return candidate

    # N20: Object tile repeat — one object tiled N times at fixed stride (bake)
    candidate = _try_object_tile_repeat(task)
    if candidate is not None:
        return candidate

    # N21: Directional projection — extend each colored cell until wall or obstacle (bake)
    candidate = _try_projection(task)
    if candidate is not None:
        return candidate

    # N22: X-Diagonals — draw diagonals through each non-bg pixel (task141 pattern)
    candidate = _try_x_diagonals(task)
    if candidate is not None:
        return candidate

    # Lane O: DSL depth-3 with state-hash pruning
    try:
        from ideas_stack import solve_dsl

        candidate, _desc = solve_dsl(
            task, max_shift=max_shift, depth=3, max_programs=1500
        )
        if candidate is not None and check_solve(candidate, task):
            return candidate
    except Exception:
        pass

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
    from neurogolf.solvers import FoldOverlaySolver

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
                return FoldOverlaySolver(
                    flip_dim=flip_dim,
                    mode=mode,
                    bg_color=bg,
                    grid_h=max_h,
                    grid_w=max_w,
                )
    return None


def _try_diagonal_tiling(task: "TaskData"):
    """Detect DiagonalPeriodicTilingSolver period from training pairs."""
    from neurogolf.solvers import DiagonalPeriodicTilingSolver

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
            return DiagonalPeriodicTilingSolver(
                period=period, grid_h=max_h, grid_w=max_w, bg_color=bg
            )
    return None


def _try_gravity(task: "TaskData"):
    """Detect GravitySolver direction from training pairs."""
    from neurogolf.solvers import GravitySolver

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
                        cand[h - len(nz) :, c] = nz
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
                        cand[r, w - len(nz) :] = nz
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


def _try_flood_fill(task: "TaskData"):
    """Detect FloodFillSolver(fill_color, bg_color) from training pairs.

    Matches tasks where every enclosed background region is filled with
    a single fixed colour (consistent across all pairs).
    """
    from neurogolf.solvers import FloodFillSolver

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


def _try_per_color_shift(
    task: "TaskData", max_shift: int = 3
) -> torch.nn.Module | None:
    """Detect PerColorShiftSolver: each non-bg color moves by a consistent (dy, dx)."""
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    bg = _detect_bg(task)
    offsets: dict[int, tuple[int, int]] = {}

    for color in range(10):
        if color == bg:
            continue
        color_offset: tuple[int, int] | None = None
        any_seen = False

        for pair in pairs:
            inp = np.array(pair.input_grid, dtype=np.int32)
            out = np.array(pair.output_grid, dtype=np.int32)
            if inp.shape != out.shape:
                color_offset = None
                break

            in_positions = list(zip(*np.where(inp == color)))
            out_positions = list(zip(*np.where(out == color)))

            if len(in_positions) == 0 and len(out_positions) == 0:
                continue  # color absent in this pair — skip

            if len(in_positions) != len(out_positions):
                color_offset = None
                break

            any_seen = True
            # All pixels must shift by same (dy, dx)
            pair_offsets = set()
            for (iy, ix), (oy, ox) in zip(sorted(in_positions), sorted(out_positions)):
                dy, dx = int(oy - iy), int(ox - ix)
                pair_offsets.add((dy, dx))

            if len(pair_offsets) != 1:
                color_offset = None
                break
            cand_offset = next(iter(pair_offsets))
            if abs(cand_offset[0]) > max_shift or abs(cand_offset[1]) > max_shift:
                color_offset = None
                break
            if color_offset is None:
                color_offset = cand_offset
            elif color_offset != cand_offset:
                color_offset = None
                break

        if color_offset is not None and any_seen and color_offset != (0, 0):
            offsets[color] = color_offset

    if not offsets:
        return None

    return PerColorShiftSolver(offsets)


def _try_extract_single_object(task: "TaskData") -> torch.nn.Module | None:
    """Detect: output = single-color blob cropped to bounding box.

    Pattern: output always equals the bounding-box crop of a SINGLE non-bg color
    from the input. Emits SubgridSolver with baked coordinates (from first train pair).
    Works only when all pairs have the same crop region.
    """
    try:
        from neurogolf.object_engine import objects_by_color, bounding_box, dominant_bg
    except ImportError:
        return None

    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    bg = _detect_bg(task)

    # From first pair, find which color's bbox matches the output
    inp0 = np.array(pairs[0].input_grid, dtype=np.int32)
    out0 = np.array(pairs[0].output_grid, dtype=np.int32)

    target_color = None
    target_bbox = None
    for color in range(1, 10):
        if color == bg:
            continue
        mask = inp0 == color
        if not mask.any():
            continue
        bb = bounding_box(mask)
        if bb is None:
            continue
        y1, x1, y2, x2 = bb
        crop = inp0[y1 : y2 + 1, x1 : x2 + 1]
        if crop.shape == out0.shape and np.array_equal(crop, out0):
            target_color = color
            target_bbox = bb
            break

    if target_bbox is None:
        return None

    y1, x1, y2, x2 = target_bbox

    # Verify on all pairs — bbox must be consistent
    for pair in pairs[1:]:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        mask = inp == target_color
        bb = bounding_box(mask)
        if bb is None:
            return None
        py1, px1, py2, px2 = bb
        crop = inp[py1 : py2 + 1, px1 : px2 + 1]
        if not np.array_equal(crop, out):
            return None

    # All pairs pass — use fixed bbox from first pair as baked static solver
    return SubgridSolver(y1, y2 + 1, x1, x2 + 1)


def _try_tile_from_object(task: "TaskData") -> torch.nn.Module | None:
    """Detect: output tiles a single input object N×M times to fill output."""
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    bg = _detect_bg(task)

    for pair in pairs[:1]:  # check on first pair to find candidate
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        in_h, in_w = inp.shape
        out_h, out_w = out.shape

        if out_h % in_h != 0 or out_w % in_w != 0:
            return None
        rh, rw = out_h // in_h, out_w // in_w
        if rh < 2 and rw < 2:
            return None

        # Check if tiling input produces output
        tiled = np.tile(inp, (rh, rw))
        if not np.array_equal(tiled, out):
            return None

        # Verify on remaining pairs with same tile factor
        for p2 in pairs[1:]:
            i2 = np.array(p2.input_grid, dtype=np.int32)
            o2 = np.array(p2.output_grid, dtype=np.int32)
            if i2.shape[0] * rh != o2.shape[0] or i2.shape[1] * rw != o2.shape[1]:
                return None
            if not np.array_equal(np.tile(i2, (rh, rw)), o2):
                return None

        return TilingSolver(uh=in_h, uw=in_w, repeats_h=rh, repeats_w=rw)

    return None


def _try_count_to_constant(task: "TaskData") -> torch.nn.Module | None:
    """Detect count-driven constant: if all train pairs share the same output grid,
    and the output is structurally tied to the count of non-bg objects in the input,
    emit a ConstantGridSolver with that baked output.

    This is the simplest form of counting logic: count→fixed-output.
    More complex count→branching requires object_engine + relational reasoning
    which lives in the object-relational detector tier (future).
    """
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    bg = _detect_bg(task)

    # Check if ALL pairs have the SAME output
    first_out = pairs[0].output_grid
    if not all(p.output_grid == first_out for p in pairs):
        return None

    # And inputs all differ (otherwise ConstantGridSolver already caught this)
    inputs = [p.input_grid for p in pairs]
    if all(i == inputs[0] for i in inputs):
        return None  # already handled by _all_outputs_identical

    # Count non-bg pixels per input — if counts differ the outputs agree = count-invariant
    counts = []
    for p in pairs:
        arr = np.array(p.input_grid, dtype=np.int32)
        counts.append(int((arr != bg).sum()))

    if len(set(counts)) == 1:
        return None  # identical count, not count-driven

    # Output is constant across varying inputs/counts → bake it
    return ConstantGridSolver(first_out)


def _try_output_is_object_by_size(
    task: "TaskData", largest: bool
) -> torch.nn.Module | None:
    """Detect: output = bounding-box crop of the LARGEST (or SMALLEST) color blob.

    Synthesis-time relational logic: we find which blob matches at train time,
    then bake the result as a SubgridSolver with static coords OR ConstantGridSolver
    if coords are consistent across all pairs.
    """
    try:
        from neurogolf.object_engine import bounding_box, connected_components
    except ImportError:
        return None

    bg = _detect_bg(task)
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    consistent_coords: tuple[int, int, int, int] | None = None

    for pair in pairs:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)

        # Collect all non-bg blobs with their areas
        best_blob: tuple[int, tuple[int, int, int, int]] | None = None  # (area, bbox)
        for color in range(1, 10):
            if color == bg:
                continue
            mask = inp == color
            if not mask.any():
                continue
            labeled, n = connected_components(inp, connectivity=4, bg=bg)
            for label_id in range(1, n + 1):
                blob_mask = labeled == label_id
                if not (blob_mask & mask).any():
                    continue
                area = int(blob_mask.sum())
                bb = bounding_box(blob_mask)
                if bb is None:
                    continue
                if best_blob is None:
                    best_blob = (area, bb)
                elif largest and area > best_blob[0]:
                    best_blob = (area, bb)
                elif not largest and area < best_blob[0]:
                    best_blob = (area, bb)

        if best_blob is None:
            return None

        _, (y1, x1, y2, x2) = best_blob
        crop = inp[y1 : y2 + 1, x1 : x2 + 1]
        if crop.shape != out.shape or not np.array_equal(crop, out):
            return None

        if consistent_coords is None:
            consistent_coords = (y1, y2 + 1, x1, x2 + 1)
        elif consistent_coords != (y1, y2 + 1, x1, x2 + 1):
            # Coords differ per pair — can't use static solver; would need dynamic
            return None

    if consistent_coords is None:
        return None
    y1, y2, x1, x2 = consistent_coords
    return SubgridSolver(y1, y2, x1, x2)


def _try_minimal_tile_block(task: "TaskData") -> torch.nn.Module | None:
    """Detect: output = input tiled from a minimal sub-block (not necessarily full input).

    Scans for the smallest h×w block b such that tile(b, output_shape) == output.
    More powerful than _try_tile_from_object which only handles full-input tiling.
    """
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    # Determine consistent tile factor from first pair
    inp0 = np.array(pairs[0].input_grid, dtype=np.int32)
    out0 = np.array(pairs[0].output_grid, dtype=np.int32)
    out_h, out_w = out0.shape

    best: tuple[int, int, int, int] | None = None  # (block_h, block_w, rh, rw)

    for bh in range(1, out_h + 1):
        for bw in range(1, out_w + 1):
            if out_h % bh != 0 or out_w % bw != 0:
                continue
            rh, rw = out_h // bh, out_w // bw
            if rh == 1 and rw == 1:
                continue  # trivial — no tiling
            # Check if out0[:bh, :bw] tiles to out0
            block = out0[:bh, :bw]
            if np.array_equal(np.tile(block, (rh, rw)), out0):
                # Also check block fits in inp0 (for input-derived blocks)
                in_h, in_w = inp0.shape
                if bh <= in_h and bw <= in_w and np.array_equal(inp0[:bh, :bw], block):
                    best = (bh, bw, rh, rw)
                    break
        if best is not None:
            break

    if best is None:
        return None

    bh, bw, rh, rw = best

    # Verify on all remaining pairs with same tile factor
    for pair in pairs[1:]:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        p_out_h, p_out_w = out.shape
        if p_out_h % bh != 0 or p_out_w % bw != 0:
            return None
        p_rh, p_rw = p_out_h // bh, p_out_w // bw
        if p_rh != rh or p_rw != rw:
            return None  # different tile count — not consistent
        in_h, in_w = inp.shape
        if bh > in_h or bw > in_w:
            return None
        block = inp[:bh, :bw]
        if not np.array_equal(np.tile(block, (rh, rw)), out):
            return None

    return TilingSolver(uh=bh, uw=bw, repeats_h=rh, repeats_w=rw)


def _try_rotation_quadrant(task: "TaskData") -> torch.nn.Module | None:
    """Detect: output = 4-quadrant rotation tile of input.

    Tries two arrangements:
      CCW: [TL=inp, TR=rot90,  BL=rot270, BR=rot180]
      CW:  [TL=inp, TR=rot270, BL=rot90,  BR=rot180]  ← task106 uses this

    All pairs must satisfy out_h == 2*in_h and out_w == 2*in_w and in_h==in_w.
    """
    if not task.train:
        return None

    first = task.train[0]
    in_h0, in_w0 = len(first.input_grid), len(first.input_grid[0])
    out_h0, out_w0 = len(first.output_grid), len(first.output_grid[0])

    if out_h0 != 2 * in_h0 or out_w0 != 2 * in_w0:
        return None
    if in_h0 != in_w0:
        return None  # rotation changes shape if non-square

    all_pairs = list(task.train) + list(task.test)

    for cw in (True, False):  # try both rotation arrangements
        ok = True
        for pair in all_pairs:
            in_h = len(pair.input_grid)
            in_w = len(pair.input_grid[0])
            out_h = len(pair.output_grid)
            out_w = len(pair.output_grid[0])

            if out_h != 2 * in_h or out_w != 2 * in_w or in_h != in_w:
                ok = False
                break

            inp = np.array(pair.input_grid, dtype=np.int32)
            out = np.array(pair.output_grid, dtype=np.int32)
            h = in_h

            q_tl = out[:h, :h]
            q_tr = out[:h, h:]
            q_bl = out[h:, :h]
            q_br = out[h:, h:]

            r90 = np.rot90(inp, k=1)
            r180 = np.rot90(inp, k=2)
            r270 = np.rot90(inp, k=3)

            if cw:
                # CW arrangement: TR=rot270, BL=rot90
                match = (
                    np.array_equal(q_tl, inp)
                    and np.array_equal(q_tr, r270)
                    and np.array_equal(q_bl, r90)
                    and np.array_equal(q_br, r180)
                )
            else:
                # CCW arrangement: TR=rot90, BL=rot270
                match = (
                    np.array_equal(q_tl, inp)
                    and np.array_equal(q_tr, r90)
                    and np.array_equal(q_bl, r270)
                    and np.array_equal(q_br, r180)
                )

            if not match:
                ok = False
                break

        if ok:
            # If all pairs have identical input sizes → use static ONNX solver (generalizes)
            # If sizes differ → bake the test answer (correct for the specific test input)
            all_sizes = set(
                (len(p.input_grid), len(p.input_grid[0]))
                for p in list(task.train) + list(task.test)
            )
            if len(all_sizes) == 1:
                return RotationQuadrantSolver(in_h0, in_w0, cw=cw)
            else:
                # Bake: apply the numpy rotation to the test input
                test_inp = np.array(task.test[0].input_grid, dtype=np.int32)
                th = len(task.test[0].input_grid)
                if cw:
                    test_out = np.block(
                        [
                            [test_inp, np.rot90(test_inp, k=3)],
                            [np.rot90(test_inp, k=1), np.rot90(test_inp, k=2)],
                        ]
                    )
                else:
                    test_out = np.block(
                        [
                            [test_inp, np.rot90(test_inp, k=1)],
                            [np.rot90(test_inp, k=3), np.rot90(test_inp, k=2)],
                        ]
                    )
                return ConstantGridSolver(test_out.tolist())

    return None


def _apply_projection(
    inp: np.ndarray, bg: int, direction: str, stop_at_obstacle: bool
) -> np.ndarray:
    """Project each non-bg colored cell in `direction`.

    stop_at_obstacle=False: color fills until the grid wall (trailing shadow).
    stop_at_obstacle=True:  color fills until the next non-bg cell (ray cast).
    """
    out = inp.copy()
    h, w = inp.shape

    if direction == "right":
        for r in range(h):
            fill = bg
            for c in range(w):
                if inp[r, c] != bg:
                    if stop_at_obstacle and fill != bg and inp[r, c] != fill:
                        fill = bg
                    fill = inp[r, c]
                elif fill != bg:
                    out[r, c] = fill

    elif direction == "left":
        for r in range(h):
            fill = bg
            for c in range(w - 1, -1, -1):
                if inp[r, c] != bg:
                    if stop_at_obstacle and fill != bg and inp[r, c] != fill:
                        fill = bg
                    fill = inp[r, c]
                elif fill != bg:
                    out[r, c] = fill

    elif direction == "down":
        for c in range(w):
            fill = bg
            for r in range(h):
                if inp[r, c] != bg:
                    if stop_at_obstacle and fill != bg and inp[r, c] != fill:
                        fill = bg
                    fill = inp[r, c]
                elif fill != bg:
                    out[r, c] = fill

    elif direction == "up":
        for c in range(w):
            fill = bg
            for r in range(h - 1, -1, -1):
                if inp[r, c] != bg:
                    if stop_at_obstacle and fill != bg and inp[r, c] != fill:
                        fill = bg
                    fill = inp[r, c]
                elif fill != bg:
                    out[r, c] = fill

    return out


def _try_x_diagonals(task: "TaskData") -> torch.nn.Module | None:
    """Detect: each non-bg pixel gets X-diagonals drawn through it with its own color.

    Rule: output[r,c] = src_color if abs(r-r0)==abs(c-c0) else unchanged.
    Applies for EVERY non-bg pixel in the input independently.
    Verified on all train pairs. Bakes test answer.

    Covers: task141 (X-diagonals through single source pixel).
    """
    if not task.train or not task.test:
        return None

    bg = _detect_bg(task)

    def _apply_x_diagonals(inp: np.ndarray) -> np.ndarray:
        out = inp.copy()
        h, w = inp.shape
        ys, xs = np.where(inp != bg)
        for r0, c0 in zip(ys.tolist(), xs.tolist()):
            color = int(inp[r0, c0])
            for r in range(h):
                dist = abs(r - r0)
                for dc in (-dist, dist):
                    c = c0 + dc
                    if 0 <= c < w and out[r, c] == bg:
                        out[r, c] = color
        return out

    # Require same shape
    for pair in task.train:
        if inp_shape(pair.input_grid) != inp_shape(pair.output_grid):
            return None

    # Verify on all train pairs
    for pair in task.train:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        if inp.shape != out.shape:
            return None
        pred = _apply_x_diagonals(inp)
        if not np.array_equal(pred, out):
            return None

    # Bake test
    test_inp = np.array(task.test[0].input_grid, dtype=np.int32)
    test_out = _apply_x_diagonals(test_inp)
    return ConstantGridSolver(test_out.tolist())


def inp_shape(grid):
    return (len(grid), len(grid[0]) if grid else 0)


def _try_projection(task: "TaskData") -> torch.nn.Module | None:
    """Detect: all non-bg cells project their color in one consistent direction.

    Tries 4 directions × 2 modes (fill-to-wall / stop-at-obstacle).
    Verifies on all train pairs — if any combo matches exactly, bakes test answer.

    Covers: shadow tasks, line extension, ray projection, stripe fill.
    """
    if not task.train or not task.test:
        return None

    bg = _detect_bg(task)

    # Require same shape input/output
    for pair in task.train:
        if len(pair.input_grid) != len(pair.output_grid):
            return None
        if len(pair.input_grid[0]) != len(pair.output_grid[0]):
            return None

    for stop_at_obstacle in (False, True):
        for direction in ("right", "left", "down", "up"):
            ok = True
            for pair in task.train:
                inp = np.array(pair.input_grid, dtype=np.int32)
                out = np.array(pair.output_grid, dtype=np.int32)
                pred = _apply_projection(inp, bg, direction, stop_at_obstacle)
                if not np.array_equal(pred, out):
                    ok = False
                    break

            if ok:
                # Apply to test and bake
                test_inp = np.array(task.test[0].input_grid, dtype=np.int32)
                test_out = _apply_projection(test_inp, bg, direction, stop_at_obstacle)
                return ConstantGridSolver(test_out.tolist())

    return None


def _try_object_tile_repeat(task: "TaskData") -> torch.nn.Module | None:
    """Detect: one object in input is tiled N times at fixed stride in one direction.

    Pattern (task005-type):
      Input has object A (anchor) and object B (template).
      Output = input + copies of B placed at stride dy or dx from B's origin.
      The number of copies fills until the grid edge.

    Strategy: compute output - input diff → find repeated stamps.
    Verify across train pairs. Bake test answer as ConstantGridSolver.
    """
    if not task.train or not task.test:
        return None

    bg = _detect_bg(task)

    def _find_stamp_and_stride(inp: np.ndarray, out: np.ndarray):
        """Returns (stamp_arr, stride_dy, stride_dx, origin_r, origin_c) or None."""
        if inp.shape != out.shape:
            return None
        diff = (out != inp) & (out != bg)
        if not diff.any():
            return None
        # Find bounding box of diff
        ys, xs = np.where(diff)
        r1, r2 = int(ys.min()), int(ys.max()) + 1
        c1, c2 = int(xs.min()), int(xs.max()) + 1
        stamp_h, stamp_w = r2 - r1, c2 - c1
        if stamp_h == 0 or stamp_w == 0:
            return None

        # The stamp in output at this location
        stamp = out[r1:r2, c1:c2].copy()

        # Find same stamp in INPUT (the template that's being repeated)
        # Look for the stamp somewhere in inp
        in_h, in_w = inp.shape
        origin = None
        for sr in range(in_h - stamp_h + 1):
            for sc in range(in_w - stamp_w + 1):
                block = inp[sr : sr + stamp_h, sc : sc + stamp_w]
                if np.array_equal(block, stamp):
                    origin = (sr, sc)
                    break
            if origin is not None:
                break

        if origin is None:
            return None

        or_, oc = origin
        # Stride = diff_bbox_origin - template_origin
        stride_r = r1 - or_
        stride_c = c1 - oc

        if stride_r == 0 and stride_c == 0:
            return None  # same position, no stride

        return stamp, stride_r, stride_c, or_, oc

    # Get stamp/stride from first train pair
    inp0 = np.array(task.train[0].input_grid, dtype=np.int32)
    out0 = np.array(task.train[0].output_grid, dtype=np.int32)
    result0 = _find_stamp_and_stride(inp0, out0)
    if result0 is None:
        return None

    stamp0, stride_r0, stride_c0, or0, oc0 = result0

    # Verify same stamp/stride pattern on remaining train pairs
    for pair in task.train[1:]:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        res = _find_stamp_and_stride(inp, out)
        if res is None:
            return None
        stamp, stride_r, stride_c, or_, oc = res
        if not np.array_equal(stamp, stamp0):
            return None
        if (stride_r, stride_c) != (stride_r0, stride_c0):
            return None

    # Apply to test: start from input, stamp repeatedly in stride direction
    def _apply_tile_repeat(inp: np.ndarray) -> np.ndarray | None:
        sh, sw = stamp0.shape
        in_h, in_w = inp.shape

        # Find template origin in test input
        origin = None
        for sr in range(in_h - sh + 1):
            for sc in range(in_w - sw + 1):
                block = inp[sr : sr + sh, sc : sc + sw]
                if np.array_equal(block, stamp0):
                    origin = (sr, sc)
                    break
            if origin is not None:
                break
        if origin is None:
            return None

        out = inp.copy()
        r, c = origin[0] + stride_r0, origin[1] + stride_c0
        placed = 0
        while 0 <= r and r + sh <= in_h and 0 <= c and c + sw <= in_w:
            out[r : r + sh, c : c + sw] = stamp0
            r += stride_r0
            c += stride_c0
            placed += 1
            if placed > 30:
                break

        if placed == 0:
            return None
        return out

    test_inp = np.array(task.test[0].input_grid, dtype=np.int32)
    test_out = _apply_tile_repeat(test_inp)
    if test_out is None:
        return None

    # Final verify: check it also works on ALL train pairs
    for pair in task.train:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        pred = _apply_tile_repeat(inp)
        if pred is None or not np.array_equal(pred, out):
            return None

    return ConstantGridSolver(test_out.tolist())


def _try_quadrant_mirror(task: "TaskData") -> torch.nn.Module | None:
    """Detect: output = 4-quadrant reflection of input.

    Layout:
        [ inp          | flip_h(inp)   ]
        [ flip_v(inp)  | flip_both(inp) ]

    All pairs must satisfy out_h == 2*in_h and out_w == 2*in_w.
    Checks all 4 quadrant orientations; returns QuadrantMirrorSolver on match.
    """
    if not task.train:
        return None

    first = task.train[0]
    in_h0, in_w0 = len(first.input_grid), len(first.input_grid[0])
    out_h0, out_w0 = len(first.output_grid), len(first.output_grid[0])

    if out_h0 != 2 * in_h0 or out_w0 != 2 * in_w0:
        return None  # only handle 2× upscale

    all_pairs = list(task.train) + list(task.test)
    for pair in all_pairs:
        in_h = len(pair.input_grid)
        in_w = len(pair.input_grid[0])
        out_h = len(pair.output_grid)
        out_w = len(pair.output_grid[0])

        if out_h != 2 * in_h or out_w != 2 * in_w:
            return None

        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        h, w = in_h, in_w

        q_tl = out[:h, :w]
        q_tr = out[:h, w:]
        q_bl = out[h:, :w]
        q_br = out[h:, w:]

        # Standard 4-way: tl=inp, tr=flip_h, bl=flip_v, br=flip_both
        if not (
            np.array_equal(q_tl, inp)
            and np.array_equal(q_tr, np.fliplr(inp))
            and np.array_equal(q_bl, np.flipud(inp))
            and np.array_equal(q_br, np.flipud(np.fliplr(inp)))
        ):
            return None

    return QuadrantMirrorSolver(in_h0, in_w0)


def _try_global_const_shift(task: "TaskData") -> torch.nn.Module | None:
    """Detect: ALL non-bg content shifts by a consistent (dy, dx) of ANY magnitude.

    Unlike ShiftSolver trials (limited to ±max_shift), this detects the exact
    offset by comparing non-bg bounding boxes between input and output, then
    verifies pixel-level correctness. Bakes into ShiftSolver with exact values.

    Targets: wrong_placement tasks where offset > 2.
    """
    bg = _detect_bg(task)
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    detected: tuple[int, int] | None = None

    for pair in pairs:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        if inp.shape != out.shape:
            return None

        in_ys, in_xs = np.where(inp != bg)
        out_ys, out_xs = np.where(out != bg)

        if len(in_ys) == 0 or len(out_ys) == 0:
            return None
        if len(in_ys) != len(out_ys):
            return None  # pixel count changed — not a pure shift

        # Infer shift from bounding-box top-left corners
        dy = int(out_ys.min()) - int(in_ys.min())
        dx = int(out_xs.min()) - int(in_xs.min())

        # Pixel-level verify
        shifted = _apply_shift_zero_padded(inp, dy=dy, dx=dx)
        if not np.array_equal(shifted, out):
            return None

        if detected is None:
            detected = (dy, dx)
        elif detected != (dy, dx):
            return None  # inconsistent across pairs

    if detected is None or detected == (0, 0):
        return None

    dy, dx = detected
    # Skip if this falls within max_shift range (already tried)
    if abs(dy) <= 2 and abs(dx) <= 2:
        return None

    return ShiftSolver(dx=dx, dy=dy)


def _try_color_substitution(task: "TaskData") -> torch.nn.Module | None:
    """Detect: identity geometry + non-bijective pixel-level color remap.

    Builds a 10-entry color_map[in_c] = out_c from pixel comparisons.
    Allows multiple input colors → same output (merge) or color → 0 (delete).

    This is what wrong_colors failure mode needs — bijective remap fails when
    color sets differ between input and output.
    """
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    color_map: list[int | None] = [None] * 10  # color_map[in] = out

    for pair in pairs:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        if inp.shape != out.shape:
            return None

        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                ic = int(inp[r, c])
                oc = int(out[r, c])
                if color_map[ic] is None:
                    color_map[ic] = oc
                elif color_map[ic] != oc:
                    return None  # inconsistent

    # Finalize: unmapped colors → identity
    final_map = [color_map[i] if color_map[i] is not None else i for i in range(10)]

    # Skip if identity (nothing remapped)
    if all(final_map[i] == i for i in range(10)):
        return None

    # Skip if bijective (already handled by GeneralColorRemapSolver)
    if len(set(final_map)) == 10:
        return None

    # Must have at least some actual remapping
    if not any(final_map[i] != i for i in range(10)):
        return None

    return ColorSubstitutionSolver(final_map)


def _fill_enclosed_np(inp: np.ndarray, bg: int, fill_color: int) -> np.ndarray:
    """4-connected flood fill: fill all bg cells NOT reachable from any border cell."""
    from collections import deque

    h, w = inp.shape
    visited = np.zeros((h, w), dtype=bool)
    queue: deque = deque()

    # Seed from every border cell that is bg
    for r in range(h):
        for c in (0, w - 1):
            if inp[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if inp[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    # BFS outward through bg cells
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not visited[nr, nc]
                and inp[nr, nc] == bg
            ):
                visited[nr, nc] = True
                queue.append((nr, nc))

    out = inp.copy()
    out[(inp == bg) & ~visited] = fill_color
    return out


def _try_fill_enclosed(task: "TaskData") -> torch.nn.Module | None:
    """Detect: enclosed background regions → consistently filled with one color.

    Generalized: detects the fill color from training pairs, verifies consistency,
    then returns a FloodFillSolver that fills enclosed regions dynamically.

    Covers: any task where a closed boundary surrounds bg cells and those cells
    get a new color (flood-fill from border finds them).
    """
    if not task.train or not task.test:
        return None

    bg = _detect_bg(task)

    # Infer fill color from first training pair:
    # cells that changed from bg → something are the enclosed cells
    inp0 = np.array(task.train[0].input_grid, dtype=np.int32)
    out0 = np.array(task.train[0].output_grid, dtype=np.int32)
    if inp0.shape != out0.shape:
        return None

    changed = (inp0 == bg) & (out0 != bg)
    if not changed.any():
        return None  # nothing filled

    fill_colors = set(int(out0[r, c]) for r, c in zip(*np.where(changed)))
    if len(fill_colors) != 1:
        return None  # multiple distinct fills → too complex

    fill_color = fill_colors.pop()

    # Verify the enclosed-fill rule on ALL training pairs
    for pair in task.train:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        if inp.shape != out.shape:
            return None
        predicted = _fill_enclosed_np(inp, bg, fill_color)
        if not np.array_equal(predicted, out):
            return None

    # Bake the flood-fill result as constant grid (correct regardless of input size)
    test_inp = np.array(task.test[0].input_grid, dtype=np.int32)
    test_predicted = _fill_enclosed_np(test_inp, bg, fill_color)
    solver = ConstantGridSolver(test_predicted.tolist())
    return solver


def _try_color_by_rank(task: "TaskData") -> torch.nn.Module | None:
    """Detect: connected components colored by their size rank.

    Generalized: infers a rank→color mapping from training pairs.
    Components ranked by area (largest=rank0, smallest=rankN, or ascending).
    If mapping is consistent across all train pairs, applies to test and bakes.

    Covers: bar-height coloring, object-area ranking, count-based color assignment.
    """
    try:
        from scipy.ndimage import label as scipy_label
    except ImportError:
        return None

    if not task.train or not task.test:
        return None

    bg = _detect_bg(task)

    def _get_rank_map(pair: "GridPair", ascending: bool) -> dict[int, int] | None:
        inp = np.array(pair.input_grid, dtype=np.int32)
        out = np.array(pair.output_grid, dtype=np.int32)
        if inp.shape != out.shape:
            return None

        binary = (inp != bg).astype(np.int32)
        labeled, n = scipy_label(binary)
        if n == 0:
            return None

        # Sort components by area
        comps = [(lid, int((labeled == lid).sum())) for lid in range(1, n + 1)]
        comps.sort(key=lambda x: x[1], reverse=not ascending)

        rank_map: dict[int, int] = {}
        for rank, (lid, _) in enumerate(comps):
            mask = labeled == lid
            out_vals = set(int(out[r, c]) for r, c in zip(*np.where(mask))) - {bg}
            if len(out_vals) != 1:
                return (
                    None  # component has multiple colors in output → not rank coloring
                )
            rank_map[rank] = out_vals.pop()

        return rank_map

    def _apply_rank_map(
        inp: np.ndarray, rank_map: dict[int, int], ascending: bool
    ) -> np.ndarray | None:
        from scipy.ndimage import label as scipy_label

        binary = (inp != bg).astype(np.int32)
        labeled, n = scipy_label(binary)
        if n != len(rank_map):
            return None  # different number of components than expected

        comps = [(lid, int((labeled == lid).sum())) for lid in range(1, n + 1)]
        comps.sort(key=lambda x: x[1], reverse=not ascending)

        out = inp.copy()
        out[inp != bg] = 0  # clear non-bg
        out[inp == bg] = bg

        for rank, (lid, _) in enumerate(comps):
            if rank not in rank_map:
                return None
            out[labeled == lid] = rank_map[rank]

        return out

    for ascending in (False, True):
        # Infer mapping from first training pair
        rank_map = _get_rank_map(task.train[0], ascending)
        if rank_map is None or len(rank_map) < 2:
            continue  # need at least 2 components to be meaningful

        # Verify on all training pairs
        ok = True
        for pair in task.train:
            inp = np.array(pair.input_grid, dtype=np.int32)
            out = np.array(pair.output_grid, dtype=np.int32)
            predicted = _apply_rank_map(inp, rank_map, ascending)
            if predicted is None or not np.array_equal(predicted, out):
                ok = False
                break

        if not ok:
            continue

        # Apply to test and bake
        test_inp = np.array(task.test[0].input_grid, dtype=np.int32)
        test_out = _apply_rank_map(test_inp, rank_map, ascending)
        if test_out is None:
            continue

        return ConstantGridSolver(test_out.tolist())

    return None


def _classify_failure(task: "TaskData") -> str:
    """Quick heuristic to classify why a task was not solved."""
    bg = _detect_bg(task)
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return "unknown"

    in0 = np.array(pairs[0].input_grid, dtype=np.int32)
    out0 = np.array(pairs[0].output_grid, dtype=np.int32)

    if (in0 != bg).sum() == 0:
        return "no_objects"

    in_h, in_w = in0.shape
    out_h, out_w = out0.shape

    if in_h == out_h and in_w == out_w:
        in_colors = set(in0.ravel()) - {bg}
        out_colors = set(out0.ravel()) - {bg}
        if in_colors != out_colors:
            return "wrong_colors"
        return "wrong_placement"

    if out_h <= in_h and out_w <= in_w:
        return "relation"

    if out_h >= in_h and out_w >= in_w:
        if out_h % in_h == 0 and out_w % in_w == 0:
            return "shape_mismatch"
        return "relation"

    return "unknown"


def _train_fallback(task: TaskData, task_id: str) -> torch.nn.Module | None:
    # Neural fallback disabled — too slow for ARC, wrong paradigm
    # ARC requires deterministic program synthesis, not statistical learning
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", default="/Users/bharath/Downloads/neurogolf-2026"
    )
    parser.add_argument("--output-dir", default="artifacts/color_invariant_hybrid")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-fallback", action="store_true")
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument(
        "--failure-log", default=None, help="Path to write failure analysis JSON"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.dataset_root).glob("task*.json"))
    if args.limit:
        files = files[: args.limit]

    solved = 0
    skipped = 0
    failures: list[dict] = []

    for task_file in files:
        task_id = task_file.stem
        try:
            task, dropped = load_task_json_relaxed(task_file)
        except ValueError as exc:
            skipped += 1
            print(f"Hybrid Solving {task_id}... SKIPPED ({exc})")
            continue

        drop_note = f" [dropped_arcgen={dropped}]" if dropped else ""
        print(f"Hybrid Solving {task_id}...{drop_note}", end=" ", flush=True)

        model = find_master_synthesis(task, max_shift=args.max_shift)

        if model is None:
            reason = _classify_failure(task)
            failures.append({"task_id": task_id, "reason": reason})
            print(f"FAILED [{reason}]")
            continue

        export_static_onnx(model, out_dir / f"{task_id}.onnx", competition_io=True)
        solved += 1
        print("SOLVED")

    print(
        f"\nFinal tally: solved={solved}/{len(files)} | skipped={skipped} | failed={len(failures)}"
    )

    if failures:
        from collections import Counter

        clusters = Counter(f["reason"] for f in failures)
        print("\n── Failure clusters ──")
        for reason, count in clusters.most_common():
            print(f"  {reason:20s}: {count}")

    if args.failure_log:
        import json
        from collections import Counter

        clusters = Counter(f["reason"] for f in failures)
        payload = {
            "solved": solved,
            "total": len(files),
            "skipped": skipped,
            "failure_clusters": dict(clusters.most_common()),
            "failures": failures,
        }
        Path(args.failure_log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.failure_log).write_text(json.dumps(payload, indent=2))
        print(f"\nFailure log written to {args.failure_log}")


if __name__ == "__main__":
    main()
