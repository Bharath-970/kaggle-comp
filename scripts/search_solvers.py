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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.constants import GRID_SIZE
from neurogolf.export import export_static_onnx
from neurogolf.grid_codec import decode_tensor_to_grid, encode_grid_to_tensor, get_color_normalization_map
from neurogolf.solvers import (
    ColorNormalizedSolver,
    CompositionalSolver,
    ConstantGridSolver,
    FlipSolver,
    GeneralColorRemapSolver,
    IdentitySolver,
    KroneckerSolver,
    NearestNeighborScaleSolver,
    RotationSolver,
    ShiftSolver,
    SubgridSolver,
    TilingSolver,
    TransposeSolver,
)
from neurogolf.task_io import GridPair, TaskData, load_task_json
from neurogolf.train import TrainConfig
from neurogolf.train_ensemble import train_ensemble_for_task


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


def _iter_all_pairs(task: TaskData) -> Iterable[GridPair]:
    for pair in task.train:
        yield pair
    for pair in task.test:
        yield pair
    for pair in task.arc_gen:
        yield pair


def _iter_search_pairs(task: TaskData) -> Iterable[GridPair]:
    # Cheap candidate generation: ARC tasks have tiny train/test, huge arc-gen.
    for pair in task.train:
        yield pair
    for pair in task.test:
        yield pair


def check_solve(model: torch.nn.Module, task: TaskData) -> bool:
    model.eval()
    with torch.no_grad():
        for pair in _iter_all_pairs(task):
            in_tensor = torch.from_numpy(encode_grid_to_tensor(pair.input_grid))
            pred_tensor = model(in_tensor).detach().cpu().numpy()
            expected = pair.output_grid
            pred = decode_tensor_to_grid(pred_tensor, len(expected), len(expected[0]))
            if pred != expected:
                return False
    return True


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
        for pair in _iter_all_pairs(task):
            expected = pair.output_grid
            in_tensor = torch.from_numpy(encode_grid_to_tensor(pair.input_grid))
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
            if check_solve(candidate, task):
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

    return None


def _train_fallback(task: TaskData, task_id: str) -> torch.nn.Module | None:
    backbone = train_ensemble_for_task(
        task=task,
        task_id=task_id,
        n_models=3,
        config=TrainConfig(
            epochs=400,
            learning_rate=4e-3,
            weight_decay=1e-5,
            arcgen_train_sample=32,
            batch_size=8,
            seed=42,
            use_augmentation=True,
        ),
        backbone_kwargs={
            "hidden_channels": 12,
            "steps": 5,
            "scratch_channels": 6,
            "mask_channels": 2,
            "phase_channels": 1,
            "use_coords": True,
            "use_depthwise": True,
        },
    )

    all_inputs = [pair.input_grid for pair in task.train]
    all_inputs.extend(pair.input_grid for pair in task.test)
    all_inputs.extend(pair.input_grid for pair in task.arc_gen)
    color_map = get_color_normalization_map(all_inputs)

    model = ColorNormalizedSolver(backbone=backbone, color_map=color_map)
    if check_solve(model, task):
        return model
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/Users/bharath/Downloads/neurogolf-2026")
    parser.add_argument("--output-dir", default="artifacts/color_invariant_hybrid")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-fallback", action="store_true")
    parser.add_argument("--max-shift", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.dataset_root).glob("task*.json"))
    if args.limit:
        files = files[:args.limit]

    solved = 0
    skipped = 0

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
        if model is None and args.train_fallback:
            print("Trying neural fallback...", end=" ", flush=True)
            model = _train_fallback(task, task_id)

        if model is None:
            print("FAILED")
            continue

        export_static_onnx(model, out_dir / f"{task_id}.onnx")
        solved += 1
        print("SOLVED")

    print(f"\nFinal tally: solved={solved}/{len(files)} | skipped={skipped}")


if __name__ == "__main__":
    main()
