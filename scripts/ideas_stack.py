#!/usr/bin/env python3
"""Five-idea implementation stack for NeuroGolf candidate mining.

Implemented lanes:
1) Color Permutation Search (zero-neural cost)
2) Object Decomposition Pipeline (per-color object translation)
3) Core Pattern Library (hand-crafted DSL program search)
4) Gradient-Based Program Synthesis (differentiable primitive selector)
5) Cross-Task Transfer Learning (lane ordering + primitive priors from solved neighbors)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from neurogolf.constants import GRID_SIZE, STATE_CHANNELS, COLOR_CHANNELS, IDENTITY_CHANNELS
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neurogolf.export import export_static_onnx
from neurogolf.grid_codec import decode_tensor_to_grid, encode_grid_to_tensor, get_color_normalization_map, apply_color_map
from neurogolf.solvers import (
    CompositionalSolver,
    ConstantGridSolver,
    DilateSolver,
    ErodeSolver,
    FlipSolver,
    GeneralColorRemapSolver,
    IdentitySolver,
    KroneckerSolver,
    NearestNeighborScaleSolver,
    RotationSolver,
    ShiftSolver,
    TilingSolver,
    TransposeSolver,
    ProjectSolver,
    AdjacencyMaskSolver,
    BorderMaskSolver,
    MultiCriteriaObjectSelector,
    RankPermutationSolver,
    BorderAwareMaskSolver,
    CollisionProjectSolver,
    ConditionalCompositionSolver,
    CenterObjectSolver,
    RelationMatrixSolver,
    RoleAssigner,
    AnchorFactory,
    PerLineageShiftSolver,
)
from neurogolf.task_io import GridPair, TaskData
from search_solvers import check_solve, find_master_synthesis, load_task_json_relaxed


@dataclass(frozen=True)
class EvalSummary:
    exact: bool
    pixel_accuracy: float
    total_pairs: int
    exact_pairs: int


@dataclass(frozen=True)
class LaneAttempt:
    lane: str
    status: str
    pixel_accuracy: float
    exact_pairs: int
    total_pairs: int
    note: str | None = None


@dataclass
class TransferMemoryEntry:
    task_id: str
    lane: str
    primitive: str | None
    feature: np.ndarray


def _parse_ids(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    out: set[str] = set()
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        if token.startswith("task"):
            out.add(token)
        else:
            out.add(f"task{int(token):03d}")
    return out


def _task_number(task_id: str) -> int:
    return int(task_id.replace("task", ""))


def _iter_pairs(task: TaskData, include_arc_gen: bool, arcgen_limit: int | None) -> list[GridPair]:
    pairs: list[GridPair] = []
    pairs.extend(task.train)
    pairs.extend(task.test)
    if include_arc_gen:
        arc = list(task.arc_gen)
        if arcgen_limit is not None:
            arc = arc[:arcgen_limit]
        pairs.extend(arc)
    return pairs


def _evaluate_model(
    model: nn.Module,
    task: TaskData,
    *,
    include_arc_gen: bool,
    arcgen_limit: int | None,
) -> EvalSummary:
    total_pairs = 0
    exact_pairs = 0
    pixel_total = 0
    pixel_correct = 0

    model.eval()
    with torch.no_grad():
        for pair in _iter_pairs(task, include_arc_gen=include_arc_gen, arcgen_limit=arcgen_limit):
            total_pairs += 1
            expected = pair.output_grid

            inp = torch.from_numpy(encode_grid_to_tensor(pair.input_grid))
            pred_tensor = model(inp).detach().cpu().numpy()
            pred_grid = decode_tensor_to_grid(pred_tensor, len(expected), len(expected[0]))

            exp_arr = np.asarray(expected, dtype=np.int32)
            pred_arr = np.asarray(pred_grid, dtype=np.int32)

            pixel_correct += int(np.sum(exp_arr == pred_arr))
            pixel_total += int(exp_arr.size)
            if pred_grid == expected:
                exact_pairs += 1

    pixel_accuracy = float(pixel_correct) / float(pixel_total) if pixel_total > 0 else 0.0
    return EvalSummary(
        exact=(total_pairs > 0 and exact_pairs == total_pairs),
        pixel_accuracy=pixel_accuracy,
        total_pairs=total_pairs,
        exact_pairs=exact_pairs,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _all_task_ids(dataset_root: Path) -> list[str]:
    return [p.stem for p in sorted(dataset_root.glob("task*.json"))]


def _discover_solved_ids(solved_dirs: list[Path]) -> set[str]:
    solved: set[str] = set()
    for d in solved_dirs:
        if not d.exists():
            continue
        for p in d.glob("task*.onnx"):
            solved.add(p.stem)
    return solved


def _load_candidates_from_report(
    report_path: Path,
    statuses: set[str],
    min_acc: float,
    max_candidates: int | None,
) -> list[str]:
    payload = json.loads(report_path.read_text())

    rows = payload.get("rows")
    if not isinstance(rows, list):
        rows = payload.get("results")
    if not isinstance(rows, list):
        selected = payload.get("selected_task_ids")
        if isinstance(selected, list):
            ids = [str(x) for x in selected]
            return ids[:max_candidates] if max_candidates is not None else ids
        raise ValueError(f"Unsupported report format: {report_path}")

    chosen: list[str] = []
    for row in rows:
        task_id = row.get("task_id")
        status = row.get("status")
        if not isinstance(task_id, str) or not isinstance(status, str):
            continue
        if status not in statuses:
            continue

        acc = row.get("pixel_accuracy")
        if isinstance(acc, (int, float)) and float(acc) < min_acc:
            continue

        chosen.append(task_id)
        if max_candidates is not None and len(chosen) >= max_candidates:
            break

    return chosen


# -------------------------
# Idea 1: Color permutation
# -------------------------


def _apply_shift_np(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = arr.shape
    out = np.zeros((h, w), dtype=arr.dtype)

    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)

    dst_y1 = max(0, dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if src_y2 > src_y1 and src_x2 > src_x1:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return out


def _geom_transforms(max_shift: int) -> list[tuple[str, nn.Module, Callable[[list[list[int]]], list[list[int]]]]]:
    out: list[tuple[str, nn.Module, Callable[[list[list[int]]], list[list[int]]]]] = [
        ("identity", IdentitySolver(), lambda g: np.asarray(g, dtype=np.int32).tolist()),
        ("transpose", TransposeSolver(), lambda g: np.asarray(g, dtype=np.int32).T.tolist()),
        ("rot90", RotationSolver(k=1), lambda g: np.rot90(np.asarray(g, dtype=np.int32), 1).tolist()),
        ("rot180", RotationSolver(k=2), lambda g: np.rot90(np.asarray(g, dtype=np.int32), 2).tolist()),
        ("rot270", RotationSolver(k=3), lambda g: np.rot90(np.asarray(g, dtype=np.int32), 3).tolist()),
        ("flip_h", FlipSolver(horizontal=True), lambda g: np.fliplr(np.asarray(g, dtype=np.int32)).tolist()),
        ("flip_v", FlipSolver(horizontal=False), lambda g: np.flipud(np.asarray(g, dtype=np.int32)).tolist()),
    ]

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            name = f"shift_{dx}_{dy}"
            out.append(
                (
                    name,
                    ShiftSolver(dx=dx, dy=dy),
                    lambda g, dx=dx, dy=dy: _apply_shift_np(np.asarray(g, dtype=np.int32), dy=dy, dx=dx).tolist(),
                )
            )
    return out


def _accumulate_bijective_constraints(
    src_grid: list[list[int]],
    dst_grid: list[list[int]],
    in_to_out: dict[int, int],
    out_to_in: dict[int, int],
) -> bool:
    if len(src_grid) != len(dst_grid) or len(src_grid[0]) != len(dst_grid[0]):
        return False

    for r in range(len(src_grid)):
        for c in range(len(src_grid[0])):
            in_color = int(src_grid[r][c])
            out_color = int(dst_grid[r][c])

            mapped_out = in_to_out.get(in_color)
            if mapped_out is None:
                in_to_out[in_color] = out_color
            elif mapped_out != out_color:
                return False

            mapped_in = out_to_in.get(out_color)
            if mapped_in is None:
                out_to_in[out_color] = in_color
            elif mapped_in != in_color:
                return False

    return True


def _complete_permutation(in_to_out: dict[int, int], out_to_in: dict[int, int]) -> list[int] | None:
    mapping = [-1] * 10
    used_outputs: set[int] = set()

    for in_color, out_color in in_to_out.items():
        if not (0 <= in_color < 10 and 0 <= out_color < 10):
            return None
        if out_color in used_outputs:
            return None
        mapping[in_color] = out_color
        used_outputs.add(out_color)

    remaining_inputs = [i for i in range(10) if mapping[i] < 0]
    remaining_outputs = [o for o in range(10) if o not in used_outputs]

    # Prefer identity assignments first.
    assigned_outputs: set[int] = set()
    for i in remaining_inputs:
        if i in remaining_outputs and i not in assigned_outputs:
            mapping[i] = i
            assigned_outputs.add(i)

    free_outputs = [o for o in remaining_outputs if o not in assigned_outputs]
    free_inputs = [i for i in remaining_inputs if mapping[i] < 0]

    if len(free_inputs) != len(free_outputs):
        return None

    for i, o in zip(free_inputs, free_outputs):
        mapping[i] = o

    if any(v < 0 for v in mapping):
        return None

    if len(set(mapping)) != 10:
        return None

    return mapping


def _derive_color_perm_for_transform(
    pairs: list[GridPair],
    transform: Callable[[list[list[int]]], list[list[int]]],
) -> list[int] | None:
    in_to_out: dict[int, int] = {}
    out_to_in: dict[int, int] = {}

    for pair in pairs:
        transformed = transform(pair.input_grid)
        ok = _accumulate_bijective_constraints(transformed, pair.output_grid, in_to_out, out_to_in)
        if not ok:
            return None

    return _complete_permutation(in_to_out, out_to_in)


def _color_enum_pairs(task: TaskData, mode: str) -> list[GridPair]:
    if mode == "train_only":
        pairs = list(task.train)
    elif mode == "test_only":
        pairs = list(task.test)
    else:
        pairs = list(task.train) + list(task.test)
    if not pairs:
        pairs = list(task.train) + list(task.test)
    return pairs


def solve_color_permutation(
    task: TaskData,
    max_shift: int,
    mode: str,
) -> tuple[nn.Module | None, str | None]:
    pairs = _color_enum_pairs(task, mode=mode)
    if not pairs:
        return None, None

    for name, geom_module, transform in _geom_transforms(max_shift=max_shift):
        color_map = _derive_color_perm_for_transform(pairs, transform)
        if color_map is None:
            continue
        candidate = CompositionalSolver(geom_module, GeneralColorRemapSolver(color_map))
        if check_solve(candidate, task, include_arc_gen=False):
            return candidate, name

    return None, None


# ------------------------------
# Idea 2: Object decomposition
# ------------------------------


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.bool_)

    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)

    dst_y1 = max(0, dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if src_y2 > src_y1 and src_x2 > src_x1:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return out


def _derive_per_color_offsets(pairs: list[GridPair]) -> dict[int, tuple[int, int]] | None:
    offsets: dict[int, tuple[int, int]] = {}

    for pair in pairs:
        in_arr = np.asarray(pair.input_grid, dtype=np.int32)
        out_arr = np.asarray(pair.output_grid, dtype=np.int32)
        if in_arr.shape != out_arr.shape:
            return None

        for color in range(1, 10):
            in_mask = in_arr == color
            out_mask = out_arr == color

            in_count = int(np.sum(in_mask))
            out_count = int(np.sum(out_mask))

            if in_count == 0 and out_count == 0:
                continue
            if in_count == 0 or out_count == 0 or in_count != out_count:
                return None

            in_pos = np.argwhere(in_mask)
            out_pos = np.argwhere(out_mask)

            dy = int(out_pos[:, 0].min() - in_pos[:, 0].min())
            dx = int(out_pos[:, 1].min() - in_pos[:, 1].min())

            shifted = _shift_mask(in_mask, dy=dy, dx=dx)
            if not np.array_equal(shifted, out_mask):
                return None

            prev = offsets.get(color)
            if prev is None:
                offsets[color] = (dx, dy)
            elif prev != (dx, dy):
                return None

    return offsets if offsets else None


class PerColorShiftSolver(nn.Module):
    """Translate each color channel independently by learned integer offsets."""

    def __init__(self, offsets: dict[int, tuple[int, int]]) -> None:
        super().__init__()
        self.offsets: list[tuple[int, int]] = [offsets.get(c, (0, 0)) for c in range(10)]

    @staticmethod
    def _shift_channel(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        shifted = x

        if dy > 0:
            shifted = F.pad(shifted[:, :, : -dy, :], (0, 0, dy, 0))
        elif dy < 0:
            up = -dy
            shifted = F.pad(shifted[:, :, up:, :], (0, 0, 0, up))

        if dx > 0:
            shifted = F.pad(shifted[:, :, :, : -dx], (dx, 0, 0, 0))
        elif dx < 0:
            left = -dx
            shifted = F.pad(shifted[:, :, :, left:], (0, left, 0, 0))

        return shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels: list[torch.Tensor] = []
        for color in range(10):
            dx, dy = self.offsets[color]
            ch = x[:, color : color + 1, :, :]
            if dx != 0 or dy != 0:
                ch = self._shift_channel(ch, dx=dx, dy=dy)
            channels.append(ch)
        return torch.cat(channels, dim=1)


def solve_object_decomposition(task: TaskData) -> tuple[nn.Module | None, dict[str, object] | None]:
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None, None

    offsets = _derive_per_color_offsets(pairs)
    if offsets is None:
        return None, None

    candidate = PerColorShiftSolver(offsets)
    if check_solve(candidate, task, include_arc_gen=False):
        return candidate, {"offsets": {str(k): [v[0], v[1]] for k, v in offsets.items()}}
    return None, None


def solve_object_template_matching(
    task: TaskData,
    *,
    max_shift: int,
    max_programs: int,
) -> tuple[nn.Module | None, str | None]:
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None, None

    # 12 canonical template transforms (geometric/morphological proxies).
    templates: list[tuple[str, nn.Module]] = [
        ("template_fill", DilateSolver(kernel_size=3, iterations=2)),
        ("template_shrink", ErodeSolver(kernel_size=3, iterations=1)),
        ("template_open", CompositionalSolver(ErodeSolver(kernel_size=3, iterations=1), DilateSolver(kernel_size=3, iterations=1))),
        ("template_close", CompositionalSolver(DilateSolver(kernel_size=3, iterations=1), ErodeSolver(kernel_size=3, iterations=1))),
        ("template_rot90", RotationSolver(k=1)),
        ("template_rot180", RotationSolver(k=2)),
        ("template_rot270", RotationSolver(k=3)),
        ("template_flip_h", FlipSolver(horizontal=True)),
        ("template_flip_v", FlipSolver(horizontal=False)),
        ("template_transpose", TransposeSolver()),
        ("template_shift1", ShiftSolver(dx=1, dy=0)),
        ("template_shift2", ShiftSolver(dx=0, dy=1)),
    ]

    if max_shift >= 2:
        templates.extend(
            [
                ("template_shift_diag", ShiftSolver(dx=1, dy=1)),
                ("template_shift_neg", ShiftSolver(dx=-1, dy=0)),
            ]
        )

    tried = 0
    for name, module in templates:
        if tried >= max_programs:
            break
        tried += 1
        candidate, wrapped = _maybe_wrap_with_color_map(module, task, pairs)
        if candidate is not None:
            return candidate, f"{name}{'+color' if wrapped else ''}"

    return None, None


# -----------------------------
# Idea 3: Core DSL search lane
# -----------------------------


def _all_pairs_same_shape(task: TaskData) -> tuple[int, int, int, int] | None:
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None

    in_h = len(pairs[0].input_grid)
    in_w = len(pairs[0].input_grid[0])
    out_h = len(pairs[0].output_grid)
    out_w = len(pairs[0].output_grid[0])

    for pair in pairs[1:]:
        if len(pair.input_grid) != in_h or len(pair.input_grid[0]) != in_w:
            return None
        if len(pair.output_grid) != out_h or len(pair.output_grid[0]) != out_w:
            return None

    return in_h, in_w, out_h, out_w


def _build_primitive_library(task: TaskData, max_shift: int) -> list[tuple[str, nn.Module]]:
    lib: list[tuple[str, nn.Module]] = [
        ("identity", IdentitySolver()),
        ("transpose", TransposeSolver()),
        ("rot90", RotationSolver(k=1)),
        ("rot180", RotationSolver(k=2)),
        ("rot270", RotationSolver(k=3)),
        ("flip_h", FlipSolver(horizontal=True)),
        ("flip_v", FlipSolver(horizontal=False)),
        ("erode1", ErodeSolver(kernel_size=3, iterations=1)),
        ("dilate1", DilateSolver(kernel_size=3, iterations=1)),
        ("erode2", ErodeSolver(kernel_size=3, iterations=2)),
        ("dilate2", DilateSolver(kernel_size=3, iterations=2)),
        ("adj_mask", AdjacencyMaskSolver()),
        ("border_mask", BorderMaskSolver()),
        ("project_u", ProjectSolver("up")),
        ("largest_obj", MultiCriteriaObjectSelector("largest")),
        ("smallest_obj", MultiCriteriaObjectSelector("smallest")),
        ("topmost_obj", MultiCriteriaObjectSelector("topmost")),
        ("center_obj", CenterObjectSolver()),
        ("border_aware_mask", BorderAwareMaskSolver()),
        ("relation_matrix", RelationMatrixSolver()),
        ("role_assigner", RoleAssigner()),
        ("anchor_factor", AnchorFactory("object_center")),
        ("coll_proj_r", CollisionProjectSolver("right")),
        ("coll_proj_l", CollisionProjectSolver("left")),
        ("coll_proj_d", CollisionProjectSolver("down")),
        ("coll_proj_u", CollisionProjectSolver("up")),
        ("rank_0_mask", MultiCriteriaObjectSelector("largest")),
        ("rank_1_mask", MultiCriteriaObjectSelector("smallest")), 
        ("rank_perm_identity", RankPermutationSolver(list(range(9)))),
        ("rank_perm_reverse", RankPermutationSolver(list(range(8, -1, -1)))),
        ("cond_keep_border", ConditionalCompositionSolver(BorderAwareMaskSolver(), IdentitySolver(), ShiftSolver(0, 0))),
        ("cond_move_nonborder", ConditionalCompositionSolver(BorderAwareMaskSolver(), IdentitySolver(), ShiftSolver(1, 1))),
        ("kronecker", KroneckerSolver()),
        ("const_bg", ConstantGridSolver([[0]*GRID_SIZE for _ in range(GRID_SIZE)])),
    ]
    
    # Partition: Only Spatial solvers (Grid-to-Grid) are used in the main DSL
    # Relational Reasoners are moved to a separate "expert" category
    lib_spatial = [
        (n, p) for n, p in lib 
        if n not in ("relation_matrix", "role_assigner", "anchor_factor")
    ]
    
    # Add frequent colors as constant grid candidates
    from collections import Counter
    all_out_pixels = [v for p in task.train for row in p.output_grid for v in row if v != 0]
    if all_out_pixels:
        most_common = Counter(all_out_pixels).most_common(2)
        for i, (color, _) in enumerate(most_common):
            lib.append((f"const_color_{color}", ConstantGridSolver([[color]*GRID_SIZE for _ in range(GRID_SIZE)])))

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            lib.append((f"shift_{dx}_{dy}", ShiftSolver(dx=dx, dy=dy)))

    shape = _all_pairs_same_shape(task)
    if shape is not None:
        in_h, in_w, out_h, out_w = shape
        if in_h > 0 and in_w > 0 and out_h % in_h == 0 and out_w % in_w == 0:
            sh = out_h // in_h
            sw = out_w // in_w
            if sh >= 1 and sw >= 1:
                lib.append((f"scale_{sh}x{sw}", NearestNeighborScaleSolver(in_h=in_h, in_w=in_w, scale_h=sh, scale_w=sw)))
                lib.append((f"tile_{sh}x{sw}", TilingSolver(uh=in_h, uw=in_w, repeats_h=sh, repeats_w=sw)))

    return lib_spatial


def _near_miss_object_refiner(
    base_model: nn.Module,
    task: TaskData,
    pairs: list[GridPair],
) -> tuple[nn.Module | None, str | None]:
    """Tries to fix small errors by shifting/recoloring individual object lineages."""
    # 1. Run model once to get prediction and IDs
    with torch.no_grad():
        results = []
        for pair in pairs:
            inp = torch.from_numpy(encode_grid_to_tensor(pair.input_grid))
            pred = base_model(inp)
            results.append((inp, pred, pair.output_grid))
            
    # Simple strategy: try local shifts for each active identity channel
    # K=16
    best_offsets = {}
    for i in range(IDENTITY_CHANNELS):
        # We greedily try to find the best shift for this ID
        best_dy, best_dx = 0, 0
        best_score = 0.0
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Try this shift in combination with existing bests for others? 
                # Greedy is faster.
                test_offsets = best_offsets.copy()
                test_offsets[i] = (dy, dx)
                
                refiner = PerLineageShiftSolver(test_offsets)
                scores = []
                for inp, pred, exp in results:
                    refined = refiner(pred)
                    pred_grid = decode_tensor_to_grid(refined.detach().cpu().numpy(), len(exp), len(exp[0]))
                    scores.append(_score_candidate(np.array(pred_grid), exp))
                
                avg = sum(scores) / len(scores)
                if avg > best_score:
                    best_score = avg
                    best_dy, best_dx = dy, dx
        
        if best_dy != 0 or best_dx != 0:
            best_offsets[i] = (best_dy, best_dx)
            
    if not best_offsets:
        return None, None
        
    final_refiner = PerLineageShiftSolver(best_offsets)
    final_model = CompositionalSolver(base_model, final_refiner)
    
    if check_solve(final_model, task, include_arc_gen=False):
        return final_model, f"object_refinement({len(best_offsets)} objs)"
    
    return None, None


def _maybe_wrap_with_color_map(
    base_model: nn.Module,
    task: TaskData,
    pairs: list[GridPair],
) -> tuple[nn.Module | None, str | None]:
    # 1. Check exact solve
    if check_solve(base_model, task, include_arc_gen=False):
        return base_model, None
        
    # 2. Try Object Refiner (Step 2 of Detective Brain)
    refined, r_note = _near_miss_object_refiner(base_model, task, pairs)
    if refined:
        return refined, r_note

    in_to_out: dict[int, int] = {}
    out_to_in: dict[int, int] = {}

    with torch.no_grad():
        for pair in pairs:
            expected = pair.output_grid
            inp = torch.from_numpy(encode_grid_to_tensor(pair.input_grid))
            pred_tensor = base_model(inp).detach().cpu().numpy()
            pred_grid = decode_tensor_to_grid(pred_tensor, len(expected), len(expected[0]))
            if not _accumulate_bijective_constraints(pred_grid, expected, in_to_out, out_to_in):
                return None, None

    cmap = _complete_permutation(in_to_out, out_to_in)
    if cmap is None:
        return None, None

    wrapped = CompositionalSolver(base_model, GeneralColorRemapSolver(cmap))
    if check_solve(wrapped, task, include_arc_gen=False):
        # Strict ONNX Size Guard
        try:
            import os
            from neurogolf.onnx_rules import check_file_size
            from neurogolf.constants import MAX_ONNX_FILE_BYTES
            
            tmp_size_chk = "/tmp/size_chk.onnx"
            export_static_onnx(wrapped, tmp_size_chk)
            size, ok = check_file_size(tmp_size_chk, MAX_ONNX_FILE_BYTES)
            if not ok:
                print(f"⚠️ Rejecting solution: Size {size} exceeds {MAX_ONNX_FILE_BYTES}")
                return None, None
        except Exception as e:
            print(f"⚠️ Size check failed: {e}")
            
        return wrapped, "color_map"
    return None, None


def _score_candidate(pred_arr: np.ndarray, expected: list[list[int]]) -> float:
    from scipy.ndimage import label
    from collections import Counter
    exp_arr = np.array(expected, dtype=np.int32)
    
    h, w = exp_arr.shape
    best_align = 0.0
    for dy in range(-h+1, h):
        for dx in range(-w+1, w):
            shifted = np.roll(pred_arr, dy, axis=0)
            shifted = np.roll(shifted, dx, axis=1)
            if dy > 0: shifted[:dy, :] = 0
            elif dy < 0: shifted[dy:, :] = 0
            if dx > 0: shifted[:, :dx] = 0
            elif dx < 0: shifted[:, dx:] = 0
            align = np.mean(exp_arr == shifted)
            if align > best_align:
                best_align = align

    pixel_score = np.mean(exp_arr == pred_arr)
    
    _, n_pred = label(pred_arr != 0)
    _, n_exp = label(exp_arr != 0)
    obj_score = 1.0 / (1.0 + abs(n_pred - n_exp))
    
    def get_bbox(arr):
        mask = arr != 0
        if not np.any(mask): return 0, 0
        r, c = np.where(mask)
        return r.max()-r.min()+1, c.max()-c.min()+1
    
    ph, pw = get_bbox(pred_arr)
    eh, ew = get_bbox(exp_arr)
    bbox_score = 1.0 / (1.0 + abs(ph - eh) + abs(pw - ew))
    
    pred_hist = Counter(pred_arr.flatten())
    exp_hist = Counter(exp_arr.flatten())
    pred_freqs = sorted(pred_hist.values(), reverse=True)
    exp_freqs = sorted(exp_hist.values(), reverse=True)
    max_len = max(len(pred_freqs), len(exp_freqs))
    pred_freqs += [0] * (max_len - len(pred_freqs))
    exp_freqs += [0] * (max_len - len(exp_freqs))
    hist_diff = sum(abs(p - e) for p, e in zip(pred_freqs, exp_freqs))
    hist_score = 1.0 / (1.0 + hist_diff / exp_arr.size)
    
    base_score = 0.6 * pixel_score + 0.2 * obj_score + 0.1 * bbox_score + 0.1 * hist_score
    return max(base_score, 0.5 * best_align + 0.5 * base_score)


def _parse_shift(n: str) -> tuple[int, int] | None:
    if not n.startswith("shift_"): return None
    parts = n.split("_")
    try: return int(parts[1]), int(parts[2])
    except: return None


def _chain_is_valid(names: list[str]) -> bool:
    categories = {
        "identity": "none",
        "transpose": "geo", "rot90": "geo", "rot180": "geo", "rot270": "geo",
        "flip_h": "geo", "flip_v": "geo",
        "erode1": "morph", "dilate1": "morph", "erode2": "morph", "dilate2": "morph",
        "adj_mask": "mask", "border_mask": "mask",
        "project_r": "proj", "project_l": "proj", "project_d": "proj", "project_u": "proj",
        "coll_proj_r": "proj", "coll_proj_l": "proj", "coll_proj_d": "proj", "coll_proj_u": "proj",
        "rank_0_mask": "mask", "rank_1_mask": "mask", "border_aware_mask": "mask",
        "rank_perm_identity": "map", "rank_perm_reverse": "map",
        "kronecker": "scale",
    }
    def get_cat(name):
        if name.startswith("shift"): return "shift"
        if name.startswith("scale"): return "scale"
        if name.startswith("tile"): return "scale"
        return categories.get(name, "unknown")
    
    cats = [get_cat(n) for n in names]
    for i in range(len(cats) - 2):
        if cats[i] == cats[i+1] == cats[i+2]: return False
        
    if "identity" in names[1:]: return False
    
    for i in range(len(names)-1):
        if names[i] == names[i+1]: return False
        if names[i] in ("project_r", "project_l") and names[i+1] in ("project_r", "project_l"): return False
        if names[i] in ("project_u", "project_d") and names[i+1] in ("project_u", "project_d"): return False
        sx_sy = _parse_shift(names[i])
        nx_ny = _parse_shift(names[i+1])
        if sx_sy is not None and nx_ny is not None:
            sx, sy = sx_sy
            nx, ny = nx_ny
            if sx+nx==0 and sy+ny==0: return False
    return True


def _translate_to_origin(grid: list[list[int]]) -> tuple:
    arr = np.array(grid, dtype=np.int32)
    mask = arr != 0
    if not np.any(mask): return tuple(tuple(row) for row in grid)
    r, c = np.where(mask)
    min_r, min_c = r.min(), c.min()
    shifted = np.zeros_like(arr)
    h, w = arr.shape
    shifted[0:h-min_r, 0:w-min_c] = arr[min_r:h, min_c:w]
    return tuple(tuple(row) for row in shifted.tolist())


def _cheap_eval(grids: list[list[list[int]]], expecteds: list[list[list[int]]]) -> float:
    """Fast proxy score using histograms and density."""
    scores = []
    for g, exp in zip(grids, expecteds):
        # 1. Histogram Match
        g_counts = np.bincount(np.array(g).flatten(), minlength=10)
        e_counts = np.bincount(np.array(exp).flatten(), minlength=10)
        hist_sim = 1.0 - np.sum(np.abs(g_counts - e_counts)) / (2.0 * np.array(exp).size)
        
        # 2. Density Match (Structure proxy)
        g_fg = (np.array(g) > 0).sum()
        e_fg = (np.array(exp) > 0).sum()
        dens_sim = 1.0 - abs(g_fg - e_fg) / (max(e_fg, 1))
        
        scores.append(0.5 * hist_sim + 0.5 * dens_sim)
    return sum(scores) / len(scores)

def _state_hash(pred_tensors: list[torch.Tensor], expecteds: list[list[list[int]]]) -> int:
    grids = []
    # Using decode_tensor_to_grid correctly
    for pt, exp in zip(pred_tensors, expecteds):
        grids.append(decode_tensor_to_grid(pt.detach().cpu().numpy(), len(exp), len(exp[0])))
    h_raw = hash(str(grids))
    cmap = get_color_normalization_map(grids)
    norm_grids = [apply_color_map(g, cmap) for g in grids]
    h_norm = hash(str(norm_grids))
    trans_grids = [_translate_to_origin(g) for g in grids]
    h_trans = hash(str(trans_grids))
    return hash((h_raw, h_norm, h_trans))


import random

def solve_dsl(
    task: TaskData,
    *,
    max_shift: int,
    depth: int,
    max_programs: int,
) -> tuple[nn.Module | None, str | None]:
    """Adaptive Beam Search compositional solver."""
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return None, None

    primitives = _build_primitive_library(task, max_shift=max_shift)
    
    # Boltzmann Search parameters
    T_start = 2.0
    T_min = 0.5
    
    # Depth-aware thresholds (Hardened v8)
    DEPTH_THRESHOLDS = {1: 0.01, 2: 0.03, 3: 0.05}
    
    # 1. State Persistence Check (K-channels)
    inps = [torch.from_numpy(encode_grid_to_tensor(p.input_grid)) for p in pairs]
    expecteds = [p.output_grid for p in pairs]

    # Heartbeat state
    start_time = time.time()
    last_pulse = start_time
    total_evals = 0

    seen_states = set()
    
    # beam: list of (score, names_list, CompositionalSolver)
    beam: list[tuple[float, list[str], nn.Module]] = []
    
    # ── Depth 1 Expansion Phase
    candidates = []
    for name, prim in primitives:
        if not _chain_is_valid([name]):
            continue
            
        try:
            scores = []
            pts = []
            with torch.no_grad():
                for inp, exp in zip(inps, expecteds):
                    out_t = prim(inp)
                    pts.append(out_t)
                    pred_grid = decode_tensor_to_grid(out_t.detach().cpu().numpy(), len(exp), len(exp[0]))
                    scores.append(_score_candidate(np.array(pred_grid), exp))
                    
            avg_score = sum(scores) / len(scores)
            
            # --- Detective Scoring (Hardened Batman v8) ---
            # 1. Structure vs Color (Depth-Aware Weighting)
            # Use cheap eval for structure signal at Depth 1
            struct_score = _cheap_eval(grids, expecteds)
            final_sim_score = 0.7 * struct_score + 0.3 * avg_score
            
            # 2. Identity Sparsity Bonus
            with torch.no_grad():
                ids = pts[0][:, 10:, :, :]
                avg_bits = ids.sum(dim=1).mean()
                sparsity_penalty = max(0, (avg_bits.item() - 4.0) * 0.05)
                
            final_score = final_sim_score - sparsity_penalty

            if final_score < DEPTH_THRESHOLDS[1]: 
                continue
            sh = _state_hash(pts, expecteds)
            
            if sh not in seen_states:
                seen_states.add(sh)
                candidates.append((final_score, [name], prim))
                
                # Full validation only if branch has high potential (structure mostly correct)
                if avg_score > 0.4:
                    c, wrapped = _maybe_wrap_with_color_map(prim, task, pairs)
                    if c:
                        return c, f"{name}{'+color' if wrapped else ''}"
        except Exception:
            continue
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    beam = candidates[:150]  # K=150
    
    # ── Depth 2..N Expansion Phase
    for current_depth in range(2, depth + 1):
        if not beam:
            break
            
        K_branch = 80 if current_depth == 2 else 40
        next_candidates = []
        
        for base_score, names, base_prim in beam:
            total_evals += 1
            
            # Status Pulse (every 5 seconds)
            curr_time = time.time()
            if curr_time - last_pulse > 5.0:
                print(f"💓 [Heartbeat] Depth {current_depth}, Evals: {total_evals}, Best Score: {beam[0][0] if beam else 0.0:.3f}")
                last_pulse = curr_time
            
            # Global Timeout (60s)
            if curr_time - start_time > 60.0:
                print("⏱️ Search timeout reached (60s). Returning best candidate.")
                if beam: return beam[0][2], "compositional_timeout"
                return None, None

            # Boltzmann Primitive Sampling
            # We decay temperature based on depth (Faster decay for hardening)
            temp = max(T_min, T_start * (0.7 ** (current_depth - 1)))
            
            # For simplicity, we sample from primitives with weight-based bias.
            # In ARC, 'Relational' primitives get higher weight if Object count > 2.
            sampled_primitives = primitives # In this local-search version, we still check all
            
            for p_name, p_mod in sampled_primitives:
                new_names = names + [p_name]
                if not _chain_is_valid(new_names):
                    continue

                try:
                    # Assuming base_prim is a CompositionalSolver or similar
                    # We need to track the intermediate tensors to avoid re-running the whole chain
                    # For simplicity in this structure, we re-run:
                    comp = CompositionalSolver(base_prim, p_mod)
                    scores = []
                    pts = []
                    with torch.no_grad():
                        for inp, exp in zip(inps, expecteds):
                            out_t = comp(inp)
                            # v9 State Sentry
                            if out_t.shape[1] == 0 or torch.isnan(out_t).any():
                                continue
                            pts.append(out_t)
                            pred_grid = decode_tensor_to_grid(out_t.detach().cpu().numpy(), len(exp), len(exp[0]))
                            scores.append(_score_candidate(np.array(pred_grid), exp))
                            
                    avg_score = sum(scores) / len(scores)
                    
                    # Strategic Pruning (Entropy/Identity Kill)
                    is_stupid = False
                    for pred_t in pts:
                        # 1. Pixel collapse (one color)
                        channels_active = (pred_t[:, :10].sum(dim=(2, 3)) > 0).float().sum()
                        if channels_active <= 1:
                            is_stupid = True; break
                        
                        # 2. Identity collapse (zero objects)
                        ids_active = (pred_t[:, 10:].sum(dim=(2, 3)) > 0.5).float().sum()
                        if ids_active == 0:
                            is_stupid = True; break
                    
                    if is_stupid: continue

                    # Hardened v8: Depth-aware scoring
                    pred_grids = []
                    for pt, exp in zip(pts, expecteds):
                        pred_grids.append(decode_tensor_to_grid(pt.detach().cpu().numpy(), len(exp), len(exp[0])))
                    
                    struct_score = _cheap_eval(pred_grids, expecteds)
                    final_sim_score = 0.5 * struct_score + 0.5 * avg_score
                    
                    with torch.no_grad():
                        ids = pts[0][:, 10:, :, :]
                        avg_bits = ids.sum(dim=1).mean()
                        sparsity_penalty = max(0, (avg_bits.item() - 4.0) * 0.05)
                    
                    final_score = final_sim_score - sparsity_penalty

                    if final_score < DEPTH_THRESHOLDS.get(current_depth, 0.05):
                        continue

                    sh = _state_hash(pts, expecteds)
                    next_candidates.append((final_score, new_names, comp))
                    if avg_score > 0.4:
                        c, wrapped = _maybe_wrap_with_color_map(comp, task, pairs)
                        if c:
                            return c, "->".join(new_names) + ("+color" if wrapped else "")
                except Exception:
                    continue
                    
        next_candidates.sort(key=lambda x: x[0], reverse=True)
        # Elite strategy: 80% top, 20% random exploration to avoid beam collapse
        elite_k = int(K_branch * 0.8)
        rand_k = K_branch - elite_k
        elite = next_candidates[:elite_k]
        others = next_candidates[elite_k:]
        if len(others) > rand_k:
            others = random.sample(others, rand_k)
        beam = elite + others

    return None, None




# ------------------------------------
# Idea 4: Gradient program synthesis
# ------------------------------------


class PrimitiveSelector(nn.Module):
    def __init__(self, primitives: list[nn.Module], init_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        self.primitives = nn.ModuleList(primitives)
        n = len(primitives)
        self.logits = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        if init_logits is not None and init_logits.numel() == n:
            with torch.no_grad():
                self.logits.copy_(init_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        out = None
        for idx, prim in enumerate(self.primitives):
            y = prim(x)
            w = weights[idx]
            out = y * w if out is None else out + y * w
        return out


def solve_gradient_program(
    task: TaskData,
    primitive_lib: list[tuple[str, nn.Module]],
    *,
    steps: int,
    lr: float,
    seed: int,
    init_logits: torch.Tensor | None,
) -> tuple[nn.Module | None, dict[str, object] | None]:
    pairs = list(task.train) + list(task.test)
    if not pairs or not primitive_lib:
        return None, None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    names = [name for name, _ in primitive_lib]
    modules = [mod for _, mod in primitive_lib]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for pair in pairs:
        xs.append(encode_grid_to_tensor(pair.input_grid)[0])
        ys.append(encode_grid_to_tensor(pair.output_grid)[0])

    x = torch.from_numpy(np.stack(xs)).float()
    y = torch.from_numpy(np.stack(ys)).float()

    selector = PrimitiveSelector(modules, init_logits=init_logits)
    optimizer = torch.optim.Adam([selector.logits], lr=lr)

    for _ in range(max(1, steps)):
        pred = selector(x)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.softmax(selector.logits, dim=0)
        top = torch.argsort(probs, descending=True)
        best_idx = int(top[0].item())
        second_idx = int(top[1].item()) if len(top) > 1 else best_idx
        margin = float(probs[best_idx] - probs[second_idx])

    best_name = names[best_idx]
    best_prim = modules[best_idx]

    wrapped, wrap_note = _maybe_wrap_with_color_map(best_prim, task, pairs)
    if wrapped is not None:
        return wrapped, {
            "selected_primitive": best_name,
            "selection_margin": round(margin, 6),
            "used_color_wrap": bool(wrap_note),
        }

    return None, {
        "selected_primitive": best_name,
        "selection_margin": round(margin, 6),
        "used_color_wrap": False,
    }


# ---------------------------------------
# Idea 5: Cross-task transfer heuristics
# ---------------------------------------


def _grid_symmetry_score(grid: list[list[int]]) -> float:
    arr = np.asarray(grid, dtype=np.int32)
    checks = [
        float(np.array_equal(arr, np.flipud(arr))),
        float(np.array_equal(arr, np.fliplr(arr))),
        float(np.array_equal(arr, np.rot90(arr, 2))),
    ]
    if arr.shape[0] == arr.shape[1]:
        checks.append(float(np.array_equal(arr, np.rot90(arr, 1))))
        checks.append(float(np.array_equal(arr, np.rot90(arr, 3))))
    return max(checks) if checks else 0.0


def _task_feature(task: TaskData) -> np.ndarray:
    pairs = list(task.train) + list(task.test)
    if not pairs:
        return np.zeros(10, dtype=np.float32)

    all_grids: list[list[list[int]]] = []
    for pair in pairs:
        all_grids.append(pair.input_grid)
        all_grids.append(pair.output_grid)

    max_colors = max(len({v for row in g for v in row}) for g in all_grids)
    mean_sym = float(np.mean([_grid_symmetry_score(g) for g in all_grids]))

    in_h = float(np.mean([len(p.input_grid) for p in pairs]))
    in_w = float(np.mean([len(p.input_grid[0]) for p in pairs]))
    out_h = float(np.mean([len(p.output_grid) for p in pairs]))
    out_w = float(np.mean([len(p.output_grid[0]) for p in pairs]))

    nz_ratio_in = float(
        np.mean(
            [
                np.mean(np.asarray(p.input_grid, dtype=np.int32) != 0)
                for p in pairs
            ]
        )
    )
    nz_ratio_out = float(
        np.mean(
            [
                np.mean(np.asarray(p.output_grid, dtype=np.int32) != 0)
                for p in pairs
            ]
        )
    )

    scale_h = out_h / max(1.0, in_h)
    scale_w = out_w / max(1.0, in_w)

    return np.asarray(
        [
            max_colors,
            mean_sym,
            in_h,
            in_w,
            out_h,
            out_w,
            nz_ratio_in,
            nz_ratio_out,
            scale_h,
            scale_w,
        ],
        dtype=np.float32,
    )


def _nearest_memory(
    memory: list[TransferMemoryEntry],
    feature: np.ndarray,
    k: int = 5,
) -> list[tuple[TransferMemoryEntry, float]]:
    if not memory:
        return []

    items: list[tuple[TransferMemoryEntry, float]] = []
    for entry in memory:
        dist = float(np.mean(np.abs(feature - entry.feature)))
        items.append((entry, dist))

    items.sort(key=lambda x: x[1])
    return items[:k]


def _transfer_lane_order(
    lanes: list[str],
    memory: list[TransferMemoryEntry],
    feature: np.ndarray,
) -> list[str]:
    neighbors = _nearest_memory(memory, feature)
    if not neighbors:
        return lanes[:]

    votes = {lane: 0.0 for lane in lanes}
    for entry, dist in neighbors:
        weight = 1.0 / (1e-3 + dist)
        votes[entry.lane] = votes.get(entry.lane, 0.0) + weight

    ordered = sorted(lanes, key=lambda lane: (-votes.get(lane, 0.0), lanes.index(lane)))
    return ordered


def _transfer_primitive_prior(
    primitive_names: list[str],
    memory: list[TransferMemoryEntry],
    feature: np.ndarray,
) -> torch.Tensor | None:
    neighbors = _nearest_memory(memory, feature)
    if not neighbors:
        return None

    scores = {name: 0.0 for name in primitive_names}
    for entry, dist in neighbors:
        if entry.primitive is None or entry.primitive not in scores:
            continue
        weight = 1.0 / (1e-3 + dist)
        scores[entry.primitive] += weight

    total = sum(scores.values())
    if total <= 0.0:
        return None

    prior = torch.tensor([scores[name] / total for name in primitive_names], dtype=torch.float32)
    return prior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Five-idea stack runner")
    parser.add_argument("--dataset-root", default="/Users/bharath/Downloads/neurogolf-2026")

    parser.add_argument("--input-report", default="artifacts/eval/sub2_neural_mine_120.json")
    parser.add_argument("--candidate-statuses", default="neural_soft_accept,failed")
    parser.add_argument("--candidate-ids", default=None)
    parser.add_argument("--min-acc", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=60)

    parser.add_argument(
        "--solved-dirs",
        nargs="+",
        default=["artifacts/push_4800_portfolio_v8", "artifacts/sub2_portfolio_v2"],
        help="Used only when candidate ids/report are unavailable",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-shift", type=int, default=5)
    parser.add_argument("--dsl-depth", type=int, default=2)
    parser.add_argument("--dsl-max-programs", type=int, default=450)

    parser.add_argument("--gradient-steps", type=int, default=180)
    parser.add_argument("--gradient-lr", type=float, default=0.2)

    parser.add_argument(
        "--color-enum-mode",
        default="train_test",
        choices=["train_test", "test_only", "train_only"],
        help="How to derive color-permutation constraints",
    )

    parser.add_argument(
        "--lanes",
        default="color_permutation,object_decomposition,dsl_library,gradient_synthesis,master_symbolic",
        help="Comma-separated lane subset to run",
    )

    parser.add_argument("--accept-pixel-acc", type=float, default=0.80)
    parser.add_argument("--include-arcgen-eval", action="store_true")
    parser.add_argument("--arcgen-eval-limit", type=int, default=50)

    parser.add_argument("--output-dir", default="artifacts/ideas_stack")
    parser.add_argument("--report", default="artifacts/eval/ideas_stack_report.json")
    parser.add_argument("--export-soft", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _select_candidates(args: argparse.Namespace, dataset_root: Path) -> list[str]:
    candidate_ids = _parse_ids(args.candidate_ids)
    if candidate_ids:
        ids = sorted(candidate_ids, key=_task_number)
        return ids[:args.limit] if args.limit else ids

    report_path = Path(args.input_report)
    statuses = {s.strip() for s in args.candidate_statuses.split(",") if s.strip()}
    if report_path.exists() and statuses:
        from_report = _load_candidates_from_report(
            report_path=report_path,
            statuses=statuses,
            min_acc=float(args.min_acc),
            max_candidates=args.max_candidates,
        )
        if from_report:
            return from_report

    all_ids = _all_task_ids(dataset_root)
    solved_ids = _discover_solved_ids([Path(p) for p in args.solved_dirs])
    remaining = sorted(set(all_ids) - solved_ids, key=_task_number)
    if args.max_candidates is not None:
        remaining = remaining[: args.max_candidates]
    return remaining[:args.limit] if args.limit else remaining


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.time()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    selected_ids = _select_candidates(args, dataset_root)
    if not selected_ids:
        raise SystemExit("No candidates selected.")

    output_dir = Path(args.output_dir)
    exact_dir = output_dir / "exact"
    soft_dir = output_dir / "soft"
    exact_dir.mkdir(parents=True, exist_ok=True)
    if args.export_soft:
        soft_dir.mkdir(parents=True, exist_ok=True)

    memory: list[TransferMemoryEntry] = []
    results: list[dict[str, object]] = []

    exact_count = 0
    soft_count = 0
    lane_exact_counts: dict[str, int] = {}

    default_lanes = [
        "color_permutation",
        "object_template_matching",
        "object_decomposition",
        "dsl_library",
        "gradient_synthesis",
        "master_symbolic",
    ]
    requested_lanes = [s.strip() for s in args.lanes.split(",") if s.strip()]
    if not requested_lanes:
        raise SystemExit("No lanes requested. Set --lanes to at least one lane.")
    unknown = [lane for lane in requested_lanes if lane not in default_lanes]
    if unknown:
        raise SystemExit(f"Unknown lanes: {', '.join(unknown)}")
    base_lanes = [lane for lane in default_lanes if lane in requested_lanes]

    print(f"Selected candidates: {len(selected_ids)}")

    for idx, task_id in enumerate(selected_ids, start=1):
        task_path = dataset_root / f"{task_id}.json"
        print(f"[{idx:03d}/{len(selected_ids):03d}] {task_id}", end=" ", flush=True)

        try:
            task, dropped = load_task_json_relaxed(task_path)
        except Exception as exc:
            print(f"SKIP(load:{exc})")
            results.append({"task_id": task_id, "status": "load_error", "error": str(exc)})
            continue

        feature = _task_feature(task)
        lane_order = _transfer_lane_order(base_lanes, memory, feature)

        task_row: dict[str, object] = {
            "task_id": task_id,
            "dropped_arcgen": dropped,
            "lane_order": lane_order,
            "lane_attempts": [],
        }

        best_soft_model: nn.Module | None = None
        best_soft_lane = ""
        best_soft_note: str | None = None
        best_soft_eval = EvalSummary(False, 0.0, 0, 0)
        solved_exact = False

        primitive_lib = _build_primitive_library(task, max_shift=args.max_shift)

        for lane in lane_order:
            model: nn.Module | None = None
            note: str | None = None
            primitive_name: str | None = None

            if lane == "color_permutation":
                model, op_name = solve_color_permutation(task, max_shift=args.max_shift, mode=args.color_enum_mode)
                note = op_name
                primitive_name = op_name

            elif lane == "object_template_matching":
                model, op_name = solve_object_template_matching(
                    task,
                    max_shift=args.max_shift,
                    max_programs=max(12, args.dsl_max_programs // 2),
                )
                note = op_name
                primitive_name = op_name

            elif lane == "object_decomposition":
                model, meta = solve_object_decomposition(task)
                note = None if meta is None else json.dumps(meta, separators=(",", ":"))

            elif lane == "dsl_library":
                model, prog_name = solve_dsl(
                    task,
                    max_shift=args.max_shift,
                    depth=args.dsl_depth,
                    max_programs=args.dsl_max_programs,
                )
                note = prog_name
                primitive_name = prog_name

            elif lane == "gradient_synthesis":
                prim_names = [name for name, _ in primitive_lib]
                prior = _transfer_primitive_prior(prim_names, memory, feature)
                model, meta = solve_gradient_program(
                    task,
                    primitive_lib,
                    steps=args.gradient_steps,
                    lr=args.gradient_lr,
                    seed=args.seed,
                    init_logits=prior,
                )
                if meta is not None:
                    note = json.dumps(meta, separators=(",", ":"))
                    primitive_name = str(meta.get("selected_primitive"))

            elif lane == "master_symbolic":
                model = find_master_synthesis(task, max_shift=args.max_shift)

            if model is None:
                task_row["lane_attempts"].append(
                    asdict(
                        LaneAttempt(
                            lane=lane,
                            status="no_model",
                            pixel_accuracy=0.0,
                            exact_pairs=0,
                            total_pairs=0,
                            note=note,
                        )
                    )
                )
                continue

            eval_info = _evaluate_model(
                model,
                task,
                include_arc_gen=bool(args.include_arcgen_eval),
                arcgen_limit=args.arcgen_eval_limit,
            )

            task_row["lane_attempts"].append(
                asdict(
                    LaneAttempt(
                        lane=lane,
                        status="exact" if eval_info.exact else "candidate",
                        pixel_accuracy=round(eval_info.pixel_accuracy, 6),
                        exact_pairs=eval_info.exact_pairs,
                        total_pairs=eval_info.total_pairs,
                        note=note,
                    )
                )
            )

            if eval_info.exact:
                task_row.update(
                    {
                        "status": "exact",
                        "winner_lane": lane,
                        "pixel_accuracy": round(eval_info.pixel_accuracy, 6),
                        "exact_pairs": eval_info.exact_pairs,
                        "total_pairs": eval_info.total_pairs,
                    }
                )

                if not args.dry_run:
                    out_path = exact_dir / f"{task_id}.onnx"
                    report = export_static_onnx(model, out_path)
                    if report is not None and report.is_valid:
                        task_row["onnx_path"] = str(out_path)
                        task_row["onnx_size_bytes"] = out_path.stat().st_size
                        exact_count += 1
                        lane_exact_counts[lane] = lane_exact_counts.get(lane, 0) + 1
                        solved_exact = True
                        memory.append(
                            TransferMemoryEntry(
                                task_id=task_id,
                                lane=lane,
                                primitive=primitive_name,
                                feature=feature,
                            )
                        )
                        print(f"EXACT({lane})")
                        break
                    task_row["status"] = "exact_export_invalid"
                    task_row["winner_lane"] = lane
                    print(f"EXACT_INVALID({lane})")
                    continue

                exact_count += 1
                lane_exact_counts[lane] = lane_exact_counts.get(lane, 0) + 1
                solved_exact = True
                memory.append(
                    TransferMemoryEntry(
                        task_id=task_id,
                        lane=lane,
                        primitive=primitive_name,
                        feature=feature,
                    )
                )
                print(f"EXACT({lane},dry)")
                break

            if eval_info.pixel_accuracy > best_soft_eval.pixel_accuracy:
                best_soft_model = model
                best_soft_lane = lane
                best_soft_note = note
                best_soft_eval = eval_info

        if solved_exact:
            results.append(task_row)
            continue

        if best_soft_model is not None and best_soft_eval.pixel_accuracy >= args.accept_pixel_acc:
            task_row.update(
                {
                    "status": "soft_accept",
                    "winner_lane": best_soft_lane,
                    "pixel_accuracy": round(best_soft_eval.pixel_accuracy, 6),
                    "exact_pairs": best_soft_eval.exact_pairs,
                    "total_pairs": best_soft_eval.total_pairs,
                    "note": best_soft_note,
                }
            )
            if args.export_soft and not args.dry_run:
                out_path = soft_dir / f"{task_id}.onnx"
                report = export_static_onnx(best_soft_model, out_path)
                if report is not None and report.is_valid:
                    task_row["onnx_path"] = str(out_path)
                    task_row["onnx_size_bytes"] = out_path.stat().st_size
                else:
                    task_row["status"] = "soft_export_invalid"
            soft_count += 1
            print(f"SOFT({best_soft_lane},{best_soft_eval.pixel_accuracy:.3f})")
        else:
            task_row.update(
                {
                    "status": "failed",
                    "pixel_accuracy": round(best_soft_eval.pixel_accuracy, 6),
                    "winner_lane": best_soft_lane if best_soft_lane else None,
                }
            )
            print(f"FAILED({best_soft_eval.pixel_accuracy:.3f})")

        results.append(task_row)

    summary = {
        "dataset_root": str(dataset_root),
        "selected_size": len(selected_ids),
        "exact_exported": exact_count,
        "soft_accepted": soft_count,
        "lane_exact_counts": lane_exact_counts,
        "elapsed_s": round(time.time() - started, 3),
        "dry_run": bool(args.dry_run),
    }

    report_payload = {
        "summary": summary,
        "selected_task_ids": selected_ids,
        "results": results,
    }

    report_path = Path(args.report)
    _write_json(report_path, report_payload)

    print("---")
    print(f"Report: {report_path}")
    print(f"Outcome: exact={exact_count}, soft={soft_count}")


if __name__ == "__main__":
    main()
