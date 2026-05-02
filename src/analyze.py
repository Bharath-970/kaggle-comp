"""
Task pattern analyzer for NeuroGolf 2026.

Detects cheap symbolic ONNX-solvable task families. The most valuable patterns are
single-op transformations with no intermediate tensors:

- color permutation: Gather(axis=1, 10 params), score ~= 22.7
- row permutation:   Gather(axis=2, 30 params), score ~= 21.6
- col permutation:   Gather(axis=3, 30 params), score ~= 21.6

All checks use all available examples, including arc-gen examples, to reduce
private/public overfitting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

COLORS = 10
H = W = 30


def load_task(task_num: int, data_dir: str = "data") -> dict:
    path = Path(data_dir) / f"task{task_num:03d}.json"
    with open(path) as f:
        return json.load(f)


def grid_to_onehot(grid) -> np.ndarray:
    """Convert 2D color grid to [10, 30, 30] float32 one-hot/padded tensor."""
    oh = np.zeros((COLORS, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < COLORS and r < H and c < W:
                oh[color, r, c] = 1.0
    return oh


def onehot_to_grid(oh_arr: np.ndarray):
    grid = []
    for r in range(H):
        row = []
        for c in range(W):
            ch = int(np.argmax(oh_arr[:, r, c])) if oh_arr[:, r, c].sum() > 0 else 0
            row.append(ch)
        while row and row[-1] == 0:
            row.pop()
        if row:
            grid.append(row)
    while grid and not grid[-1]:
        grid.pop()
    return grid


def get_examples(task: dict, include_arcgen: bool = True):
    """Return list of (input_onehot [10,30,30], output_onehot [10,30,30]) pairs."""
    examples = []
    sets = task.get("train", []) + task.get("test", [])
    if include_arcgen:
        sets += task.get("arc-gen", [])
    for ex in sets:
        inp = ex.get("input", [])
        out = ex.get("output", [])
        if not inp or not out:
            continue
        if max(len(inp), max((len(r) for r in inp), default=0)) > 30:
            continue
        if max(len(out), max((len(r) for r in out), default=0)) > 30:
            continue
        examples.append((grid_to_onehot(inp), grid_to_onehot(out)))
    return examples


def apply_network_numpy(examples, transform_fn) -> bool:
    for i_oh, o_oh in examples:
        pred = transform_fn(i_oh)
        if not np.array_equal(pred, o_oh):
            return False
    return True


# ──────────────────────────────────────────────
# Basic symbolic pattern detectors
# ──────────────────────────────────────────────


def check_identity(examples) -> bool:
    return apply_network_numpy(examples, lambda x: x)


def check_hflip(examples) -> bool:
    return apply_network_numpy(examples, lambda x: x[:, :, ::-1].copy())


def check_vflip(examples) -> bool:
    return apply_network_numpy(examples, lambda x: x[:, ::-1, :].copy())


def check_rot180(examples) -> bool:
    return apply_network_numpy(examples, lambda x: x[:, ::-1, ::-1].copy())


def check_transpose_hw(examples) -> bool:
    return apply_network_numpy(examples, lambda x: x.transpose(0, 2, 1).copy())


def check_rot90_cw(examples) -> bool:
    def fn(x):
        t = x.transpose(0, 2, 1)
        return t[:, ::-1, :].copy()

    return apply_network_numpy(examples, fn)


def check_rot90_ccw(examples) -> bool:
    def fn(x):
        t = x.transpose(0, 2, 1)
        return t[:, :, ::-1].copy()

    return apply_network_numpy(examples, fn)


def check_rot270_cw(examples) -> bool:
    return check_rot90_ccw(examples)


def check_transp_v(examples) -> bool:
    """Anti-diagonal reflection: output[:, r, c] = input[:, 29-c, 29-r]."""

    def fn(x):
        t = x.transpose(0, 2, 1)
        return t[:, ::-1, ::-1].copy()

    return apply_network_numpy(examples, fn)


def detect_color_permutation(examples) -> Optional[list[int]]:
    """
    Check if output is a consistent color permutation of input.

    Returns Gather permutation `perm` where output channel oc = input channel perm[oc].
    """
    mapping = {}  # input color -> output color
    for i_oh, o_oh in examples:
        for r in range(H):
            for c in range(W):
                in_sum = i_oh[:, r, c].sum()
                out_sum = o_oh[:, r, c].sum()
                if in_sum <= 0 and out_sum <= 0:
                    continue
                if in_sum <= 0 or out_sum <= 0:
                    return None
                in_ch = int(np.argmax(i_oh[:, r, c]))
                out_ch = int(np.argmax(o_oh[:, r, c]))
                prev = mapping.get(in_ch)
                if prev is not None and prev != out_ch:
                    return None
                mapping[in_ch] = out_ch

    if not mapping:
        return None

    in_to_out = {c: mapping.get(c, c) for c in range(COLORS)}
    pred_perm = list(range(COLORS))
    for ic, oc in in_to_out.items():
        pred_perm[oc] = ic

    for i_oh, o_oh in examples:
        pred = i_oh[pred_perm, :, :]
        if not np.array_equal(pred, o_oh):
            return None
    return pred_perm


# ──────────────────────────────────────────────
# New: spatial Gather detectors
# ──────────────────────────────────────────────


def _row_is_empty(arr: np.ndarray, r: int) -> bool:
    return bool(arr[:, r, :].sum() == 0)


def _col_is_empty(arr: np.ndarray, c: int) -> bool:
    return bool(arr[:, :, c].sum() == 0)


def _find_matching_row(
    i_oh: np.ndarray, out_row: np.ndarray, preferred: int
) -> Optional[int]:
    if out_row.sum() == 0:
        if _row_is_empty(i_oh, preferred):
            return preferred
        empties = [r for r in range(H) if _row_is_empty(i_oh, r)]
        return empties[0] if empties else None
    for r in range(H):
        if np.array_equal(i_oh[:, r, :], out_row):
            return r
    return None


def _find_matching_col(
    i_oh: np.ndarray, out_col: np.ndarray, preferred: int
) -> Optional[int]:
    if out_col.sum() == 0:
        if _col_is_empty(i_oh, preferred):
            return preferred
        empties = [c for c in range(W) if _col_is_empty(i_oh, c)]
        return empties[0] if empties else None
    for c in range(W):
        if np.array_equal(i_oh[:, :, c], out_col):
            return c
    return None


def detect_row_permutation(examples) -> Optional[list[int]]:
    """Find row_perm where output[:, i, :] = input[:, row_perm[i], :]."""
    if not examples:
        return None

    intersections = [set(range(H)) for _ in range(H)]
    for i_oh, o_oh in examples:
        for i in range(H):
            out_row = o_oh[:, i, :]
            cands = {r for r in range(H) if np.array_equal(i_oh[:, r, :], out_row)}
            if not cands:
                return None
            intersections[i] &= cands
            if not intersections[i]:
                return None

    perm = []
    for i in range(H):
        cands = intersections[i]
        if i in cands:
            perm.append(i)
        else:
            perm.append(min(cands))

    # Verify globally.
    rp = np.array(perm, dtype=np.int64)
    for i_oh, o_oh in examples:
        pred = np.take(i_oh, rp, axis=1)
        if not np.array_equal(pred, o_oh):
            return None

    if perm == list(range(H)):
        return None
    return perm


def detect_col_permutation(examples) -> Optional[list[int]]:
    """Find col_perm where output[:, :, j] = input[:, :, col_perm[j]]."""
    if not examples:
        return None

    intersections = [set(range(W)) for _ in range(W)]
    for i_oh, o_oh in examples:
        for j in range(W):
            out_col = o_oh[:, :, j]
            cands = {c for c in range(W) if np.array_equal(i_oh[:, :, c], out_col)}
            if not cands:
                return None
            intersections[j] &= cands
            if not intersections[j]:
                return None

    perm = []
    for j in range(W):
        cands = intersections[j]
        if j in cands:
            perm.append(j)
        else:
            perm.append(min(cands))

    cp = np.array(perm, dtype=np.int64)
    for i_oh, o_oh in examples:
        pred = np.take(i_oh, cp, axis=2)
        if not np.array_equal(pred, o_oh):
            return None

    if perm == list(range(W)):
        return None
    return perm


def _canonical_seq(colors) -> tuple[int, ...]:
    """Canonicalize a color sequence, ignoring actual color names."""
    mapping = {}
    nxt = 0
    out = []
    for c in colors:
        c = int(c)
        if c not in mapping:
            mapping[c] = nxt
            nxt += 1
        out.append(mapping[c])
    return tuple(out)


def _row_canon(oh_arr: np.ndarray, r: int) -> tuple[int, ...]:
    row = oh_arr[:, r, :]
    # For padding row with no one-hot, use a special signature.
    if row.sum() == 0:
        return tuple([-1] * W)
    return _canonical_seq(np.argmax(row, axis=0))


def _col_canon(oh_arr: np.ndarray, c: int) -> tuple[int, ...]:
    col = oh_arr[:, :, c]
    if col.sum() == 0:
        return tuple([-1] * H)
    return _canonical_seq(np.argmax(col, axis=0))


def _greedy_perm_by_signatures(
    in_sigs, out_sigs, out_empty_flags
) -> Optional[list[int]]:
    perm = []
    used = set()
    for i, sig in enumerate(out_sigs):
        matches = [r for r, s in enumerate(in_sigs) if s == sig]
        if not matches:
            return None
        chosen = None
        if i in matches and out_empty_flags[i]:
            chosen = i
        else:
            for r in matches:
                if r not in used:
                    chosen = r
                    break
            if chosen is None:
                chosen = matches[0] if out_empty_flags[i] else None
        if chosen is None:
            return None
        perm.append(int(chosen))
        if not out_empty_flags[i]:
            used.add(chosen)
    return perm


def detect_row_col_permutation(examples) -> Optional[tuple[list[int], list[int]]]:
    """Find output = Gather(Gather(input, row_perm, axis=2), col_perm, axis=3)."""
    if not examples:
        return None

    first_i, first_o = examples[0]
    in_sigs = [_row_canon(first_i, r) for r in range(H)]
    out_sigs = [_row_canon(first_o, r) for r in range(H)]
    out_empty = [_row_is_empty(first_o, r) for r in range(H)]
    row_perm = _greedy_perm_by_signatures(in_sigs, out_sigs, out_empty)
    if row_perm is None:
        return None

    rp = np.array(row_perm, dtype=np.int64)
    row_applied = [(np.take(i_oh, rp, axis=1), o_oh) for i_oh, o_oh in examples]
    col_perm = detect_col_permutation(row_applied)
    if col_perm is None:
        return None

    cp = np.array(col_perm, dtype=np.int64)
    for i_oh, o_oh in examples:
        pred = np.take(np.take(i_oh, rp, axis=1), cp, axis=2)
        if not np.array_equal(pred, o_oh):
            return None

    if row_perm == list(range(H)) and col_perm == list(range(W)):
        return None
    return row_perm, col_perm


def detect_row_then_color(examples) -> Optional[tuple[list[int], list[int]]]:
    """Find output = color_perm(row_gather(input))."""
    if not examples:
        return None
    first_i, first_o = examples[0]
    in_sigs = [_row_canon(first_i, r) for r in range(H)]
    out_sigs = [_row_canon(first_o, r) for r in range(H)]
    out_empty = [_row_is_empty(first_o, r) for r in range(H)]
    row_perm = _greedy_perm_by_signatures(in_sigs, out_sigs, out_empty)
    if row_perm is None:
        return None
    rp = np.array(row_perm, dtype=np.int64)
    transformed = [(np.take(i_oh, rp, axis=1), o_oh) for i_oh, o_oh in examples]
    color_perm = detect_color_permutation(transformed)
    if color_perm is None or color_perm == list(range(COLORS)):
        return None
    for i_oh, o_oh in examples:
        pred = np.take(i_oh, rp, axis=1)[color_perm, :, :]
        if not np.array_equal(pred, o_oh):
            return None
    return row_perm, color_perm


def detect_col_then_color(examples) -> Optional[tuple[list[int], list[int]]]:
    """Find output = color_perm(col_gather(input))."""
    if not examples:
        return None
    first_i, first_o = examples[0]
    in_sigs = [_col_canon(first_i, c) for c in range(W)]
    out_sigs = [_col_canon(first_o, c) for c in range(W)]
    out_empty = [_col_is_empty(first_o, c) for c in range(W)]
    col_perm = _greedy_perm_by_signatures(in_sigs, out_sigs, out_empty)
    if col_perm is None:
        return None
    cp = np.array(col_perm, dtype=np.int64)
    transformed = [(np.take(i_oh, cp, axis=2), o_oh) for i_oh, o_oh in examples]
    color_perm = detect_color_permutation(transformed)
    if color_perm is None or color_perm == list(range(COLORS)):
        return None
    for i_oh, o_oh in examples:
        pred = np.take(i_oh, cp, axis=2)[color_perm, :, :]
        if not np.array_equal(pred, o_oh):
            return None
    return col_perm, color_perm


def detect_geo_color_perm(examples) -> Optional[tuple[str, list[int]]]:
    """Check basic geometry transform followed by non-trivial color permutation."""
    geos = {
        "hflip": lambda x: x[:, :, ::-1].copy(),
        "vflip": lambda x: x[:, ::-1, :].copy(),
        "rot180": lambda x: x[:, ::-1, ::-1].copy(),
        "transpose_hw": lambda x: x.transpose(0, 2, 1).copy(),
        "rot90_cw": lambda x: x.transpose(0, 2, 1)[:, ::-1, :].copy(),
        "rot90_ccw": lambda x: x.transpose(0, 2, 1)[:, :, ::-1].copy(),
        "transp_v": lambda x: x.transpose(0, 2, 1)[:, ::-1, ::-1].copy(),
    }
    ident = list(range(COLORS))
    for name, fn in geos.items():
        transformed = [(fn(i_oh), o_oh) for i_oh, o_oh in examples]
        perm = detect_color_permutation(transformed)
        if perm is not None and perm != ident:
            return name, perm
    return None


# ──────────────────────────────────────────────
# Existing tile/constant detectors
# ──────────────────────────────────────────────


def check_tile_2x2(examples) -> bool:
    return check_tile_nxm(examples, 2, 2)


def check_tile_3x3(examples) -> bool:
    return check_tile_nxm(examples, 3, 3)


def check_tile_nxm(examples, n: int, m: int) -> bool:
    """
    Current full-grid tile detector. Kept for compatibility with existing builders.
    Note: active-region tiling is better captured by row/col Gather detectors above.
    """
    for i_oh, o_oh in examples:
        tiled = np.tile(i_oh, (1, n, m))
        expected = np.zeros((COLORS, H, W), dtype=np.float32)
        rh, rw = min(tiled.shape[1], H), min(tiled.shape[2], W)
        expected[:, :rh, :rw] = tiled[:, :rh, :rw]
        if not np.array_equal(expected, o_oh):
            return False
    return True


def check_channel_any(examples) -> bool:
    return False


def check_constant_output(examples):
    """Return constant output tensor if output is same for all examples."""
    first_out = examples[0][1] if examples else None
    if first_out is None:
        return None
    for _, o_oh in examples[1:]:
        if not np.array_equal(o_oh, first_out):
            return None
    return first_out


# ──────────────────────────────────────────────
# Content-aware spatial transforms
# ──────────────────────────────────────────────

def _active_bounds(oh_arr):
    """Find the actual content bounds in a [C,H,W] one-hot array."""
    occupied = oh_arr.any(axis=0)  # [H, W]
    rows = np.where(occupied.any(axis=1))[0]
    cols = np.where(occupied.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, 0, 0
    return int(rows[0]), int(rows[-1]+1), int(cols[0]), int(cols[-1]+1)


def check_content_aware_hflip(examples) -> tuple | None:
    """
    Check if output = content-aware hflip.
    Content stays in same region [r0:r1, c0:c1], but values flip within region.
    Returns (r0, r1, c0, c1) if match, else None.
    """
    if not examples:
        return None

    # Check bounds are fixed across all examples
    r0, r1, c0, c1 = None, None, None, None
    for i_oh, o_oh in examples:
        bounds = _active_bounds(i_oh)
        if r0 is None:
            r0, r1, c0, c1 = bounds
        elif bounds != (r0, r1, c0, c1):
            return None  # Bounds not consistent

        # Verify output is zero outside content region
        o_check = o_oh.copy()
        o_check[:, r0:r1, c0:c1] = 0
        if o_check.any():
            return None

        # Extract content, flip horizontally, compare
        in_crop = i_oh[:, r0:r1, c0:c1]
        out_crop = o_oh[:, r0:r1, c0:c1]
        pred = in_crop[:, :, ::-1].copy()
        if not np.array_equal(pred, out_crop):
            return None

    return (r0, r1, c0, c1)


def check_content_aware_vflip(examples) -> tuple | None:
    """
    Check if output = content-aware vflip.
    Content stays in same region [r0:r1, c0:c1], but values flip vertically within region.
    Returns (r0, r1, c0, c1) if match, else None.
    """
    if not examples:
        return None

    r0, r1, c0, c1 = None, None, None, None
    for i_oh, o_oh in examples:
        bounds = _active_bounds(i_oh)
        if r0 is None:
            r0, r1, c0, c1 = bounds
        elif bounds != (r0, r1, c0, c1):
            return None

        o_check = o_oh.copy()
        o_check[:, r0:r1, c0:c1] = 0
        if o_check.any():
            return None

        in_crop = i_oh[:, r0:r1, c0:c1]
        out_crop = o_oh[:, r0:r1, c0:c1]
        pred = in_crop[:, ::-1, :].copy()
        if not np.array_equal(pred, out_crop):
            return None

    return (r0, r1, c0, c1)


def check_content_aware_rot180(examples) -> tuple | None:
    """
    Check if output = content-aware rot180 (hflip + vflip).
    Returns (r0, r1, c0, c1) if match, else None.
    """
    if not examples:
        return None

    r0, r1, c0, c1 = None, None, None, None
    for i_oh, o_oh in examples:
        bounds = _active_bounds(i_oh)
        if r0 is None:
            r0, r1, c0, c1 = bounds
        elif bounds != (r0, r1, c0, c1):
            return None

        o_check = o_oh.copy()
        o_check[:, r0:r1, c0:c1] = 0
        if o_check.any():
            return None

        in_crop = i_oh[:, r0:r1, c0:c1]
        out_crop = o_oh[:, r0:r1, c0:c1]
        pred = in_crop[:, ::-1, ::-1].copy()
        if not np.array_equal(pred, out_crop):
            return None

    return (r0, r1, c0, c1)


# ──────────────────────────────────────────────
# Main analyzer
# ──────────────────────────────────────────────


def analyze_task(task_num: int, data_dir: str = "data") -> dict:
    task = load_task(task_num, data_dir)
    examples = get_examples(task, include_arcgen=True)
    if not examples:
        return {"pattern": "unknown", "params": None}

    # Zero/single-op best cases.
    if check_identity(examples):
        return {"pattern": "identity", "params": None, "score_est": 25.0}
    if check_transpose_hw(examples):
        return {"pattern": "transpose_hw", "params": None, "score_est": 25.0}
    if check_hflip(examples):
        return {"pattern": "hflip", "params": None, "score_est": 23.6}
    if check_vflip(examples):
        return {"pattern": "vflip", "params": None, "score_est": 23.6}

    # Content-aware transforms (for variable-size content)
    ca_h = check_content_aware_hflip(examples)
    if ca_h is not None:
        return {"pattern": "content_aware_hflip", "params": ca_h, "score_est": 23.6}
    ca_v = check_content_aware_vflip(examples)
    if ca_v is not None:
        return {"pattern": "content_aware_vflip", "params": ca_v, "score_est": 23.6}
    ca_r = check_content_aware_rot180(examples)
    if ca_r is not None:
        return {"pattern": "content_aware_rot180", "params": ca_r, "score_est": 22.9}

    # Color Gather: cost=10.
    perm = detect_color_permutation(examples)
    if perm is not None:
        return {"pattern": "color_perm", "params": perm, "score_est": 22.7}

    # Spatial Gather: cost=30. This is the new high-impact family.
    row_perm = detect_row_permutation(examples)
    if row_perm is not None:
        return {"pattern": "row_perm", "params": row_perm, "score_est": 21.6}
    col_perm = detect_col_permutation(examples)
    if col_perm is not None:
        return {"pattern": "col_perm", "params": col_perm, "score_est": 21.6}

    # Multi-op symbolic families; still usually better than neural fallbacks.
    if check_transp_v(examples):
        return {"pattern": "transp_v", "params": None, "score_est": 14.5}
    if check_rot90_cw(examples):
        return {"pattern": "rot90_cw", "params": None, "score_est": 14.5}
    if check_rot90_ccw(examples):
        return {"pattern": "rot90_ccw", "params": None, "score_est": 14.5}
    if check_rot180(examples):
        return {"pattern": "rot180", "params": None, "score_est": 22.9}

    rc = detect_row_col_permutation(examples)
    if rc is not None:
        return {"pattern": "row_col_perm", "params": rc, "score_est": 14.5}
    rtc = detect_row_then_color(examples)
    if rtc is not None:
        return {"pattern": "row_then_color", "params": rtc, "score_est": 14.5}
    ctc = detect_col_then_color(examples)
    if ctc is not None:
        return {"pattern": "col_then_color", "params": ctc, "score_est": 14.5}
    geo_col = detect_geo_color_perm(examples)
    if geo_col is not None:
        return {"pattern": "geo_color_perm", "params": geo_col, "score_est": 14.5}

    # Existing tile patterns. Range now includes pure-axis cases.
    if check_tile_2x2(examples):
        return {"pattern": "tile_2x2", "params": None, "score_est": 13.0}
    if check_tile_3x3(examples):
        return {"pattern": "tile_3x3", "params": None, "score_est": 12.3}
    for n in range(1, 6):
        for m in range(1, 6):
            if (n, m) in {(1, 1), (2, 2), (3, 3)}:
                continue
            if check_tile_nxm(examples, n, m):
                return {"pattern": "tile_nxm", "params": (n, m), "score_est": 12.0}

    const = check_constant_output(examples)
    if const is not None:
        return {"pattern": "constant", "params": const, "score_est": 15.2}

    return {"pattern": "unknown", "params": None, "score_est": None}
