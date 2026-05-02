"""
Zero-MAC ONNX primitives for NeuroGolf 2026.

Cost formula: max(1, 25 - ln(MACs + memory + params))
Score 21 requires cost < 55. Score 25 requires cost = 0.

Memory = sum of bytes of intermediate tensors (NOT input/output).
Params = element count of initializers (onnx_tool counts float params).
Key: int64 index initializers are counted by onnx_tool but may be small.
"""

import math

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64
IR_VER = 10
OPSET = [oh.make_opsetid("", 10)]
GRID = [1, 10, 30, 30]  # [batch, channels, height, width]
INT64_MIN = np.iinfo(np.int64).min


def _make_model(nodes, initializers, value_infos=None):
    """Build a valid ONNX model with static shapes."""
    x = oh.make_tensor_value_info("input", FLOAT, GRID)
    y = oh.make_tensor_value_info("output", FLOAT, GRID)
    graph = oh.make_graph(nodes, "g", [x], [y], initializers)
    if value_infos:
        for vi in value_infos:
            graph.value_info.append(vi)
    m = oh.make_model(graph, ir_version=IR_VER, opset_imports=OPSET)
    onnx.checker.check_model(m)
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    return m


# ─────────────────────────────────────────────
# Score 25: 0 cost ops (no initializers, no intermediates)
# ─────────────────────────────────────────────


def identity():
    """Output = Input. Cost=0, Score=25."""
    node = oh.make_node("Identity", ["input"], ["output"])
    return _make_model([node], [])


def hflip():
    """Flip left-right (axis=3). Cost≈4 int64 params."""
    starts = onh.from_array(np.array([29], np.int64), "sl_s")
    ends = onh.from_array(np.array([INT64_MIN], np.int64), "sl_e")
    axes = onh.from_array(np.array([3], np.int64), "sl_ax")
    steps = onh.from_array(np.array([-1], np.int64), "sl_st")
    node = oh.make_node(
        "Slice", ["input", "sl_s", "sl_e", "sl_ax", "sl_st"], ["output"]
    )
    return _make_model([node], [starts, ends, axes, steps])


def vflip():
    """Flip top-bottom (axis=2). Cost≈4 int64 params."""
    starts = onh.from_array(np.array([29], np.int64), "sl_s")
    ends = onh.from_array(np.array([INT64_MIN], np.int64), "sl_e")
    axes = onh.from_array(np.array([2], np.int64), "sl_ax")
    steps = onh.from_array(np.array([-1], np.int64), "sl_st")
    node = oh.make_node(
        "Slice", ["input", "sl_s", "sl_e", "sl_ax", "sl_st"], ["output"]
    )
    return _make_model([node], [starts, ends, axes, steps])


def transpose_hw():
    """Swap H and W axes (transpose main diagonal). Cost=0."""
    # perm is an attribute, not initializer → 0 params
    node = oh.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])
    return _make_model([node], [])


def rot180():
    """Rotate 180° using a single multi-axis Slice. Cost≈8 params, no intermediate."""
    starts = onh.from_array(np.array([29, 29], np.int64), "r180_s")
    ends = onh.from_array(np.array([INT64_MIN, INT64_MIN], np.int64), "r180_e")
    axes = onh.from_array(np.array([2, 3], np.int64), "r180_a")
    steps = onh.from_array(np.array([-1, -1], np.int64), "r180_t")
    node = oh.make_node(
        "Slice", ["input", "r180_s", "r180_e", "r180_a", "r180_t"], ["output"]
    )
    return _make_model([node], [starts, ends, axes, steps])


def rot90_cw():
    """Rotate 90° clockwise = Transpose(HW) then VFlip."""
    # Transpose HW → tmp1
    n1 = oh.make_node("Transpose", ["input"], ["tmp1"], perm=[0, 1, 3, 2])
    # VFlip tmp1 → output
    sv = onh.from_array(np.array([29], np.int64), "sv_s")
    ev = onh.from_array(np.array([INT64_MIN], np.int64), "sv_e")
    av = onh.from_array(np.array([2], np.int64), "sv_a")
    tv = onh.from_array(np.array([-1], np.int64), "sv_t")
    n2 = oh.make_node("Slice", ["tmp1", "sv_s", "sv_e", "sv_a", "sv_t"], ["output"])
    vi = oh.make_tensor_value_info("tmp1", FLOAT, GRID)
    return _make_model([n1, n2], [sv, ev, av, tv], [vi])


def rot90_ccw():
    """Rotate 90° counter-clockwise = Transpose(HW) then HFlip."""
    n1 = oh.make_node("Transpose", ["input"], ["tmp1"], perm=[0, 1, 3, 2])
    sv = onh.from_array(np.array([29], np.int64), "sv_s")
    ev = onh.from_array(np.array([INT64_MIN], np.int64), "sv_e")
    av = onh.from_array(np.array([3], np.int64), "sv_a")
    tv = onh.from_array(np.array([-1], np.int64), "sv_t")
    n2 = oh.make_node("Slice", ["tmp1", "sv_s", "sv_e", "sv_a", "sv_t"], ["output"])
    vi = oh.make_tensor_value_info("tmp1", FLOAT, GRID)
    return _make_model([n1, n2], [sv, ev, av, tv], [vi])


def rot270_cw():
    """Same as rot90_ccw."""
    return rot90_ccw()


# ─────────────────────────────────────────────
# Content-aware spatial transforms
# ─────────────────────────────────────────────

def content_aware_hflip(r0, r1, c0, c1):
    """
    Crop to content region [r0:r1, c0:c1], apply hflip, pad back to [1,10,30,30].
    Cost ≈ 8 int64 (Slice) + 8 int64 (Pad) + transform.
    Score ≈ 23-24.
    """
    # Slice: input[0, :, r0:r1, c0:c1] → cropped [1, 10, r1-r0, c1-c0]
    slice_starts = onh.from_array(np.array([0, 0, r0, c0], np.int64), "s_start")
    slice_ends = onh.from_array(np.array([1, 10, r1, c1], np.int64), "s_end")
    n_slice = oh.make_node("Slice", ["input", "s_start", "s_end"], ["cropped"])

    # HFlip on axis 3 of cropped
    flip_start = onh.from_array(np.array([c1-c0-1], np.int64), "flip_s")
    flip_end = onh.from_array(np.array([INT64_MIN], np.int64), "flip_e")
    flip_axes = onh.from_array(np.array([3], np.int64), "flip_ax")
    flip_step = onh.from_array(np.array([-1], np.int64), "flip_st")
    n_flip = oh.make_node("Slice", ["cropped", "flip_s", "flip_e", "flip_ax", "flip_st"], ["flipped"])

    # Pad back: [1,10,r1-r0,c1-c0] → [1,10,30,30]
    pad_pads = onh.from_array(np.array([0,0,r0,c0,0,0,30-r1,30-c1], np.int64), "pad_v")
    n_pad = oh.make_node("Pad", ["flipped", "pad_v"], ["output"], mode="constant")

    vi_crop = oh.make_tensor_value_info("cropped", FLOAT, [1, 10, r1-r0, c1-c0])
    vi_flip = oh.make_tensor_value_info("flipped", FLOAT, [1, 10, r1-r0, c1-c0])
    return _make_model(
        [n_slice, n_flip, n_pad],
        [slice_starts, slice_ends, flip_start, flip_end, flip_axes, flip_step, pad_pads],
        [vi_crop, vi_flip]
    )


def content_aware_vflip(r0, r1, c0, c1):
    """
    Crop to content region [r0:r1, c0:c1], apply vflip, pad back to [1,10,30,30].
    """
    slice_starts = onh.from_array(np.array([0, 0, r0, c0], np.int64), "s_start")
    slice_ends = onh.from_array(np.array([1, 10, r1, c1], np.int64), "s_end")
    n_slice = oh.make_node("Slice", ["input", "s_start", "s_end"], ["cropped"])

    # VFlip on axis 2 of cropped
    flip_start = onh.from_array(np.array([r1-r0-1], np.int64), "flip_s")
    flip_end = onh.from_array(np.array([INT64_MIN], np.int64), "flip_e")
    flip_axes = onh.from_array(np.array([2], np.int64), "flip_ax")
    flip_step = onh.from_array(np.array([-1], np.int64), "flip_st")
    n_flip = oh.make_node("Slice", ["cropped", "flip_s", "flip_e", "flip_ax", "flip_st"], ["flipped"])

    # Pad back
    pad_pads = onh.from_array(np.array([0,0,r0,c0,0,0,30-r1,30-c1], np.int64), "pad_v")
    n_pad = oh.make_node("Pad", ["flipped", "pad_v"], ["output"], mode="constant")

    vi_crop = oh.make_tensor_value_info("cropped", FLOAT, [1, 10, r1-r0, c1-c0])
    vi_flip = oh.make_tensor_value_info("flipped", FLOAT, [1, 10, r1-r0, c1-c0])
    return _make_model(
        [n_slice, n_flip, n_pad],
        [slice_starts, slice_ends, flip_start, flip_end, flip_axes, flip_step, pad_pads],
        [vi_crop, vi_flip]
    )


def content_aware_rot180(r0, r1, c0, c1):
    """
    Crop to content region [r0:r1, c0:c1], apply rot180 (hflip + vflip), pad back.
    """
    slice_starts = onh.from_array(np.array([0, 0, r0, c0], np.int64), "s_start")
    slice_ends = onh.from_array(np.array([1, 10, r1, c1], np.int64), "s_end")
    n_slice = oh.make_node("Slice", ["input", "s_start", "s_end"], ["cropped"])

    # HFlip on axis 3
    flip_h_start = onh.from_array(np.array([c1-c0-1], np.int64), "fh_s")
    flip_h_end = onh.from_array(np.array([INT64_MIN], np.int64), "fh_e")
    flip_h_axes = onh.from_array(np.array([3], np.int64), "fh_ax")
    flip_h_step = onh.from_array(np.array([-1], np.int64), "fh_st")
    n_flip_h = oh.make_node("Slice", ["cropped", "fh_s", "fh_e", "fh_ax", "fh_st"], ["tmp_h"])

    # VFlip on axis 2
    flip_v_start = onh.from_array(np.array([r1-r0-1], np.int64), "fv_s")
    flip_v_end = onh.from_array(np.array([INT64_MIN], np.int64), "fv_e")
    flip_v_axes = onh.from_array(np.array([2], np.int64), "fv_ax")
    flip_v_step = onh.from_array(np.array([-1], np.int64), "fv_st")
    n_flip_v = oh.make_node("Slice", ["tmp_h", "fv_s", "fv_e", "fv_ax", "fv_st"], ["flipped"])

    # Pad back
    pad_pads = onh.from_array(np.array([0,0,r0,c0,0,0,30-r1,30-c1], np.int64), "pad_v")
    n_pad = oh.make_node("Pad", ["flipped", "pad_v"], ["output"], mode="constant")

    vi_crop = oh.make_tensor_value_info("cropped", FLOAT, [1, 10, r1-r0, c1-c0])
    vi_tmp_h = oh.make_tensor_value_info("tmp_h", FLOAT, [1, 10, r1-r0, c1-c0])
    vi_flip = oh.make_tensor_value_info("flipped", FLOAT, [1, 10, r1-r0, c1-c0])
    return _make_model(
        [n_slice, n_flip_h, n_flip_v, n_pad],
        [slice_starts, slice_ends, flip_h_start, flip_h_end, flip_h_axes, flip_h_step,
         flip_v_start, flip_v_end, flip_v_axes, flip_v_step, pad_pads],
        [vi_crop, vi_tmp_h, vi_flip]
    )


def color_permutation(perm):
    """
    Reorder color channels by permutation.
    perm: list of 10 ints — perm[i] = which input channel goes to output channel i.
    E.g. if input color 1 should become output color 2: perm[2] = 1.
    Uses Gather on axis=1. Cost = 10 int64 params.
    """
    assert len(perm) == 10
    idx = onh.from_array(np.array(perm, np.int64), "perm_idx")
    node = oh.make_node("Gather", ["input", "perm_idx"], ["output"], axis=1)
    return _make_model([node], [idx])


def row_gather(perm):
    """Arbitrary row permutation/repetition using Gather(axis=2). Cost=30 params."""
    assert len(perm) == 30
    idx = onh.from_array(np.array(perm, np.int64), "row_idx")
    node = oh.make_node("Gather", ["input", "row_idx"], ["output"], axis=2)
    return _make_model([node], [idx])


def col_gather(perm):
    """Arbitrary column permutation/repetition using Gather(axis=3). Cost=30 params."""
    assert len(perm) == 30
    idx = onh.from_array(np.array(perm, np.int64), "col_idx")
    node = oh.make_node("Gather", ["input", "col_idx"], ["output"], axis=3)
    return _make_model([node], [idx])


def row_then_color(row_perm, color_perm):
    """Row Gather then color Gather. One [1,10,30,30] intermediate."""
    assert len(row_perm) == 30 and len(color_perm) == 10
    row_idx = onh.from_array(np.array(row_perm, np.int64), "row_idx")
    col_idx = onh.from_array(np.array(color_perm, np.int64), "color_idx")
    n1 = oh.make_node("Gather", ["input", "row_idx"], ["tmp"], axis=2)
    n2 = oh.make_node("Gather", ["tmp", "color_idx"], ["output"], axis=1)
    vi = oh.make_tensor_value_info("tmp", FLOAT, GRID)
    return _make_model([n1, n2], [row_idx, col_idx], [vi])


def col_then_color(col_perm, color_perm):
    """Column Gather then color Gather. One [1,10,30,30] intermediate."""
    assert len(col_perm) == 30 and len(color_perm) == 10
    col_idx = onh.from_array(np.array(col_perm, np.int64), "col_idx")
    color_idx = onh.from_array(np.array(color_perm, np.int64), "color_idx")
    n1 = oh.make_node("Gather", ["input", "col_idx"], ["tmp"], axis=3)
    n2 = oh.make_node("Gather", ["tmp", "color_idx"], ["output"], axis=1)
    vi = oh.make_tensor_value_info("tmp", FLOAT, GRID)
    return _make_model([n1, n2], [col_idx, color_idx], [vi])


def row_col_gather(row_perm, col_perm):
    """Row Gather then column Gather. One [1,10,30,30] intermediate."""
    assert len(row_perm) == 30 and len(col_perm) == 30
    row_idx = onh.from_array(np.array(row_perm, np.int64), "row_idx")
    col_idx = onh.from_array(np.array(col_perm, np.int64), "col_idx")
    n1 = oh.make_node("Gather", ["input", "row_idx"], ["tmp"], axis=2)
    n2 = oh.make_node("Gather", ["tmp", "col_idx"], ["output"], axis=3)
    vi = oh.make_tensor_value_info("tmp", FLOAT, GRID)
    return _make_model([n1, n2], [row_idx, col_idx], [vi])


def transp_v():
    """Anti-diagonal reflection = Transpose(H,W) then reverse H and W."""
    n1 = oh.make_node("Transpose", ["input"], ["tmp1"], perm=[0, 1, 3, 2])
    starts = onh.from_array(np.array([29, 29], np.int64), "tv_s")
    ends = onh.from_array(np.array([INT64_MIN, INT64_MIN], np.int64), "tv_e")
    axes = onh.from_array(np.array([2, 3], np.int64), "tv_a")
    steps = onh.from_array(np.array([-1, -1], np.int64), "tv_t")
    n2 = oh.make_node("Slice", ["tmp1", "tv_s", "tv_e", "tv_a", "tv_t"], ["output"])
    vi = oh.make_tensor_value_info("tmp1", FLOAT, GRID)
    return _make_model([n1, n2], [starts, ends, axes, steps], [vi])


def _geo_nodes_to_output(geo_name, in_name, out_name, tmp_prefix="g"):
    """Return (nodes, initializers, value_infos) for a geo transform from in_name to out_name."""
    nodes, inits, vis = [], [], []
    if geo_name == "identity":
        nodes.append(oh.make_node("Identity", [in_name], [out_name]))
    elif geo_name == "transpose_hw":
        nodes.append(
            oh.make_node("Transpose", [in_name], [out_name], perm=[0, 1, 3, 2])
        )
    elif geo_name in {"hflip", "vflip"}:
        axis = 3 if geo_name == "hflip" else 2
        s = onh.from_array(np.array([29], np.int64), f"{tmp_prefix}_s")
        e = onh.from_array(np.array([INT64_MIN], np.int64), f"{tmp_prefix}_e")
        a = onh.from_array(np.array([axis], np.int64), f"{tmp_prefix}_a")
        t = onh.from_array(np.array([-1], np.int64), f"{tmp_prefix}_t")
        nodes.append(
            oh.make_node(
                "Slice",
                [
                    in_name,
                    f"{tmp_prefix}_s",
                    f"{tmp_prefix}_e",
                    f"{tmp_prefix}_a",
                    f"{tmp_prefix}_t",
                ],
                [out_name],
            )
        )
        inits.extend([s, e, a, t])
    elif geo_name == "rot180":
        s = onh.from_array(np.array([29, 29], np.int64), f"{tmp_prefix}_s")
        e = onh.from_array(
            np.array([INT64_MIN, INT64_MIN], np.int64), f"{tmp_prefix}_e"
        )
        a = onh.from_array(np.array([2, 3], np.int64), f"{tmp_prefix}_a")
        t = onh.from_array(np.array([-1, -1], np.int64), f"{tmp_prefix}_t")
        nodes.append(
            oh.make_node(
                "Slice",
                [
                    in_name,
                    f"{tmp_prefix}_s",
                    f"{tmp_prefix}_e",
                    f"{tmp_prefix}_a",
                    f"{tmp_prefix}_t",
                ],
                [out_name],
            )
        )
        inits.extend([s, e, a, t])
    elif geo_name in {"rot90_cw", "rot90_ccw", "transp_v"}:
        tmp = f"{tmp_prefix}_tmp"
        nodes.append(oh.make_node("Transpose", [in_name], [tmp], perm=[0, 1, 3, 2]))
        vis.append(oh.make_tensor_value_info(tmp, FLOAT, GRID))
        if geo_name == "rot90_cw":
            axes = [2]
            starts = [29]
            ends = [INT64_MIN]
            steps = [-1]
        elif geo_name == "rot90_ccw":
            axes = [3]
            starts = [29]
            ends = [INT64_MIN]
            steps = [-1]
        else:
            axes = [2, 3]
            starts = [29, 29]
            ends = [INT64_MIN, INT64_MIN]
            steps = [-1, -1]
        s = onh.from_array(np.array(starts, np.int64), f"{tmp_prefix}_s")
        e = onh.from_array(np.array(ends, np.int64), f"{tmp_prefix}_e")
        a = onh.from_array(np.array(axes, np.int64), f"{tmp_prefix}_a")
        t = onh.from_array(np.array(steps, np.int64), f"{tmp_prefix}_t")
        nodes.append(
            oh.make_node(
                "Slice",
                [
                    tmp,
                    f"{tmp_prefix}_s",
                    f"{tmp_prefix}_e",
                    f"{tmp_prefix}_a",
                    f"{tmp_prefix}_t",
                ],
                [out_name],
            )
        )
        inits.extend([s, e, a, t])
    else:
        raise ValueError(f"Unknown geo transform: {geo_name}")
    return nodes, inits, vis


def geo_then_color_perm(geo_name, color_perm):
    """Apply geometry transform, then color Gather."""
    assert len(color_perm) == 10
    geo_nodes, geo_inits, geo_vis = _geo_nodes_to_output(
        geo_name, "input", "geo_out", "geo"
    )
    geo_vis.append(oh.make_tensor_value_info("geo_out", FLOAT, GRID))
    idx = onh.from_array(np.array(color_perm, np.int64), "color_idx")
    gather = oh.make_node("Gather", ["geo_out", "color_idx"], ["output"], axis=1)
    return _make_model(geo_nodes + [gather], geo_inits + [idx], geo_vis)


def tile_2x(axis_h=True, axis_w=True):
    """
    Tile input 2× along H and/or W.
    Output still [1,10,30,30] via Slice after Tile.
    For grids that upscale 2×: input is ~15×15, output is ~30×30.
    """
    reps_data = [1, 1, 2 if axis_h else 1, 2 if axis_w else 1]
    reps = onh.from_array(np.array(reps_data, np.int64), "tile_reps")
    tiled_h = reps_data[2] * 30
    tiled_w = reps_data[3] * 30
    tiled_shape = [1, 10, tiled_h, tiled_w]
    n1 = oh.make_node("Tile", ["input", "tile_reps"], ["tiled"])
    # Slice back to [1,10,30,30]
    s_ = onh.from_array(np.array([0, 0, 0, 0], np.int64), "sl_s")
    e_ = onh.from_array(np.array([1, 10, 30, 30], np.int64), "sl_e")
    n2 = oh.make_node("Slice", ["tiled", "sl_s", "sl_e"], ["output"])
    vi = oh.make_tensor_value_info("tiled", FLOAT, tiled_shape)
    return _make_model([n1, n2], [reps, s_, e_], [vi])


def tile_2x2():
    """Compatibility wrapper: tile 2x in H and W."""
    return tile_2x(True, True)


def tile_3x():
    """Tile 3× H and W."""
    reps = onh.from_array(np.array([1, 1, 3, 3], np.int64), "tile_reps")
    s_ = onh.from_array(np.array([0, 0, 0, 0], np.int64), "sl_s")
    e_ = onh.from_array(np.array([1, 10, 30, 30], np.int64), "sl_e")
    n1 = oh.make_node("Tile", ["input", "tile_reps"], ["tiled"])
    n2 = oh.make_node("Slice", ["tiled", "sl_s", "sl_e"], ["output"])
    vi = oh.make_tensor_value_info("tiled", FLOAT, [1, 10, 90, 90])
    return _make_model([n1, n2], [reps, s_, e_], [vi])


def tile_3x3():
    """Compatibility wrapper: tile 3x in H and W."""
    return tile_3x()


def tile_hw(rh, rw):
    """Tile rh× and rw× (generic)."""
    reps = onh.from_array(np.array([1, 1, rh, rw], np.int64), "tile_reps")
    s_ = onh.from_array(np.array([0, 0, 0, 0], np.int64), "sl_s")
    e_ = onh.from_array(np.array([1, 10, 30, 30], np.int64), "sl_e")
    n1 = oh.make_node("Tile", ["input", "tile_reps"], ["tiled"])
    n2 = oh.make_node("Slice", ["tiled", "sl_s", "sl_e"], ["output"])
    vi = oh.make_tensor_value_info("tiled", FLOAT, [1, 10, rh * 30, rw * 30])
    return _make_model([n1, n2], [reps, s_, e_], [vi])


def constant_output(grid_np):
    """
    Output is a fixed constant [1,10,30,30] array regardless of input.

    Implementation: Where(false_mask, input, const_out).
    - 0 MACs
    - no intermediate value_info tensors
    - params = 9000 (const float) + 9000 (bool mask)
    """
    const_tensor = onh.from_array(
        grid_np.astype(np.float32).reshape(1, 10, 30, 30), "const_out"
    )
    false_mask = onh.from_array(np.zeros((1, 10, 30, 30), dtype=np.bool_), "false_mask")
    node = oh.make_node("Where", ["false_mask", "input", "const_out"], ["output"])
    return _make_model([node], [const_tensor, false_mask])


def constant_output_half(grid_np):
    """Same but with float16 weights — halves param memory counted by onnx_tool?
    Actually onnx_tool counts elements not bytes, so same params.
    But calculate_memory counts bytes of value_info tensors, not initializers.
    Use float32 — float16 not worth it here."""
    return constant_output(grid_np)


def conv1x1(W_np, bias_np=None):
    """
    Single 1×1 convolution: W shape [out_ch, in_ch, 1, 1].
    MACs = out_ch * in_ch * 1 * 1 * 30 * 30
    For 10×10: MACs = 90,000. Params = 100.
    Cost ≈ 90,100. Score ≈ 13.6.
    """
    W = onh.from_array(W_np.astype(np.float32), "W")
    inputs = ["input", "W"]
    if bias_np is not None:
        B = onh.from_array(bias_np.astype(np.float32), "B")
        inputs.append("B")
    else:
        B = None
    node = oh.make_node(
        "Conv", inputs, ["output"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]
    )
    inits = [W] + ([B] if B else [])
    return _make_model([node], inits)


def conv3x3(W_np, bias_np=None):
    """
    Single 3×3 convolution with same padding.
    MACs = out_ch * in_ch * 9 * 30 * 30.
    For 10×10: MACs = 810,000. Params = 900.
    Cost ≈ 810,900. Score ≈ 11.4.
    """
    W = onh.from_array(W_np.astype(np.float32), "W")
    inputs = ["input", "W"]
    if bias_np is not None:
        B = onh.from_array(bias_np.astype(np.float32), "B")
        inputs.append("B")
    else:
        B = None
    node = oh.make_node(
        "Conv", inputs, ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    inits = [W] + ([B] if B else [])
    return _make_model([node], inits)


def conv_stack(weights_biases):
    """
    Multi-layer conv stack. weights_biases = [(W1,b1,ks1), (W2,b2,ks2), ...]
    Intermediate tensors named tmp0, tmp1, ...
    """
    nodes = []
    inits = []
    value_infos = []
    prev = "input"
    for i, (W_np, b_np, ks) in enumerate(weights_biases):
        is_last = i == len(weights_biases) - 1
        out_name = "output" if is_last else f"tmp{i}"
        wname = f"W{i}"
        bname = f"B{i}"
        W = onh.from_array(W_np.astype(np.float32), wname)
        pad = ks // 2
        inp_list = [prev, wname]
        if b_np is not None:
            B = onh.from_array(b_np.astype(np.float32), bname)
            inits.append(B)
            inp_list.append(bname)
        inits.append(W)
        node = oh.make_node(
            "Conv",
            inp_list,
            [out_name],
            kernel_shape=[ks, ks],
            pads=[pad, pad, pad, pad],
        )
        nodes.append(node)
        if not is_last:
            # Need activation between layers — use Relu
            relu_out = f"relu{i}"
            relu_node = oh.make_node("Relu", [out_name], [relu_out])
            nodes.append(relu_node)
            vi = oh.make_tensor_value_info(out_name, FLOAT, GRID)
            vi2 = oh.make_tensor_value_info(relu_out, FLOAT, GRID)
            value_infos.extend([vi, vi2])
            prev = relu_out
        else:
            prev = out_name
    return _make_model(nodes, inits, value_infos)


def score_estimate(macs, memory, params):
    cost = macs + memory + params
    return max(1.0, 25.0 - math.log(max(1.0, cost)))
