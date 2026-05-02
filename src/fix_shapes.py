"""
Fix shape-error ONNX files by setting fully static shapes [1,10,30,30].

The existing submission has 75 files where the input/output shapes are dynamic
(H and W are symbolic, not concrete 30). This causes calculate_memory() to fail.

This script rebuilds those files with static shapes, preserving all nodes/weights.
If the fixed model still passes verification, it replaces the broken one.

Usage:
    python src/fix_shapes.py [--output_dir output] [--data_dir data]
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnxruntime

sys.path.insert(0, "data")
try:
    from neurogolf_utils import neurogolf_utils as ng
    HAS_NG = True
except ImportError:
    HAS_NG = False

FLOAT = onnx.TensorProto.FLOAT
GRID = [1, 10, 30, 30]
IR_VER = 10
OPSET = [oh.make_opsetid("", 10)]

SHAPE_ERROR_TASKS = [
    14, 15, 16, 23, 24, 25, 30, 31, 36, 47, 49, 50, 53, 63, 65, 67, 73, 81,
    85, 86, 92, 98, 110, 125, 127, 132, 150, 151, 155, 157, 160, 161, 162,
    166, 170, 177, 180, 185, 187, 193, 196, 198, 220, 222, 224, 225, 226,
    230, 232, 249, 254, 269, 276, 278, 293, 298, 299, 300, 303, 307, 309,
    310, 314, 320, 323, 337, 339, 340, 344, 346, 351, 376, 384, 389, 396
]


def make_static(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Force all input/output/value_info shapes to be static [1,10,30,30].
    Also re-runs shape inference with the static input shape.
    """
    def _set_static_shape(tensor_value_info):
        """Set [1,10,30,30] static shape on a TensorValueInfo proto."""
        tt = tensor_value_info.type.tensor_type
        tt.shape.ClearField('dim')
        for d in [1, 10, 30, 30]:
            dim = tt.shape.dim.add()
            dim.dim_value = d

    # Step 1: Fix input shape
    for inp in model.graph.input:
        if inp.name == "input":
            _set_static_shape(inp)

    # Step 2: Fix output shape
    for out in model.graph.output:
        if out.name == "output":
            _set_static_shape(out)

    # Step 3: Clear stale value_info
    del model.graph.value_info[:]

    # Step 4: Re-run shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:
        model = onnx.shape_inference.infer_shapes(model)

    return model


def verify_model(model_bytes: bytes, examples: list) -> bool:
    """Run examples through the model, return True if all pass."""
    try:
        session = onnxruntime.InferenceSession(model_bytes)
    except Exception as e:
        return False

    for i_oh, o_oh in examples:
        inp = i_oh[np.newaxis].astype(np.float32)
        try:
            result = session.run(["output"], {"input": inp})[0]
        except Exception:
            return False
        pred = (result[0] > 0.0).astype(np.float32)
        if not np.array_equal(pred, o_oh):
            return False
    return True


def load_examples(task_num: int, data_dir: str) -> list:
    """Load all examples (train + test + arc-gen) as (input_onehot, output_onehot) pairs."""
    with open(Path(data_dir) / f"task{task_num:03d}.json") as f:
        task = json.load(f)

    COLORS, H, W = 10, 30, 30
    examples = []
    for ex in task.get("train", []) + task.get("test", []) + task.get("arc-gen", []):
        inp_g = ex.get("input", [])
        out_g = ex.get("output", [])
        if not inp_g or not out_g:
            continue
        if max(len(inp_g), max((len(r) for r in inp_g), default=0)) > 30:
            continue
        if max(len(out_g), max((len(r) for r in out_g), default=0)) > 30:
            continue

        i_oh = np.zeros((COLORS, H, W), dtype=np.float32)
        for r, row in enumerate(inp_g):
            for c, color in enumerate(row):
                if 0 <= color < COLORS:
                    i_oh[color, r, c] = 1.0

        o_oh = np.zeros((COLORS, H, W), dtype=np.float32)
        for r, row in enumerate(out_g):
            for c, color in enumerate(row):
                if 0 <= color < COLORS:
                    o_oh[color, r, c] = 1.0

        examples.append((i_oh, o_oh))
    return examples


def fix_task(task_num: int, output_dir: str, data_dir: str) -> dict:
    """Fix a single task's ONNX file. Returns status dict."""
    result = {"task": task_num, "status": "skipped"}
    fpath = Path(output_dir) / f"task{task_num:03d}.onnx"

    if not fpath.exists():
        result["status"] = "missing"
        return result

    # Load original
    try:
        model = onnx.load(str(fpath))
    except Exception as e:
        result["status"] = f"load_error: {e}"
        return result

    # Apply static shape fix
    try:
        fixed = make_static(model)
    except Exception as e:
        result["status"] = f"fix_error: {e}"
        return result

    # Verify it still works
    examples = load_examples(task_num, data_dir)
    if not examples:
        result["status"] = "no_examples"
        return result

    fixed_bytes = fixed.SerializeToString()
    if not verify_model(fixed_bytes, examples):
        result["status"] = "verify_failed"
        return result

    # Score the fixed model
    score_info = None
    if HAS_NG:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp = f.name
        try:
            onnx.save(fixed, tmp)
            macs, mem, params = ng.score_network(tmp)
            if macs is not None:
                cost = macs + mem + params
                sc = max(1.0, 25.0 - math.log(max(1.0, cost)))
                score_info = (macs, mem, params, sc)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # Save fixed version
    onnx.save(fixed, str(fpath))
    result["status"] = "fixed"
    if score_info:
        result["macs"] = score_info[0]
        result["memory"] = score_info[1]
        result["params"] = score_info[2]
        result["score"] = score_info[3]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--tasks", nargs="*", type=int, default=list(range(1, 401)))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Shape Fix: {len(args.tasks)} tasks")
    print(f"{'='*60}\n")

    total_score_gained = 0
    fixed = 0
    failed = 0

    for i, task_num in enumerate(args.tasks):
        sys.stdout.write(f"\r  [{i+1:2d}/{len(args.tasks)}] task{task_num:03d}... ")
        sys.stdout.flush()

        result = fix_task(task_num, args.output_dir, args.data_dir)
        status = result["status"]

        if status == "fixed":
            fixed += 1
            sc = result.get("score", 0)
            total_score_gained += sc
            print(f"\r  ✓ task{task_num:03d}: fixed → score={sc:.2f} "
                  f"(MACs={result.get('macs',0):,}, "
                  f"mem={result.get('memory',0):,}, "
                  f"params={result.get('params',0)})")
        else:
            failed += 1
            print(f"\r  ✗ task{task_num:03d}: {status}")

    print(f"\n{'='*60}")
    print(f"  Fixed:          {fixed}")
    print(f"  Failed/Missing: {failed}")
    print(f"  Score gained:   ~{total_score_gained:.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
