"""
Symbolic pipeline — solves all 400 tasks with zero/near-zero MAC ONNX networks.
Saves results to output/ directory.
Outputs a list of tasks that need neural training (for RunPod).

Usage:
    python src/pipeline_symbolic.py [--data_dir data] [--output_dir output]
"""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "data")

from src.analyze import (
    analyze_task,
    check_content_aware_hflip,
    check_content_aware_rot180,
    check_content_aware_vflip,
    get_examples,
    load_task,
)
from src.onnx_builder import (
    col_gather,
    col_then_color,
    color_permutation,
    constant_output,
    content_aware_hflip,
    content_aware_rot180,
    content_aware_vflip,
    conv1x1,
    conv3x3,
    geo_then_color_perm,
    hflip,
    identity,
    rot90_ccw,
    rot90_cw,
    rot180,
    rot270_cw,
    row_col_gather,
    row_gather,
    row_then_color,
    score_estimate,
    tile_2x2,
    tile_3x3,
    tile_hw,
    transp_v,
    transpose_hw,
    vflip,
)

try:
    from neurogolf_utils import neurogolf_utils as ng

    HAS_NG = True
except ImportError:
    HAS_NG = False


PATTERN_TO_BUILDER = {
    "identity": lambda p: identity(),
    "hflip": lambda p: hflip(),
    "vflip": lambda p: vflip(),
    "rot180": lambda p: rot180(),
    "transpose_hw": lambda p: transpose_hw(),
    "rot90_cw": lambda p: rot90_cw(),
    "rot90_ccw": lambda p: rot90_ccw(),
    "rot270_cw": lambda p: rot90_ccw(),
    "transp_v": lambda p: transp_v(),
    "color_perm": lambda p: color_permutation(p),
    "row_perm": lambda p: row_gather(p),
    "col_perm": lambda p: col_gather(p),
    "row_then_color": lambda p: row_then_color(p[0], p[1]),
    "col_then_color": lambda p: col_then_color(p[0], p[1]),
    "row_col_perm": lambda p: row_col_gather(p[0], p[1]),
    "geo_color_perm": lambda p: geo_then_color_perm(p[0], p[1]),
    "tile_2x2": lambda p: tile_2x2(),
    "tile_3x3": lambda p: tile_3x3(),
    "tile_nxm": lambda p: tile_hw(p[0], p[1]),
    "constant": lambda p: constant_output(p),
    "content_aware_hflip": lambda p: content_aware_hflip(p[0], p[1], p[2], p[3]),
    "content_aware_vflip": lambda p: content_aware_vflip(p[0], p[1], p[2], p[3]),
    "content_aware_rot180": lambda p: content_aware_rot180(p[0], p[1], p[2], p[3]),
}


def verify_onnx_model(model: onnx.ModelProto, examples) -> bool:
    """Run all examples through the model and check correctness."""
    try:
        model_bytes = model.SerializeToString()
        session = onnxruntime.InferenceSession(model_bytes)
    except Exception as e:
        print(f"      [VERIFY] Session error: {e}")
        return False

    for i_oh, o_oh in examples:
        inp = i_oh[np.newaxis].astype(np.float32)  # [1,10,30,30]
        try:
            result = session.run(["output"], {"input": inp})[0]
        except Exception as e:
            print(f"      [VERIFY] Run error: {e}")
            return False
        pred = (result[0] > 0.0).astype(np.float32)
        if not np.array_equal(pred, o_oh):
            return False
    return True


def check_static_shapes(model: onnx.ModelProto) -> bool:
    """Check that all tensor shapes are statically defined (required by scorer)."""
    try:
        from neurogolf_utils import neurogolf_utils as ng_inner
    except ImportError:
        return True  # Can't check, assume OK

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmppath = f.name
        onnx.save(model, tmppath)
    try:
        macs, mem, params = ng_inner.score_network(tmppath)
        return macs is not None
    except Exception:
        return False
    finally:
        os.unlink(tmppath)


def get_model_score(model: onnx.ModelProto):
    """Returns (macs, memory, params, score) or None if invalid."""
    import tempfile

    if not HAS_NG:
        return None
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmppath = f.name
        onnx.save(model, tmppath)
    try:
        macs, mem, params = ng.score_network(tmppath)
        if macs is None:
            return None
        cost = macs + mem + params
        sc = max(1.0, 25.0 - math.log(max(1.0, cost)))
        return macs, mem, params, sc
    except Exception:
        return None
    finally:
        os.unlink(tmppath)


def solve_task(
    task_num: int, data_dir: str, output_dir: str, existing_score: float = None
) -> dict:
    """
    Try to solve task with symbolic approach.
    Returns dict with keys: task_num, pattern, score, status
    """
    result = {
        "task_num": task_num,
        "pattern": None,
        "score": existing_score,
        "status": "skipped",
    }

    # Analyze the task
    try:
        analysis = analyze_task(task_num, data_dir)
    except Exception as e:
        result["status"] = "analyze_error"
        result["error"] = str(e)
        return result

    pattern = analysis.get("pattern")
    params = analysis.get("params")

    if pattern == "unknown":
        result["status"] = "needs_neural"
        return result

    # Build the ONNX model
    builder = PATTERN_TO_BUILDER.get(pattern)
    if builder is None:
        result["status"] = "needs_neural"
        return result

    try:
        model = builder(params)
    except Exception as e:
        print(f"  Task {task_num:03d}: Build error for pattern '{pattern}': {e}")
        traceback.print_exc()
        result["status"] = "build_error"
        result["error"] = str(e)
        return result

    # Verify on ALL examples (train + test + arc-gen)
    task = load_task(task_num, data_dir)
    examples = get_examples(task, include_arcgen=True)
    if not verify_onnx_model(model, examples):
        result["status"] = "verify_failed"
        result["pattern"] = pattern
        return result

    # Get score
    score_data = get_model_score(model)
    new_score = score_data[3] if score_data else analysis.get("score_est", 20.0)

    # Only save if better than existing
    out_path = Path(output_dir) / f"task{task_num:03d}.onnx"
    if existing_score is not None and new_score <= existing_score:
        result["status"] = "kept_existing"
        result["pattern"] = pattern
        result["score"] = existing_score
        return result

    # Save
    onnx.save(model, str(out_path))
    result["status"] = "solved"
    result["pattern"] = pattern
    result["score"] = new_score
    if score_data:
        result["macs"] = score_data[0]
        result["memory"] = score_data[1]
        result["params"] = score_data[2]
    return result


def run_pipeline(data_dir: str = "data", output_dir: str = "output", tasks=None):
    """
    Main pipeline runner.
    Returns: list of task results, list of tasks needing neural training.
    """
    os.makedirs(output_dir, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = list(range(1, 401))

    # Get existing scores for all tasks
    existing_scores = {}
    if HAS_NG:
        print("Scoring existing ONNX files...")
        for t in tasks:
            fpath = Path(output_dir) / f"task{t:03d}.onnx"
            if fpath.exists():
                try:
                    macs, mem, params = ng.score_network(str(fpath))
                    if macs is not None:
                        cost = macs + mem + params
                        existing_scores[t] = max(1.0, 25.0 - math.log(max(1.0, cost)))
                except Exception:
                    pass

    results = []
    needs_neural = []
    solved_count = 0
    improved_count = 0
    total_score = sum(existing_scores.values())

    print(f"\n{'=' * 60}")
    print(f"Starting symbolic solver on {len(tasks)} tasks")
    print(f"{'=' * 60}\n")

    for i, task_num in enumerate(tasks):
        existing = existing_scores.get(task_num)
        sys.stdout.write(f"\r[{i + 1:3d}/{len(tasks)}] Task {task_num:03d}...")
        sys.stdout.flush()

        result = solve_task(task_num, data_dir, output_dir, existing)
        results.append(result)

        status = result["status"]
        pattern = result.get("pattern", "")
        pattern_str = str(pattern) if pattern is not None else ""
        score = result.get("score", 0)
        score_num = float(score) if isinstance(score, (int, float)) else 0.0

        if status == "solved":
            solved_count += 1
            old = existing or 0
            gain = score_num - old
            improved_count += 1 if gain > 0.01 else 0
            total_score += gain
            sym = "✓"
        elif status == "needs_neural":
            needs_neural.append(task_num)
            sym = "○"
        elif status == "kept_existing":
            sym = "="
        else:
            sym = "✗"

        print(
            f"\r  [{i + 1:3d}/{len(tasks)}] {sym} Task {task_num:03d}: "
            f"{status:15s} | pattern={pattern_str:12s} | score={score_num:.1f}"
        )

    print(f"\n{'=' * 60}")
    print(f"  Solved symbolically: {solved_count}")
    print(f"  Improved:            {improved_count}")
    print(f"  Needs neural:        {len(needs_neural)}")
    print(f"  Estimated total:     {total_score:.1f}")
    print(f"{'=' * 60}\n")

    # Save needs_neural list
    neural_path = Path(output_dir) / "needs_neural.json"
    with open(neural_path, "w") as f:
        json.dump(needs_neural, f)
    print(f"Tasks needing neural training saved to: {neural_path}")
    print(f"Neural tasks: {needs_neural}")

    return results, needs_neural


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--tasks", nargs="*", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
