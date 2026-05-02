#!/usr/bin/env python3
"""
NeuroGolf 2026 — RunPod Neural Trainer
=======================================
Trains minimal PyTorch CNNs for tasks that couldn't be solved symbolically.
Uses ALL 262 arc-gen examples per task for training + generalization.

Setup on RunPod:
    pip install onnx==1.21.0 onnxruntime==1.24.4 onnx-tool==1.0.1 numpy==2.4.4 torch

Upload:
    1. This script
    2. data/ directory (task001.json ... task400.json + neurogolf_utils/)
    3. output/needs_neural.json  (list of tasks needing neural training)
    4. output/*.onnx             (existing solutions for comparison)

Run:
    python train_neural.py --data_dir data --output_dir output

"""

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnxruntime
import torch.nn.functional as F

COLORS = 10
H = W = 30
FLOAT = onnx.TensorProto.FLOAT
IR_VER = 10
OPSET = [oh.make_opsetid("", 10)]
GRID = [1, 10, 30, 30]

# ─────────────────────────────────────────────
# PyTorch imports (GPU training)
# ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available")


# ─────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────


def grid_to_onehot(grid):
    oh_arr = np.zeros((COLORS, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < COLORS and r < H and c < W:
                oh_arr[color, r, c] = 1.0
    return oh_arr


def load_examples(task_num, data_dir):
    """Load ALL examples (train + test + arc-gen) as numpy arrays."""
    with open(Path(data_dir) / f"task{task_num:03d}.json") as f:
        task = json.load(f)
    examples = []
    for ex in task.get("train", []) + task.get("test", []) + task.get("arc-gen", []):
        inp, out = ex.get("input", []), ex.get("output", [])
        if not inp or not out:
            continue
        if max(len(inp), max(len(r) for r in inp) if inp else 0) > 30:
            continue
        if max(len(out), max(len(r) for r in out) if out else 0) > 30:
            continue
        i_oh = grid_to_onehot(inp)
        o_oh = grid_to_onehot(out)
        examples.append((i_oh, o_oh))
    return examples


# ─────────────────────────────────────────────
# PyTorch model architectures (smallest first)
# ─────────────────────────────────────────────


class Conv1x1Net(nn.Module):
    """1-layer 1×1 conv. Best cost if task is solvable this way.
    MACs = C_out * C_in * H * W = 10 * 10 * 30 * 30 = 90,000
    Params = 100. Score ≈ 13.6"""

    def __init__(self, in_ch=10, out_ch=10):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x):
        return self.conv(x)


class Conv3x3Net(nn.Module):
    """1-layer 3×3 conv. MACs = 810,000, Params = 900. Score ≈ 11.4"""

    def __init__(self, in_ch=10, out_ch=10, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class TinyUNet(nn.Module):
    """2-layer CNN: 3×3 → 1×1. Mid channels = 4.
    MACs = (10*4*9 + 4*10*1)*H*W = 3,240,000+. Score ≈ 10."""

    def __init__(self, in_ch=10, out_ch=10, mid_ch=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, bias=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SmallCNN(nn.Module):
    """3-layer CNN: 3×3 → 3×3 → 1×1 with mid_ch=4.
    Still relatively cheap, good for medium-hard tasks."""

    def __init__(self, in_ch=10, out_ch=10, mid_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class MedCNN(nn.Module):
    """4-layer CNN for harder tasks."""

    def __init__(self, in_ch=10, out_ch=10, mid_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class LargeCNN(nn.Module):
    """5-layer CNN for the hardest tasks. Fallback."""

    def __init__(self, in_ch=10, out_ch=10, mid_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


# Architecture ladder (cheapest first)
ARCHITECTURES = [
    ("conv1x1", Conv1x1Net, {}),
    ("conv3x3", Conv3x3Net, {}),
    ("tiny_unet", TinyUNet, {"mid_ch": 4}),
    ("small_cnn", SmallCNN, {"mid_ch": 4}),
    ("small_cnn", SmallCNN, {"mid_ch": 8}),
    ("med_cnn", MedCNN, {"mid_ch": 8}),
    ("med_cnn", MedCNN, {"mid_ch": 16}),
    ("large_cnn", LargeCNN, {"mid_ch": 16}),
    ("large_cnn", LargeCNN, {"mid_ch": 32}),
]


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────


def train_model(model, examples, epochs=2000, lr=1e-3, patience=300):
    """
    Train a PyTorch model on ARC examples.
    Uses per-pixel multiclass cross-entropy on argmax color channels.
    Returns (model, best_loss).
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build tensors
    X = (
        torch.from_numpy(np.stack([e[0] for e in examples])).float().to(DEVICE)
    )  # [N,10,30,30]
    Y = (
        torch.from_numpy(np.stack([e[1] for e in examples])).float().to(DEVICE)
    )  # [N,10,30,30]
    Y_cls = Y.argmax(dim=1).long()  # [N,30,30]

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)  # [N,10,30,30]
        loss = F.cross_entropy(logits, Y_cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = float(loss.item())
        if lv < best_loss - 1e-7:
            best_loss = lv
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model = model.cpu()
    return model, best_loss


def model_is_correct(model, examples) -> bool:
    """Check model produces correct one-hot outputs for all examples."""
    model.eval()
    with torch.no_grad():
        for i_oh, o_oh in examples:
            x = torch.from_numpy(i_oh[np.newaxis]).float()
            logits = model(x).numpy()[0]  # [10,30,30]
            cls = np.argmax(logits, axis=0)  # [30,30]
            pred = np.zeros_like(o_oh, dtype=np.float32)
            for c in range(10):
                pred[c] = (cls == c).astype(np.float32)
            if not np.array_equal(pred, o_oh):
                return False
    return True


# ─────────────────────────────────────────────
# ONNX export with static shapes
# ─────────────────────────────────────────────


def export_to_onnx(model, path: str):
    """Export PyTorch model to ONNX with fully static shapes (no dynamic axes)."""
    model.eval()
    dummy = torch.zeros(1, 10, 30, 30)
    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=10,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # All shapes static!
        do_constant_folding=True,
        verbose=False,
    )
    # Load and fix: ensure shapes are fully static via shape inference
    m = onnx.load(path)
    # Explicitly set shapes for input and output to be extra safe
    for item in list(m.graph.input) + list(m.graph.output):
        tt = item.type.tensor_type
        if tt.HasField("shape"):
            for dim in tt.shape.dim:
                dim.dim_value = 30 if dim.dim_value <= 0 else dim.dim_value
            # Force batch dimension to 1 and colors to 10 if they are flexible
            if len(tt.shape.dim) >= 2:
                tt.shape.dim[0].dim_value = 1
                tt.shape.dim[1].dim_value = 10

    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m)
    onnx.save(m, path)


def verify_onnx(path: str, examples) -> bool:
    """Verify ONNX model produces correct outputs."""
    try:
        session = onnxruntime.InferenceSession(path)
    except Exception as e:
        print(f"      ONNX load error: {e}")
        return False
    for i_oh, o_oh in examples:
        inp = i_oh[np.newaxis].astype(np.float32)
        try:
            result = session.run(["output"], {"input": inp})[0]
        except Exception as e:
            print(f"      ONNX run error: {e}")
            return False
        logits = result[0][0]
        cls = np.argmax(logits, axis=0)
        pred = np.zeros_like(o_oh, dtype=np.float32)
        for c in range(10):
            pred[c] = (cls == c).astype(np.float32)
        if not np.array_equal(pred, o_oh):
            return False
    return True


def get_onnx_score(path: str):
    """Get (macs, memory, params, score) from onnx_tool."""
    try:
        import onnx_tool

        model = onnx_tool.loadmodel(path, {"verbose": False, "constant_folding": True})
        g = model.graph
        g.graph_reorder_nodes()
        g.shape_infer(None)
        g.profile()
        if not g.valid_profile:
            return None

        # calculate_memory equivalent
        m = onnx.load(path)
        m2 = onnx.shape_inference.infer_shapes(m)
        memory = 0
        for item in (
            list(m2.graph.input) + list(m2.graph.value_info) + list(m2.graph.output)
        ):
            if item.name in ["input", "output"]:
                continue
            if not item.type.HasField("tensor_type"):
                continue
            tt = item.type.tensor_type
            if not tt.HasField("shape"):
                return None
            n = 1
            for dim in tt.shape.dim:
                if not dim.HasField("dim_value"):
                    return None
                n *= dim.dim_value
            np_dt = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
            memory += n * np.dtype(np_dt).itemsize

        macs = int(sum(g.macs))
        params = int(g.params)
        cost = macs + memory + params
        sc = max(1.0, 25.0 - math.log(max(1.0, cost)))
        return macs, memory, params, sc
    except Exception as e:
        print(f"      Score error: {e}")
        return None


# ─────────────────────────────────────────────
# Per-task solver
# ─────────────────────────────────────────────


def solve_task_neural(task_num: int, data_dir: str, output_dir: str) -> dict:
    """
    Try each architecture from cheapest to most expensive.
    Stop as soon as a correct one is found.
    """
    result = {"task_num": task_num, "status": "failed", "score": 0}

    examples = load_examples(task_num, data_dir)
    if not examples:
        result["status"] = "no_examples"
        return result

    # Split: keep last 10% as validation (never used in training)
    n_val = max(1, len(examples) // 10)
    val_examples = examples[-n_val:]
    train_examples = examples[:-n_val]
    if not train_examples:
        train_examples = examples

    print(
        f"  Task {task_num:03d}: {len(train_examples)} train, {len(val_examples)} val examples"
    )

    out_path = str(Path(output_dir) / f"task{task_num:03d}.onnx")

    # Existing score
    existing_score = 0
    if Path(out_path).exists():
        sc_data = get_onnx_score(out_path)
        existing_score = sc_data[3] if sc_data else 0

    best_score = existing_score
    best_model_path = out_path if existing_score > 0 else None

    for arch_name, ModelClass, kwargs in ARCHITECTURES:
        print(f"    Trying {arch_name} {kwargs}...", end="", flush=True)

        best_local_model = None
        best_local_loss = float("inf")
        n_restarts = 3

        last_err = None
        for seed in range(n_restarts):
            try:
                torch.manual_seed(1234 + seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(1234 + seed)
            except Exception:
                pass

            model = ModelClass(**kwargs)
            n_params = sum(p.numel() for p in model.parameters())
            epochs = 4000 if n_params < 500 else 3000 if n_params < 2000 else 1800
            try:
                trained, loss = train_model(
                    model, train_examples, epochs=epochs, lr=1e-3, patience=400
                )
            except Exception as e:
                last_err = e
                continue
            if loss < best_local_loss:
                best_local_loss = loss
                best_local_model = trained

        if best_local_model is None:
            print(f" ERROR: training failed ({last_err})")
            continue

        model = best_local_model
        print(f" loss={best_local_loss:.4f}", end="", flush=True)

        # Check correctness on ALL examples (including val)
        if not model_is_correct(model, examples):
            print(" ✗ (wrong output)")
            continue

        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmppath = f.name

        try:
            export_to_onnx(model, tmppath)
        except Exception as e:
            print(f" ONNX export error: {e}")
            os.unlink(tmppath)
            continue

        # Verify ONNX (different from PyTorch due to export)
        if not verify_onnx(tmppath, examples):
            print(f" ✗ (ONNX mismatch)")
            os.unlink(tmppath)
            continue

        # Score
        sc_data = get_onnx_score(tmppath)
        if sc_data is None:
            print(f" ✗ (shape error)")
            os.unlink(tmppath)
            continue

        macs, memory, params, sc = sc_data
        print(f" ✓ score={sc:.2f} (MACs={macs:,}, mem={memory:,}, params={params})")

        # Keep if better
        if sc > best_score:
            best_score = sc
            if best_model_path and best_model_path != out_path:
                os.unlink(best_model_path)
            import shutil

            shutil.copy(tmppath, out_path)
            best_model_path = out_path
            result["status"] = "solved"
            result["score"] = sc
            result["arch"] = arch_name
            result["macs"] = macs
            result["memory"] = memory
            result["params"] = params
        else:
            print(f"    (score {sc:.2f} not better than existing {best_score:.2f})")

        os.unlink(tmppath)

        # Stop if we have a great score already
        if best_score >= 22:
            break

    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Path to task JSON files")
    parser.add_argument(
        "--output_dir", default="output", help="Path to save ONNX files"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        type=int,
        default=None,
        help="Specific task numbers (default: load from needs_neural.json)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Re-train ALL tasks, not just needs_neural"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tasks:
        tasks = args.tasks
    elif args.all:
        tasks = list(range(1, 401))
    else:
        neural_path = Path(args.output_dir) / "needs_neural.json"
        if neural_path.exists():
            with open(neural_path) as f:
                tasks = json.load(f)
            print(f"Loaded {len(tasks)} tasks from {neural_path}")
        else:
            print("No needs_neural.json found. Run pipeline_symbolic.py first.")
            print("Falling back to all 400 tasks...")
            tasks = list(range(1, 401))

    print(f"\n{'=' * 60}")
    print(f"Neural trainer: {len(tasks)} tasks | Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    results = []
    total_score = 0
    solved = 0

    for i, task_num in enumerate(tasks):
        print(f"\n[{i + 1}/{len(tasks)}] Task {task_num:03d}")
        result = solve_task_neural(task_num, args.data_dir, args.output_dir)
        results.append(result)
        if result["status"] == "solved":
            solved += 1
            total_score += result["score"]
            print(f"  ✓ Task {task_num:03d} solved: score={result['score']:.2f}")
        else:
            print(f"  ✗ Task {task_num:03d} {result['status']}")

    # Save results
    results_path = Path(args.output_dir) / "neural_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Solved: {solved}/{len(tasks)}")
    print(f"Score from neural: {total_score:.1f}")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
