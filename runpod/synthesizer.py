import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

def load_task(task_num, data_dir="data"):
    task_path = os.path.join(data_dir, f"task{task_num:03d}.json")
    if not os.path.exists(task_path): return None
    with open(task_path, "r") as f: return json.load(f)


def parse_tasks_arg(values):
    """Parses task selectors like ['1-50', '73', '100-120'] into sorted unique ints."""
    if not values:
        return list(range(1, 401))
    tasks = set()
    for token in values:
        token = str(token).strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            lo, hi = (a, b) if a <= b else (b, a)
            for t in range(lo, hi + 1):
                if 1 <= t <= 400:
                    tasks.add(t)
        else:
            t = int(token)
            if 1 <= t <= 400:
                tasks.add(t)
    return sorted(tasks) if tasks else list(range(1, 401))

def to_one_hot(grid, h=30, w=30):
    """Converts a grid to a (10, 30, 30) one-hot tensor."""
    t = torch.zeros((10, h, w), device='cuda' if torch.cuda.is_available() else 'cpu')
    g = torch.tensor(grid, device=t.device)
    gh, gw = min(h, g.shape[0]), min(w, g.shape[1])
    for c in range(10):
        t[c, :gh, :gw] = (g[:gh, :gw] == c).float()
    return t

def get_symmetries(X):
    """Returns 8 symmetries of a (C, H, W) tensor."""
    return [
        ("id", X),
        ("rot90", X.transpose(1, 2).flip(2)),
        ("rot180", X.flip(1).flip(2)),
        ("rot270", X.transpose(1, 2).flip(1)),
        ("fliph", X.flip(2)),
        ("flipv", X.flip(1)),
        ("transp", X.transpose(1, 2)),
        ("transp_v", X.transpose(1, 2).flip(1).flip(2)),
    ]


def apply_symmetry_grid(grid, sym_name):
    """Apply a 2D symmetry to a padded 30x30 integer grid."""
    if sym_name == "id":
        return grid
    if sym_name == "rot90":
        return grid.transpose(0, 1).flip(1)
    if sym_name == "rot180":
        return grid.flip(0).flip(1)
    if sym_name == "rot270":
        return grid.transpose(0, 1).flip(0)
    if sym_name == "fliph":
        return grid.flip(1)
    if sym_name == "flipv":
        return grid.flip(0)
    if sym_name == "transp":
        return grid.transpose(0, 1)
    if sym_name == "transp_v":
        return grid.transpose(0, 1).flip(0).flip(1)
    return grid


def output_bounds(output_grid):
    """Return the active bounds of a 2D output grid."""
    rows = np.where(output_grid.any(axis=1))[0]
    cols = np.where(output_grid.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0
    return int(rows[-1] + 1), int(cols[-1] + 1)

def find_spatial_mapping(task):
    """Finds a consistent symmetry and color mapping for a task."""
    examples = []
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            if ex.get("input") and ex.get("output"):
                examples.append(ex)
    if not examples:
        return None

    sym_names = ["id", "rot90", "rot180", "rot270", "fliph", "flipv", "transp", "transp_v"]

    def build_mapping_from_pairs(pairs):
        global_mapping = {}
        for v_in, v_out in pairs:
            if v_in in global_mapping and global_mapping[v_in] != v_out:
                return None
            global_mapping[v_in] = v_out
        full_mapping = list(range(10))
        for k, v in global_mapping.items():
            full_mapping[k] = v
        return full_mapping

    # Family 1: horizontal duplication by a factor of 2.
    for sym_name in sym_names:
        global_pairs = []
        possible = True
        for ex in examples:
            inp_raw = np.array(ex["input"])
            out_raw = np.array(ex["output"])
            if out_raw.shape != (inp_raw.shape[0], inp_raw.shape[1] * 2):
                possible = False
                break
            inp = apply_symmetry_grid(torch.tensor(inp_raw).clone(), sym_name).numpy()
            expected = np.concatenate([inp, inp], axis=1)
            if not np.array_equal(expected, out_raw):
                possible = False
                break
            for r in range(out_raw.shape[0]):
                for c in range(out_raw.shape[1]):
                    global_pairs.append((int(inp[r, c % inp.shape[1]]), int(out_raw[r, c])))
        if possible:
            full_mapping = build_mapping_from_pairs(global_pairs)
            if full_mapping is not None:
                return {"kind": "repeat_h2", "sym": sym_name, "mapping": full_mapping}

    # Family 2: top-left quadrant crop for odd-sized grids with a central divider.
    for sym_name in sym_names:
        global_pairs = []
        possible = True
        for ex in examples:
            inp_raw = np.array(ex["input"])
            out_raw = np.array(ex["output"])
            if inp_raw.shape[0] % 2 == 0 or inp_raw.shape[1] % 2 == 0:
                possible = False
                break
            expected = inp_raw[: inp_raw.shape[0] // 2, : inp_raw.shape[1] // 2]
            inp = apply_symmetry_grid(torch.tensor(inp_raw).clone(), sym_name).numpy()
            if not np.array_equal(expected, out_raw):
                possible = False
                break
            for r in range(out_raw.shape[0]):
                for c in range(out_raw.shape[1]):
                    global_pairs.append((int(inp[r, c]), int(out_raw[r, c])))
        if possible:
            full_mapping = build_mapping_from_pairs(global_pairs)
            if full_mapping is not None:
                return {"kind": "crop_quadrant", "sym": sym_name, "mapping": full_mapping}

    # Existing fixed-size periodic crop/tile search.
    for sym_name in sym_names:
        out_shapes = [np.array(ex['output']).shape for ex in examples]
        if not all(s == out_shapes[0] for s in out_shapes):
            continue
        Ho, Wo = out_shapes[0]

        sym_inputs = []
        for ex in examples:
            inp_raw = np.array(ex['input'])
            Hi_raw, Wi_raw = inp_raw.shape
            inp = torch.zeros((30, 30), dtype=torch.long)
            inp[:Hi_raw, :Wi_raw] = torch.tensor(inp_raw)
            inp = apply_symmetry_grid(inp, sym_name)
            sym_inputs.append(inp)

        max_block_h = min(Ho, 30)
        max_block_w = min(Wo, 30)
        for bh in range(1, max_block_h + 1):
            for bw in range(1, max_block_w + 1):
                for r in range(30 - bh + 1):
                    for c in range(30 - bw + 1):
                        global_mapping = {}
                        possible = True
                        for i, ex in enumerate(examples):
                            out = torch.tensor(ex['output'], device='cpu')
                            for pr in range(Ho):
                                src_r = r + (pr % bh)
                                for pc in range(Wo):
                                    src_c = c + (pc % bw)
                                    v_in = int(sym_inputs[i][src_r, src_c].item())
                                    v_out = int(out[pr, pc].item())
                                    if v_in in global_mapping:
                                        if global_mapping[v_in] != v_out:
                                            possible = False
                                            break
                                    else:
                                        global_mapping[v_in] = v_out
                                if not possible:
                                    break
                            if not possible:
                                break

                        if possible:
                            full_mapping = list(range(10))
                            for k, v in global_mapping.items():
                                full_mapping[k] = v
                            return {"kind": "spatial_window", "sym": sym_name, "r": r, "c": c, "bh": bh, "bw": bw, "Ho": Ho, "Wo": Wo, "mapping": full_mapping}

    return None

def create_symbolic_model(spec_or_sym_name, r=None, c=None, bh=None, bw=None, Ho=None, Wo=None, mapping=None):
    """Generates a symbolic ONNX model for a detected task family."""
    if isinstance(spec_or_sym_name, dict):
        spec = spec_or_sym_name
        kind = spec["kind"]
        sym_name = spec.get("sym", "id")
        mapping = spec["mapping"]
        r = spec.get("r", 0)
        c = spec.get("c", 0)
        bh = spec.get("bh", 30)
        bw = spec.get("bw", 30)
        Ho = spec.get("Ho", 30)
        Wo = spec.get("Wo", 30)
    else:
        kind = "spatial_window"
        sym_name = spec_or_sym_name

    rev_mapping = [0] * 10
    for ci in range(10):
        co = mapping[ci]
        rev_mapping[co] = ci

    indices_map = np.zeros((1, 10, 30, 30, 4), dtype=np.int64)
    indices_map[..., 0] = 0
    indices_map[..., 1] = 0
    indices_map[..., 2] = 0
    indices_map[..., 3] = 0

    def backtrack(sym, src_r, src_c):
        ri, wi = src_r, src_c
        if sym == "rot90":
            ri, wi = 29 - src_c, src_r
        elif sym == "rot180":
            ri, wi = 29 - src_r, 29 - src_c
        elif sym == "rot270":
            ri, wi = src_c, 29 - src_r
        elif sym == "fliph":
            ri, wi = src_r, 29 - src_c
        elif sym == "flipv":
            ri, wi = 29 - src_r, src_c
        elif sym == "transp":
            ri, wi = src_c, src_r
        elif sym == "transp_v":
            ri, wi = 29 - src_c, 29 - src_r
        return ri, wi

    if kind == "repeat_h2":
        for co in range(10):
            ci = rev_mapping[co]
            for pr in range(30):
                for pc in range(30):
                    src_r, src_c = pr, pc // 2
                    ri, wi = backtrack(sym_name, src_r, src_c)
                    indices_map[0, co, pr, pc] = [0, ci, ri, wi]
    elif kind == "crop_quadrant":
        for co in range(10):
            ci = rev_mapping[co]
            for pr in range(30):
                for pc in range(30):
                    if pr < 15 and pc < 15:
                        src_r, src_c = pr, pc
                        ri, wi = backtrack(sym_name, src_r, src_c)
                        indices_map[0, co, pr, pc] = [0, ci, ri, wi]
                    else:
                        indices_map[0, co, pr, pc] = [0, 0, 0, 0]
    else:
        for co in range(10):
            ci = rev_mapping[co]
            for pr in range(Ho):
                src_r = r + (pr % bh)
                for pc in range(Wo):
                    src_c = c + (pc % bw)
                    ri, wi = backtrack(sym_name, src_r, src_c)
                    indices_map[0, co, pr, pc] = [0, ci, ri, wi]

    inits = [onh.from_array(indices_map, 'indices')]
    nodes = [oh.make_node('GatherND', ['input', 'indices'], ['output'])]

    input_info = oh.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = oh.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 10, 30, 30])

    graph = oh.make_graph(nodes, 'neurogolf_synth', [input_info], [output_info], inits)
    model = oh.make_model(graph, ir_version=10, opset_imports=[oh.make_opsetid('', 12)])

    model = onnx.shape_inference.infer_shapes(model)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output_synth")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Task ids/ranges, e.g. --tasks 1-100 133 200-240")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = parse_tasks_arg(args.tasks)
    print(f"Starting synthesis on {len(tasks)} tasks...")
    solved = 0
    for i, t in enumerate(tasks, start=1):
        task = load_task(t, args.data_dir)
        if not task: continue
        
        if i % 50 == 0:
            print(f"  Processing Task {t:03d} ({i}/{len(tasks)})...")
            
        mapping_data = find_spatial_mapping(task)
        if mapping_data:
            if isinstance(mapping_data, dict):
                print(f"✓ Solved Task {t:03d}: family={mapping_data['kind']}, sym={mapping_data.get('sym', 'id')}")
            else:
                sym_name, r, c, bh, bw, Ho, Wo, full_mapping = mapping_data
                print(f"✓ Solved Task {t:03d}: Sym={sym_name}, Window=({r},{c}) {bh}x{bw} -> {Ho}x{Wo}")
            try:
                model = create_symbolic_model(mapping_data)
                onnx.save(model, os.path.join(args.output_dir, f"task{t:03d}.onnx"))
                solved += 1
            except Exception as e:
                print(f"  Error generating ONNX for task {t:03d}: {e}")
            
    print(f"Finished. Total Solved: {solved}/{len(tasks)}")

if __name__ == "__main__": main()
