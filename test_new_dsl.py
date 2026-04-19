import sys
import torch
from neurogolf.grid_codec import decode_tensor_to_grid, get_color_normalization_map, apply_color_map
import numpy as np
from scipy.ndimage import label
from collections import Counter
from neurogolf.solvers import CompositionalSolver
from torch import nn
from search_solvers import _accumulate_bijective_constraints, _complete_permutation, check_solve
from neurogolf.solvers import GeneralColorRemapSolver

def _score_candidate(pred_arr: np.ndarray, expected: list[list[int]]) -> float:
    exp_arr = np.array(expected, dtype=np.int32)
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
    
    return 0.6 * pixel_score + 0.2 * obj_score + 0.1 * bbox_score + 0.1 * hist_score

def _parse_shift(n: str):
    if not n.startswith("shift_"): return None, None
    parts = n.split("_")
    try: return int(parts[1]), int(parts[2])
    except: return None, None

def _chain_is_valid(names: list[str]) -> bool:
    categories = {
        "identity": "none",
        "transpose": "geo", "rot90": "geo", "rot180": "geo", "rot270": "geo",
        "flip_h": "geo", "flip_v": "geo",
        "erode1": "morph", "dilate1": "morph", "erode2": "morph", "dilate2": "morph",
        "adj_mask": "morph", "border_mask": "morph",
        "project_r": "proj", "project_l": "proj", "project_d": "proj", "project_u": "proj",
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
        sx, sy = _parse_shift(names[i])
        nx, ny = _parse_shift(names[i+1])
        if sx is not None and nx is not None and sx+nx==0 and sy+ny==0: return False
    return True

def _state_hash(pred_tensors: list[torch.Tensor], expecteds: list[list[list[int]]]) -> int:
    grids = []
    for pt, exp in zip(pred_tensors, expecteds):
        grids.append(decode_tensor_to_grid(pt.detach().cpu().numpy(), len(exp), len(exp[0])))
    h_raw = hash(str(grids))
    cmap = get_color_normalization_map(grids)
    norm_grids = [apply_color_map(g, cmap) for g in grids]
    h_norm = hash(str(norm_grids))
    return hash((h_raw, h_norm))

