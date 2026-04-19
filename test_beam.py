import numpy as np
from scipy.ndimage import label

def _score_candidate(pred_arr: np.ndarray, expected: list[list[int]]) -> float:
    exp_arr = np.array(expected, dtype=np.int32)
    pixel_score = np.mean(exp_arr == pred_arr)
    _, n_pred = label(pred_arr != 0)
    _, n_exp = label(exp_arr != 0)
    obj_diff = abs(n_pred - n_exp)
    obj_score = 1.0 / (1.0 + obj_diff)
    
    def get_bbox(arr):
        mask = arr != 0
        if not np.any(mask): return 0, 0
        r, c = np.where(mask)
        return r.max()-r.min()+1, c.max()-c.min()+1
    
    ph, pw = get_bbox(pred_arr)
    eh, ew = get_bbox(exp_arr)
    bbox_diff = abs(ph - eh) + abs(pw - ew)
    bbox_score = 1.0 / (1.0 + bbox_diff)
    
    from collections import Counter
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

e = [[0,0],[0,1]]
p1 = [[1,0],[0,0]] # Shifted
p2 = [[1,1],[1,1]] # Garbage

print("Shifted:", _score_candidate(np.array(p1), e))
print("Garbage:", _score_candidate(np.array(p2), e))
