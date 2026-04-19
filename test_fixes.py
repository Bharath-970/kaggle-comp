import numpy as np

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

grid1 = [[0, 0, 0], [0, 1, 1], [0, 0, 0]]
grid2 = [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
print(_translate_to_origin(grid1))
print(_translate_to_origin(grid2))
assert _translate_to_origin(grid1) == _translate_to_origin(grid2)

def best_alignment_score(pred_arr: np.ndarray, expected_arr: np.ndarray) -> float:
    h, w = expected_arr.shape
    best_align = 0.0
    for dy in range(-h+1, h):
        for dx in range(-w+1, w):
            shifted = np.roll(pred_arr, dy, axis=0)
            shifted = np.roll(shifted, dx, axis=1)
            if dy > 0: shifted[:dy, :] = 0
            elif dy < 0: shifted[dy:, :] = 0
            if dx > 0: shifted[:, :dx] = 0
            elif dx < 0: shifted[:, dx:] = 0
            align = np.mean(expected_arr == shifted)
            if align > best_align:
                best_align = align
    return best_align

pred = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
expected = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
print("Align score:", best_alignment_score(pred, expected))
