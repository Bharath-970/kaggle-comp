import torch

from neurogolf.grid_codec import decode_tensor_to_grid, encode_grid_to_tensor
from neurogolf.solvers import ColorNormalizedSolver, GeneralColorRemapSolver, IdentitySolver, ShiftSolver


def _run_solver(model: torch.nn.Module, grid: list[list[int]]) -> list[list[int]]:
    in_tensor = torch.from_numpy(encode_grid_to_tensor(grid))
    out_tensor = model(in_tensor).detach().cpu().numpy()
    return decode_tensor_to_grid(out_tensor, output_height=len(grid), output_width=len(grid[0]))


def test_general_color_remap_solver_uses_input_to_output_map() -> None:
    grid = [[1, 2], [2, 1]]
    swap_1_2 = [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]
    solver = GeneralColorRemapSolver(swap_1_2)

    assert _run_solver(solver, grid) == [[2, 1], [1, 2]]


def test_color_normalized_solver_round_trips_identity_backbone() -> None:
    grid = [[1, 2], [3, 4]]
    color_map = [0, 3, 1, 4, 2, 5, 6, 7, 8, 9]
    solver = ColorNormalizedSolver(IdentitySolver(), color_map=color_map)

    assert _run_solver(solver, grid) == grid


def test_shift_solver_zero_pads_instead_of_wrapping() -> None:
    grid = [[1, 0], [0, 0]]
    solver = ShiftSolver(dx=1, dy=0)

    assert _run_solver(solver, grid) == [[0, 1], [0, 0]]
