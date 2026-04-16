import numpy as np

from neurogolf.grid_codec import decode_tensor_to_grid, encode_grid_to_tensor


def test_encode_grid_to_tensor_shape_and_values() -> None:
    grid = [[1, 2], [3, 4]]
    tensor = encode_grid_to_tensor(grid)

    assert tensor.shape == (1, 10, 30, 30)
    assert np.isclose(tensor[0, 1, 0, 0], 1.0)
    assert np.isclose(tensor[0, 2, 0, 1], 1.0)
    assert np.isclose(tensor[0, 3, 1, 0], 1.0)
    assert np.isclose(tensor[0, 4, 1, 1], 1.0)


def test_decode_tensor_to_grid_round_trip() -> None:
    grid = [[9, 0], [5, 1]]
    tensor = encode_grid_to_tensor(grid)
    decoded = decode_tensor_to_grid(tensor, output_height=2, output_width=2)
    assert decoded == grid
