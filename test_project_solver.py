import torch
import torch.nn.functional as F
from torch import nn

class ProjectSolver(nn.Module):
    def __init__(self, direction: str, in_h: int, in_w: int):
        super().__init__()
        self.direction = direction
        self.in_h = in_h
        self.in_w = in_w
        self.grid_size = 30

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        domain_mask = torch.zeros((1, 1, self.grid_size, self.grid_size), device=x.device)
        domain_mask[:, :, :self.in_h, :self.in_w] = 1.0

        for _ in range(max(self.in_h, self.in_w)):
            if self.direction == "right":
                shifted = F.pad(out[:, :, :, :-1], (1, 0, 0, 0))
            elif self.direction == "left":
                shifted = F.pad(out[:, :, :, 1:], (0, 1, 0, 0))
            elif self.direction == "down":
                shifted = F.pad(out[:, :, :-1, :], (0, 0, 1, 0))
            elif self.direction == "up":
                shifted = F.pad(out[:, :, 1:, :], (0, 0, 0, 1))

            bg_mask = (out[:, 0:1, :, :] > 0.5).float()
            fill_colors = shifted[:, 1:, :, :]
            has_color = (fill_colors.sum(dim=1, keepdim=True) > 0.5).float()
            update_mask = bg_mask * has_color * domain_mask

            # Use torch.where to update
            out_bg = torch.where(update_mask > 0.5, torch.zeros_like(out[:, 0:1]), out[:, 0:1])
            # expand update_mask to 9 colors
            expanded_update = update_mask.expand(-1, 9, -1, -1)
            out_colors = torch.where(expanded_update > 0.5, fill_colors, out[:, 1:])
            
            out = torch.cat([out_bg, out_colors], dim=1)

        return out

if __name__ == "__main__":
    from neurogolf.grid_codec import encode_grid_to_tensor, decode_tensor_to_grid
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ]
    t = torch.from_numpy(encode_grid_to_tensor(grid))
    solver = ProjectSolver("right", 3, 5)
    out_t = solver(t)
    out_grid = decode_tensor_to_grid(out_t.numpy(), 3, 5)
    print("Project Right:")
    [print(r) for r in out_grid]

    solver = ProjectSolver("down", 3, 5)
    out_t = solver(t)
    out_grid = decode_tensor_to_grid(out_t.numpy(), 3, 5)
    print("\nProject Down:")
    [print(r) for r in out_grid]

