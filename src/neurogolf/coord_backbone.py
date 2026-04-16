"""Coordinate-aware and object-centric backbone for high-efficiency ARC."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

from .constants import COLOR_CHANNELS, GRID_SIZE


class CoordBackbone(nn.Module):
    """
    Ultra-efficient backbone that uses coordinates and global priors.
    Designed to beat the 4076 leaderboard score via extreme parameter golf.
    """

    def __init__(
        self,
        in_channels: int = COLOR_CHANNELS,
        hidden_channels: int = 16,
        out_channels: int = COLOR_CHANNELS,
        use_coords: bool = True,
        use_global: bool = True,
    ) -> None:
        super().__init__()
        self.use_coords = use_coords
        self.use_global = use_global
        
        # 2 coordinate channels (x, y)
        actual_in = in_channels + (2 if use_coords else 0)
        
        # 1x1 projection to hidden space
        self.proj = nn.Conv2d(actual_in, hidden_channels, kernel_size=1)
        
        # Depthwise spatial update (minimal MACs)
        self.spatial = nn.Conv2d(
            hidden_channels, 
            hidden_channels, 
            kernel_size=3, 
            padding=1, 
            groups=hidden_channels,
            bias=False
        )
        
        # Global context branch
        if use_global:
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.global_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
            
        # Readout
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def _get_coords(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        # Create normalized coordinates [0, 1]
        y_range = torch.linspace(0.0, 1.0, h, device=x.device)
        x_range = torch.linspace(0.0, 1.0, w, device=x.device)
        
        # Grid [H, W]
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Expand to [Batch, 2, H, W]
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coords:
            coords = self._get_coords(x)
            x = torch.cat([x, coords], dim=1)
            
        feat = F.relu(self.proj(x))
        
        # Local update
        feat = feat + self.spatial(feat)
        
        # Global update
        if self.use_global:
            g = self.global_pool(feat)
            g = torch.sigmoid(self.global_proj(g))
            feat = feat * g # Gated global interaction
            
        return self.readout(feat)

def count_golf_cost(model: nn.Module) -> dict[str, int]:
    params = sum(p.numel() for p in model.parameters())
    # Approximation for 30x30 grid
    # This is just a helper for our search logic
    return {"params": params}
