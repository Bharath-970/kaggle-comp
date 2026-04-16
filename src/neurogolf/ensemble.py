"""ONNX-friendly ensembling of multiple model branches."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


class EnsembleSolver(nn.Module):
    """
    Combines multiple models into a single ONNX graph.
    Performs 'Mean' ensembling on probabilities (logits).
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum the outputs of all branches
        out = self.branches[0](x)
        for i in range(1, len(self.branches)):
            out = out + self.branches[i](x)
        
        # We don't need to divide by N because Argmax(Sum) == Argmax(Mean)
        return out
