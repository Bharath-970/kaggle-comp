"""Scoring and cost utilities for NeuroGolf."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    """Computational cost components for one task model."""

    parameters: int
    memory_bytes: int
    macs: int

    @property
    def total_cost(self) -> int:
        return self.parameters + self.memory_bytes + self.macs


def score_from_cost(cost: float) -> float:
    """Compute per-task score from cost using competition formula."""
    if cost <= 0:
        raise ValueError("Cost must be positive.")
    return max(1.0, 25.0 - math.log(cost))


def score_from_cost_breakdown(cost: CostBreakdown) -> float:
    return score_from_cost(cost.total_cost)


def max_cost_for_score(target_score: float) -> float:
    """Inverse score -> cost mapping for scores above floor."""
    if target_score <= 1.0:
        raise ValueError("target_score must be greater than 1.0.")
    return math.exp(25.0 - target_score)


def average_score_needed(total_target_score: float, solved_tasks: int) -> float:
    if solved_tasks <= 0:
        raise ValueError("solved_tasks must be positive.")
    return total_target_score / solved_tasks
