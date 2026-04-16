from __future__ import annotations

import math
from typing import Mapping


def rank_families(
    priors: Mapping[str, float],
    *,
    uncertainty: Mapping[str, float] | None = None,
    top_k: int = 3,
    exploration_floor: float = 0.05,
) -> list[tuple[str, float]]:
    if not priors:
        return []

    uncertainty = uncertainty or {}

    scored: dict[str, float] = {}
    for family, prior_value in priors.items():
        uncertainty_boost = 1.0 + max(0.0, uncertainty.get(family, 0.0))
        score = max(exploration_floor, prior_value) * uncertainty_boost
        scored[family] = score

    total = sum(scored.values())
    normalized = {k: v / total for k, v in scored.items()} if total > 0 else scored

    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    return ranked[: max(1, top_k)]


def allocate_trials(
    ranked_families: list[tuple[str, float]],
    *,
    total_trials: int,
    minimum_per_family: int = 1,
) -> dict[str, int]:
    if total_trials <= 0 or not ranked_families:
        return {}

    allocations = {family: minimum_per_family for family, _ in ranked_families}
    remaining = max(0, total_trials - minimum_per_family * len(ranked_families))

    if remaining == 0:
        return allocations

    scores = [score for _, score in ranked_families]
    score_sum = sum(scores)
    if score_sum == 0:
        for family, _ in ranked_families:
            allocations[family] += remaining // len(ranked_families)
        return allocations

    for family, score in ranked_families:
        extra = math.floor(remaining * (score / score_sum))
        allocations[family] += extra

    allocated = sum(allocations.values())
    while allocated < total_trials:
        for family, _ in ranked_families:
            allocations[family] += 1
            allocated += 1
            if allocated == total_trials:
                break

    return allocations
