from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

SEMANTIC_LABELS = (
    "free",
    "mask",
    "predicate",
    "counter",
    "gate",
    "object_id",
    "scratch",
)


@dataclass
class ChannelPrior:
    channel_index: int
    weights: Dict[str, float]


def initialize_soft_priors(num_channels: int) -> list[ChannelPrior]:
    priors: list[ChannelPrior] = []
    for idx in range(num_channels):
        weights = {label: 0.0 for label in SEMANTIC_LABELS}
        weights["free"] = 1.0
        priors.append(ChannelPrior(channel_index=idx, weights=weights))
    return priors


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        return {label: 1.0 / len(weights) for label in weights}
    return {label: max(0.0, value) / total for label, value in weights.items()}


def apply_semantic_drift(
    priors: list[ChannelPrior],
    *,
    channel_index: int,
    target_label: str,
    strength: float = 0.15,
) -> list[ChannelPrior]:
    if target_label not in SEMANTIC_LABELS:
        raise ValueError(f"Unknown semantic label: {target_label}")

    if channel_index < 0 or channel_index >= len(priors):
        raise IndexError("channel_index out of range")

    strength = min(max(strength, 0.0), 1.0)
    prior = priors[channel_index]

    updated = dict(prior.weights)
    for label in updated:
        updated[label] *= (1.0 - strength)
    updated[target_label] += strength
    prior.weights = normalize_weights(updated)
    return priors
