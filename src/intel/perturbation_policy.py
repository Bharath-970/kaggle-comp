from __future__ import annotations

from dataclasses import dataclass

from src.intel.task_intelligence import TaskIntelligence


@dataclass(frozen=True)
class PerturbationPolicy:
    allow_translation: bool
    allow_rotation: bool
    allow_color_permutation: bool
    allow_noise: bool


def from_task_intelligence(task_intel: TaskIntelligence) -> PerturbationPolicy:
    hints = task_intel.invariance_hints
    return PerturbationPolicy(
        allow_translation=bool(hints.get("allow_translation_perturbation", False)),
        allow_rotation=bool(hints.get("allow_rotation_perturbation", False)),
        allow_color_permutation=bool(hints.get("allow_color_permutation", False)),
        allow_noise=bool(hints.get("allow_noise", False)),
    )
