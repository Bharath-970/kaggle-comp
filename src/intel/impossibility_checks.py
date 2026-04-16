from __future__ import annotations

from dataclasses import dataclass

from src.intel.task_intelligence import TaskIntelligence


@dataclass(frozen=True)
class CandidateCapabilities:
    receptive_field: int
    supports_color_mapping: bool
    supports_shape_change: bool
    supports_counting_logic: bool


def early_impossibility_reasons(
    task_intel: TaskIntelligence,
    capabilities: CandidateCapabilities,
) -> list[str]:
    reasons: list[str] = []

    requires_shape_change = any(
        (p.input_height, p.input_width) != (p.output_height, p.output_width)
        for p in task_intel.pair_features
    )
    if requires_shape_change and not capabilities.supports_shape_change:
        reasons.append("candidate cannot express output shape changes")

    if task_intel.inferred_families.get("color_map", 0.0) > 0.35 and not capabilities.supports_color_mapping:
        reasons.append("candidate lacks color mapping capability")

    if task_intel.inferred_families.get("counting", 0.0) > 0.3 and not capabilities.supports_counting_logic:
        reasons.append("candidate lacks counting logic capability")

    # Very small receptive fields fail many composition tasks.
    if task_intel.inferred_families.get("composition", 0.0) > 0.25 and capabilities.receptive_field < 3:
        reasons.append("receptive field too small for composition-heavy task")

    return reasons
