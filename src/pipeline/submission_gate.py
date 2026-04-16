from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.correctness import CorrectnessReport, evaluate_model_on_payload
from src.eval.cost import CostBreakdown, compute_cost
from src.onnx.compliance_guard import ComplianceReport, validate_model


@dataclass
class GateDecision:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    compliance: ComplianceReport | None = None
    correctness: CorrectnessReport | None = None
    cost: CostBreakdown | None = None


def evaluate_candidate(
    model_path: str | Path,
    task_payload: dict[str, Any],
    *,
    max_examples_per_subset: int | None = None,
    stop_on_first_failure: bool = False,
) -> GateDecision:
    path = Path(model_path)
    reasons: list[str] = []

    compliance = validate_model(path)
    if not compliance.ok:
        reasons.extend(compliance.errors)

    correctness = evaluate_model_on_payload(
        path,
        task_payload,
        max_per_subset=max_examples_per_subset,
        stop_on_first_failure=stop_on_first_failure,
    )
    if not correctness.ok:
        reasons.append(
            f"correctness failed on {correctness.total_examples - correctness.passed_examples} examples"
        )

    cost = compute_cost(path)

    return GateDecision(
        ok=len(reasons) == 0,
        reasons=reasons,
        compliance=compliance,
        correctness=correctness,
        cost=cost,
    )
