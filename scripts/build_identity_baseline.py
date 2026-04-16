from __future__ import annotations

import argparse

from src.eval.cost import compute_cost
from src.onnx.compliance_guard import validate_model
from src.onnx.native_builder import build_identity_model, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate identity ONNX baseline")
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX path, for example models/task001_identity.onnx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_identity_model()
    path = save_model(model, args.output)

    compliance = validate_model(path)
    cost = compute_cost(path)

    print(f"saved: {path}")
    print(f"compliance_ok: {compliance.ok}")
    if compliance.errors:
        print(f"compliance_errors: {compliance.errors}")
    print(
        "cost: "
        f"params={cost.parameters}, "
        f"memory={cost.memory_bytes}, "
        f"macs={cost.macs}, "
        f"score={cost.score:.4f}"
    )


if __name__ == "__main__":
    main()
