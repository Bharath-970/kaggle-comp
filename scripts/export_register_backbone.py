#!/usr/bin/env python3
"""CLI for exporting a channel-register backbone ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.export import build_and_export_register_backbone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export register backbone to ONNX")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--scratch-channels", type=int, default=8)
    parser.add_argument("--mask-channels", type=int, default=2)
    parser.add_argument("--phase-channels", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--opset", type=int, default=18)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = build_and_export_register_backbone(
        output_path=args.output,
        steps=args.steps,
        scratch_channels=args.scratch_channels,
        mask_channels=args.mask_channels,
        phase_channels=args.phase_channels,
        hidden_channels=args.hidden_channels,
        opset=args.opset,
    )

    print(f"Exported: {args.output}")
    if report is None:
        print("Validation skipped")
        return

    print(f"File size: {report.file_size_bytes}")
    print(f"Banned ops: {list(report.banned_ops)}")
    print(f"Dynamic tensors: {list(report.dynamic_tensors)}")
    print(f"External data files: {list(report.external_data_files)}")
    print(f"Valid: {report.is_valid}")


if __name__ == "__main__":
    main()
