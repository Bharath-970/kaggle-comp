# NeuroGolf 2026 Compression Engine

This repository implements a cost-minimizing ONNX synthesis pipeline for ARC-AGI tasks.

## Current Implementation Scope

- Task intelligence feature extraction and invariance inference
- Grid encoding and decoding to the required [1, 10, 30, 30] tensor format
- ONNX-native model generation utilities
- Compliance validation (static shape, banned ops, file size)
- Cost estimation (parameters, memory, MACs, score)
- Search scaffolding (routing, minimal lane, equivalence tools)
- Submission gate and submission allocator scaffolding

## Quick Start

1. Create and activate a Python 3.11+ environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run tests:

   pytest -q

## Project Layout

- src/intel: Task analysis and search priors
- src/data: ARC grid and tensor transformations
- src/onnx: Direct ONNX graph construction and compliance checks
- src/search: Search lanes, equivalence tools, and routing
- src/eval: Correctness and cost evaluation
- src/pipeline: Submission decision and pre-submit validation
- scripts: Small entrypoint scripts
- tests: Unit tests for deterministic core behavior

## Notes

- All ONNX builders use static shapes for input and output tensors.
- Banned operators are guarded in compliance checks.
- This is a foundation sprint, not the full competition engine yet.
