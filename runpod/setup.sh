#!/bin/bash
# NeuroGolf 2026 — RunPod Setup Script
# Run this on a fresh RunPod pod (RTX 4090 or A100)
set -e

echo "=== NeuroGolf RunPod Setup ==="

# Install exact competition dependencies
pip install --upgrade pip setuptools wheel && \
pip install --quiet \
    onnx \
    onnxruntime \
    onnx-tool \
    numpy && \
pip install --quiet \
    -i https://download.pytorch.org/whl/cu121 \
    torch torchvision

echo "=== Dependencies installed ==="
python -c "import onnx; import onnxruntime; import onnx_tool; import torch; print('All OK'); print('CUDA:', torch.cuda.is_available())"

echo "=== Setup complete. Now run: ==="
echo "python runpod/synthesizer.py --data_dir data --output_dir output_synth --tasks 1-400"
echo "python src/fix_shapes.py --output_dir output_synth --data_dir data"
