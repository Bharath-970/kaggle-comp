#!/bin/bash
# Run NeuroGolf synthesis pipeline on RunPod over SSH.

set -euo pipefail

RUNPOD_IP="${1:?Usage: ./runpod/run_remote_pipeline.sh <RUNPOD_IP> <RUNPOD_PORT> [TASK_SPEC...]}"
RUNPOD_PORT="${2:-22}"
shift 2 || true

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/workspace/neurogolf"
TASK_ARGS="$*"
if [ -z "$TASK_ARGS" ]; then
  TASK_ARGS="1-400"
fi

ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$RUNPOD_PORT" "root@$RUNPOD_IP" \
  "cd $REMOTE_ROOT && \
   bash runpod/setup.sh && \
   python runpod/synthesizer.py --data_dir data --output_dir output_synth --tasks $TASK_ARGS && \
   python src/fix_shapes.py --output_dir output_synth --data_dir data && \
   cp output/*.onnx output_synth/ 2>/dev/null || true && \
   python src/pack_submission.py --output_dir output_synth --submission submission.zip"

echo "RunPod pipeline completed."
echo "Download:"
echo "  rsync -avz -e \"ssh -i $SSH_KEY -p $RUNPOD_PORT\" root@$RUNPOD_IP:$REMOTE_ROOT/output_synth/*.onnx output/"
echo "  rsync -avz -e \"ssh -i $SSH_KEY -p $RUNPOD_PORT\" root@$RUNPOD_IP:$REMOTE_ROOT/submission.zip ."
