#!/bin/bash
# NeuroGolf 2026 — RunPod Upload Script
# Run locally AFTER pipeline_symbolic.py to upload files to RunPod

set -e
export COPYFILE_DISABLE=1

RUNPOD_IP="${1:?Usage: ./upload_to_runpod.sh <RUNPOD_IP> <RUNPOD_PORT>}"
RUNPOD_PORT="${2:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/workspace/neurogolf"

echo "Uploading to RunPod at $RUNPOD_IP:$RUNPOD_PORT..."

SSH_CMD="ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $RUNPOD_PORT root@$RUNPOD_IP"
USE_RSYNC=0
if command -v rsync >/dev/null 2>&1 && $SSH_CMD "command -v rsync >/dev/null 2>&1"; then
    USE_RSYNC=1
fi

# Create remote directory structure
$SSH_CMD "mkdir -p $REMOTE_ROOT/{data,output,runpod,src}"

upload_dir() {
    local src_dir="$1"
    local dst_dir="$2"
    if [ "$USE_RSYNC" -eq 1 ]; then
        rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $RUNPOD_PORT" \
            "$src_dir" "root@$RUNPOD_IP:$dst_dir"
    else
        # Fallback when rsync is unavailable locally.
        tar -C "$(dirname "$src_dir")" -cf - "$(basename "$src_dir")" | \
            $SSH_CMD "tar -C $dst_dir -xf -"
    fi
}

upload_file_if_exists() {
    local src_file="$1"
    local dst_dir="$2"
    if [ -f "$src_file" ]; then
        if [ "$USE_RSYNC" -eq 1 ]; then
            rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $RUNPOD_PORT" \
                "$src_file" "root@$RUNPOD_IP:$dst_dir"
        else
            tar -C "$(dirname "$src_file")" -cf - "$(basename "$src_file")" | \
                $SSH_CMD "tar -C $dst_dir -xf -"
        fi
    fi
}

# Upload competition data (task JSONs + utils)
echo "Uploading task data..."
upload_dir "data" "$REMOTE_ROOT"

# Upload source
echo "Uploading source..."
upload_dir "src" "$REMOTE_ROOT"
upload_dir "runpod" "$REMOTE_ROOT"

# Upload existing output (ONNX + needs_neural.json)
echo "Uploading existing outputs..."
upload_file_if_exists "output/needs_neural.json" "$REMOTE_ROOT/output/"

echo ""
echo "=== Upload complete! Now on RunPod: ==="
echo "  cd $REMOTE_ROOT"
echo "  bash runpod/setup.sh"
echo "  python runpod/synthesizer.py --tasks 1-400"
echo "  python src/fix_shapes.py --output_dir output_synth --data_dir data"
echo "  cp output/*.onnx output_synth/ 2>/dev/null || true"
echo "  python src/pack_submission.py --output_dir output_synth --submission submission.zip"
echo ""
echo "=== After synthesis, download results: ==="
echo "  rsync -avz -e 'ssh -i $SSH_KEY -p $RUNPOD_PORT' root@$RUNPOD_IP:$REMOTE_ROOT/output_synth/*.onnx output/"
echo "  rsync -avz -e 'ssh -i $SSH_KEY -p $RUNPOD_PORT' root@$RUNPOD_IP:$REMOTE_ROOT/submission.zip ."
