import os
import sys
from pathlib import Path
import modal

# project-specific absolute paths
project_root = "/Users/bharath/Documents/kaggle comp"
dataset_path = "/Users/bharath/Downloads/neurogolf-2026"

# Create the App
app = modal.App("neurogolf_sweep_4500")

# Define the Image - DEEP BAKE ALL CODE AND DATA INTO THE IMAGE
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "onnx", "onnxruntime")
    .add_local_dir(os.path.join(project_root, "src"), remote_path="/root/src")
    .add_local_dir(os.path.join(project_root, "scripts"), remote_path="/root/scripts")
    .add_local_dir(dataset_path, remote_path="/root/tasks")
)

@app.function(
    image=image,
    timeout=900,
    cpu=2.0
)
def solve_single_task(task_path: str):
    import sys
    sys.path.append("/root/src")
    sys.path.append("/root")
    
    from neurogolf.task_io import load_task_json
    from neurogolf.export import export_static_onnx
    import torch
    import io
    
    task_id = Path(task_path).stem
    # Heartbeat print for real-time monitoring
    print(f"[{task_id}] Container Started. Loading data...")
    
    try:
        task = load_task_json(task_path)
    except Exception as e:
        return task_id, None, f"File Error: {str(e)}"
    
    from scripts.search_solvers import find_master_synthesis, check_solve
    from neurogolf.train_ensemble import train_ensemble_for_task
    from neurogolf.train import TrainConfig
    from neurogolf.grid_codec import get_color_normalization_map
    from neurogolf.solvers import ColorNormalizedSolver

    # 1. Faster Symbolic Check
    model = find_master_synthesis(task, max_shift=2)
    if not model:
        print(f"[{task_id}] Symbolic Failed. Starting 5,000-Epoch Fallback (ETA: 4-6 mins)...")
        # Fallback to high-intensity neural ensemble
        ensemble_backbone = train_ensemble_for_task(
            task, task_id, n_models=3,
            config=TrainConfig(epochs=5000, learning_rate=4e-3),
            backbone_kwargs={"hidden_channels": 12, "steps": 6, "use_coords": True, "use_depthwise": True}
        )
        # Derive unified map from ALL task inputs
        all_inputs = [p.input_grid for p in task.train]
        all_inputs.extend([p.input_grid for p in task.test])
        all_inputs.extend([p.input_grid for p in task.arc_gen])
        cmap = get_color_normalization_map(all_inputs)

        model = ColorNormalizedSolver(ensemble_backbone, cmap)
        if not check_solve(model, task):
            return task_id, None, "Solved but Verification Failed"

    model.cpu()
    tmp_path = f"/tmp/{task_id}.onnx"
    export_static_onnx(model, tmp_path)
    with open(tmp_path, "rb") as f:
        bytes_data = f.read()
    
    print(f"[{task_id}] SUCCESS. Exporting model...")
    return task_id, bytes_data, "Success"

@app.local_entrypoint()
def main():
    import time
    dataset_path = "/Users/bharath/Downloads/neurogolf-2026"
    files = sorted(Path(dataset_path).glob("task*.json"))
    remote_paths = [f"/root/tasks/{f.name}" for f in files]
    
    print(f"🚀 [ROBUST-SWEEP] Starting 400-Task Sweep on Modal...")
    print(f"💡 Note: Normalization is now Task-Global to prevent Verification Failures.")
    
    out_dir = Path(project_root) / "artifacts" / "final_submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    solved = 0
    
    # RESUME LOGIC: Skip tasks that already have a model on disk
    existing_tids = {f.stem for f in out_dir.glob("*.onnx")}
    to_process = []
    for f, remote in zip(files, remote_paths):
        if f.stem not in existing_tids:
            to_process.append(remote)
        else:
            solved += 1 # Count already solved for the tally
    
    if solved > 0:
        print(f"⏩ Resuming: {solved} tasks already solved. Skipping...")
    
    if not to_process:
        print(f"🎉 All {len(files)} tasks are already solved!")
        return

    # Iterate dynamically to save results the second they finish (UNORDERED for speed)
    for tid, onnx_data, status in solve_single_task.map(to_process, order_outputs=False):
        if onnx_data:
            with open(out_dir / f"{tid}.onnx", "wb") as f:
                f.write(onnx_data)
            solved += 1
            print(f"✅ {tid}: {status} (Total: {solved}/400)")
        else:
            print(f"❌ {tid}: {status}")
            
    print(f"\n🏆 Final Results: {solved}/400 Solved in {int(time.time() - start_time)}s.")

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            main()
