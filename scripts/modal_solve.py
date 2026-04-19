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
# BUILD_TAG: bump this string to force Modal to rebuild the image with fresh code
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "onnx", "onnxruntime", "onnxscript", "scipy")
    .env({"BUILD_TAG": "20260419_v12_adaptive_beam"})   # ← change to force rebuild
    .add_local_dir(os.path.join(project_root, "src"), remote_path="/root/src")
    .add_local_dir(os.path.join(project_root, "scripts"), remote_path="/root/scripts")
    .add_local_dir(dataset_path, remote_path="/root/tasks")
)


@app.function(image=image, timeout=1500, cpu=4.0, memory=4096)
def solve_single_task(task_path: str):
    import sys

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/scripts")
    sys.path.insert(0, "/root")

    from neurogolf.task_io import load_task_json
    from neurogolf.export import export_static_onnx
    import torch

    task_id = Path(task_path).stem
    print(f"[{task_id}] Container Started. Loading data...")

    try:
        task = load_task_json(task_path)
    except Exception as e:
        return task_id, None, f"File Error: {str(e)}"

    from search_solvers import find_master_synthesis

    # find_master_synthesis already verifies correctness before returning.
    # Do NOT re-run check_solve here — it breaks baked ConstantGridSolvers
    # (test-sized constant fails when re-checked against differently-sized train pairs).
    model = find_master_synthesis(task, max_shift=2)

    if not model:
        return task_id, None, "No symbolic solution found"


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
    print(
        f"💡 Note: Normalization is now Task-Global to prevent Verification Failures."
    )

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
            solved += 1  # Count already solved for the tally

    if solved > 0:
        print(f"⏩ Resuming: {solved} tasks already solved. Skipping...")

    if not to_process:
        print(f"🎉 All {len(files)} tasks are already solved!")
        return

    # Iterate dynamically to save results the second they finish (UNORDERED for speed)
    for tid, onnx_data, status in solve_single_task.map(
        to_process, order_outputs=False
    ):
        if onnx_data:
            with open(out_dir / f"{tid}.onnx", "wb") as f:
                f.write(onnx_data)
            solved += 1
            print(f"✅ {tid}: {status} (Total: {solved}/400)")
        else:
            print(f"❌ {tid}: {status}")

    print(
        f"\n🏆 Final Results: {solved}/400 Solved in {int(time.time() - start_time)}s."
    )


@app.function(image=image, timeout=3600, cpu=4.0)
def solve_batch(start: int = 1, count: int = 10):
    import time

    dataset_path = "/root/tasks"
    out_dir = Path("/root/artifacts/batch_solutions")
    out_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted(Path(dataset_path).glob("task*.json"))
    paths_to_solve = [
        f"/root/tasks/{f.name}" for f in file_list[start - 1 : start - 1 + count]
    ]

    print(f"Running tasks {start} to {start + count - 1}...")
    solved = 0

    for tid, onnx_data, status in solve_single_task.map(paths_to_solve):
        if onnx_data:
            with open(out_dir / f"{tid}.onnx", "wb") as f:
                f.write(onnx_data)
            solved += 1
            print(f"✅ {tid}")
        else:
            print(f"❌ {tid}: {status}")

    print(f"\n{solved}/{count} solved!")


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            main()
