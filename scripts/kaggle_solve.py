import os
import sys
import time
from pathlib import Path
import multiprocessing as mp

# Setup paths based on environment
# Detect the root directory of this script (which should be inside a 'scripts' folder)
this_script_dir = os.path.dirname(os.path.abspath(__file__))
# If we are in 'scripts/', the project root is one level up
project_root = os.path.abspath(os.path.join(this_script_dir, ".."))

if os.path.exists("/kaggle/input"):
    # On Kaggle, we might be running from /kaggle/working or directly from /kaggle/input
    # We'll set dataset_path to look deep for task.json files
    dataset_path = "/kaggle/input"
else:
    # Local fallback
    dataset_path = "/Users/bharath/Downloads/neurogolf-2026"

# Inject paths
src_dir = os.path.join(project_root, "src")
scripts_dir = os.path.join(project_root, "scripts")
if src_dir not in sys.path: sys.path.insert(0, src_dir)
if scripts_dir not in sys.path: sys.path.insert(0, scripts_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)

# Top level imports
from neurogolf.task_io import load_task_json
from neurogolf.export import export_static_onnx
from search_solvers import find_master_synthesis


def solve_single_task(task_path: str):
    task_id = Path(task_path).stem
    
    try:
        task = load_task_json(task_path)
    except Exception as e:
        return task_id, None, f"File Error: {str(e)}"

    model = find_master_synthesis(task, max_shift=2)
    if not model:
        return task_id, None, "No symbolic solution found"

    model.cpu()
    tmp_path = f"/tmp/{task_id}.onnx"
    export_static_onnx(model, tmp_path)
    with open(tmp_path, "rb") as f:
        bytes_data = f.read()

    return task_id, bytes_data, "Success"


def main():
    path_obj = Path(dataset_path)
    files = sorted(path_obj.glob("task*.json"))
    if not files and os.path.exists("/kaggle/input"):
        # Kaggle specific dataset mounts can have varied names
        print(f"Warning: No files found directly in {dataset_path}. Deep searching...")
        files = list(Path("/kaggle/input").rglob("task*.json"))
            
    files = sorted(files)
    if not files:
        print(f"❌ No tasks found in {dataset_path}!")
        return

    print(f"🚀 [KAGGLE-SWEEP] Starting {len(files)}-Task Sweep using ProcessPool...")
    
    out_dir = Path(project_root) / "artifacts" / "final_submission"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    solved = 0

    existing_tids = {f.stem for f in out_dir.glob("*.onnx")}
    to_process = [str(f) for f in files if f.stem not in existing_tids]
    solved += len(existing_tids)

    if solved > 0:
        print(f"⏩ Resuming: {solved} tasks already solved on disk. Skipping...")

    if not to_process:
        print(f"🎉 All {len(files)} tasks are already solved!")
        return

    # Use max cores available on Kaggle instances (usually 4 cores)
    num_cores = max(1, mp.cpu_count())
    print(f"⚡ Booting {num_cores} parallel workers...")

    # Set up parallel execution
    with mp.Pool(processes=num_cores) as pool:
        for tid, onnx_data, status in pool.imap_unordered(solve_single_task, to_process):
            if onnx_data:
                with open(out_dir / f"{tid}.onnx", "wb") as f:
                    f.write(onnx_data)
                solved += 1
                print(f"✅ {tid}: {status} (Total: {solved}/400)")
            else:
                print(f"❌ {tid}: {status}")

    print(f"\n🏆 Final Results: {solved}/{len(files)} Solved in {int(time.time() - start_time)}s.")


if __name__ == "__main__":
    if sys.platform == "darwin":
        mp.set_start_method('spawn', force=True)
    main()
