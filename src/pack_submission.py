"""
Pack all output/*.onnx into submission.zip for Kaggle.
Also prints final estimated score.

Usage:
    python src/pack_submission.py [--output_dir output] [--submission submission.zip]
"""
import argparse
import json
import math
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "data/neurogolf_utils")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="output")
    parser.add_argument("--submission",  default="submission.zip")
    parser.add_argument("--data_dir",    default="data")
    args = parser.parse_args()

    try:
        import neurogolf_utils as ng
        HAS_NG = True
    except ImportError:
        HAS_NG = False
        print("[WARNING] neurogolf_utils not found — skipping score estimation")

    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob("task*.onnx"))

    if not files:
        print(f"No ONNX files found in {output_dir}/")
        return

    total_score = 0.0
    task_scores = {}
    bad_tasks = []

    print(f"\nScoring {len(files)} ONNX files...")
    print(f"{'Task':<10} {'Score':>6} {'MACs':>12} {'Memory':>10} {'Params':>8}")
    print("-" * 52)

    for fpath in files:
        task_name = fpath.stem  # e.g. "task001"
        task_num = int(task_name.replace("task", ""))

        if HAS_NG:
            try:
                macs, mem, params = ng.score_network(str(fpath))
                if macs is None:
                    bad_tasks.append(task_num)
                    print(f"{task_name:<10} {'SHAPE_ERR':>6}")
                    continue
                cost = macs + mem + params
                sc = max(1.0, 25.0 - math.log(max(1.0, cost)))
                task_scores[task_num] = sc
                total_score += sc
                print(f"{task_name:<10} {sc:>6.2f} {macs:>12,} {mem:>10,} {params:>8,}")
            except Exception as e:
                bad_tasks.append(task_num)
                print(f"{task_name:<10} ERROR: {e}")
        else:
            task_scores[task_num] = 0

    print("-" * 52)
    print(f"\n{'Total tasks:':<20} {len(files)}")
    print(f"{'Valid (scored):':<20} {len(task_scores)}")
    print(f"{'Shape errors:':<20} {len(bad_tasks)}")
    print(f"{'Estimated score:':<20} {total_score:.1f}")
    print(f"{'Target:':<20} 7000")
    print(f"{'Gap:':<20} {7000 - total_score:.1f}")

    if bad_tasks:
        print(f"\n[WARNING] {len(bad_tasks)} tasks have shape errors (not scored):")
        print(bad_tasks)

    # Score distribution
    print("\nScore distribution:")
    for thresh in [23, 21, 19, 17, 15, 12, 10, 5, 1]:
        count = sum(1 for s in task_scores.values() if s >= thresh)
        print(f"  >= {thresh:2d}: {count:3d} tasks")

    # Create submission.zip
    submission_path = args.submission
    with zipfile.ZipFile(submission_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            # Check file size limit: 1.44 MB
            size = fpath.stat().st_size
            if size > 1.44 * 1024 * 1024:
                print(f"[WARNING] {fpath.name} exceeds 1.44MB limit ({size:,} bytes) — EXCLUDED")
                continue
            zf.write(fpath, fpath.name)

    zip_size = Path(submission_path).stat().st_size
    print(f"\nSubmission: {submission_path} ({zip_size/1024:.1f} KB)")
    print(f"Files included: {len(files)}")

    # Save score breakdown
    score_path = output_dir / "score_breakdown.json"
    with open(score_path, "w") as f:
        json.dump({
            "total": total_score,
            "tasks": task_scores,
            "bad_tasks": bad_tasks,
            "n_files": len(files),
        }, f, indent=2)
    print(f"Score breakdown saved to: {score_path}")


if __name__ == "__main__":
    main()
