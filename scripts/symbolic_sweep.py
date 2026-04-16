#!/usr/bin/env python3
"""Fast symbolic-only sweep to estimate deterministic coverage before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from search_solvers import find_master_synthesis, load_task_json_relaxed, check_solve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/Users/bharath/Downloads/neurogolf-2026")
    parser.add_argument("--max-shift", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default="artifacts/eval/symbolic_sweep.json")
    args = parser.parse_args()

    files = sorted(Path(args.dataset_root).glob("task*.json"))
    if args.limit is not None:
        files = files[: args.limit]

    started = time.time()
    solved = 0
    skipped = 0
    rows: list[dict[str, object]] = []

    for idx, task_file in enumerate(files, start=1):
        task_id = task_file.stem
        try:
            task, dropped = load_task_json_relaxed(task_file)
        except ValueError as exc:
            skipped += 1
            rows.append({"task_id": task_id, "status": "skipped", "reason": str(exc)})
            continue

        model = find_master_synthesis(task, max_shift=args.max_shift)
        ok = False
        if model is not None:
            ok = check_solve(model, task)

        if ok:
            solved += 1
            status = "solved"
        else:
            status = "failed"

        rows.append(
            {
                "task_id": task_id,
                "status": status,
                "dropped_arcgen": dropped,
            }
        )

        if idx % 25 == 0 or idx == len(files):
            print(f"Processed {idx}/{len(files)} | solved={solved} | skipped={skipped}", flush=True)

    payload = {
        "dataset_root": args.dataset_root,
        "max_shift": args.max_shift,
        "limit": args.limit,
        "summary": {
            "total": len(files),
            "solved": solved,
            "skipped": skipped,
            "failed": len(files) - solved - skipped,
            "elapsed_s": round(time.time() - started, 3),
        },
        "tasks": rows,
    }

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print("---")
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
