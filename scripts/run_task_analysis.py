from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.intel.task_intelligence import analyze_task, load_task_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ARC task and emit intelligence summary")
    parser.add_argument("--task-id", required=True, help="Task identifier, for example task001")
    parser.add_argument("--task-path", required=True, help="Path to task json file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_task_json(args.task_path)
    intel = analyze_task(args.task_id, payload)
    print(json.dumps(asdict(intel), indent=2))


if __name__ == "__main__":
    main()
