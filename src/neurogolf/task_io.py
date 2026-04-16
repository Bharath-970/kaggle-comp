"""Load ARC task JSON files with lightweight validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .grid_codec import encode_grid_to_tensor


@dataclass(frozen=True)
class GridPair:
    input_grid: list[list[int]]
    output_grid: list[list[int]]


@dataclass(frozen=True)
class TaskData:
    train: tuple[GridPair, ...]
    test: tuple[GridPair, ...]
    arc_gen: tuple[GridPair, ...]


def _parse_pairs(raw_pairs: list[dict]) -> tuple[GridPair, ...]:
    parsed: list[GridPair] = []
    for raw in raw_pairs:
        input_grid = raw["input"]
        output_grid = raw["output"]

        # Validate by attempting encoding against fixed competition tensor shape.
        encode_grid_to_tensor(input_grid)
        encode_grid_to_tensor(output_grid)

        parsed.append(GridPair(input_grid=input_grid, output_grid=output_grid))

    return tuple(parsed)


def load_task_json(task_path: str | Path) -> TaskData:
    path = Path(task_path)
    payload = json.loads(path.read_text())

    train_pairs = _parse_pairs(payload.get("train", []))
    test_pairs = _parse_pairs(payload.get("test", []))
    arc_gen_pairs = _parse_pairs(payload.get("arc-gen", []))

    return TaskData(train=train_pairs, test=test_pairs, arc_gen=arc_gen_pairs)
