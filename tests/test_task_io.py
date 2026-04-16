import json

from neurogolf.task_io import load_task_json


def test_load_task_json(tmp_path) -> None:
    task_payload = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3, 3]], "output": [[4, 4]]}],
        "arc-gen": [{"input": [[5]], "output": [[6]]}],
    }

    task_path = tmp_path / "task001.json"
    task_path.write_text(json.dumps(task_payload))

    task = load_task_json(task_path)

    assert len(task.train) == 1
    assert len(task.test) == 1
    assert len(task.arc_gen) == 1
    assert task.train[0].input_grid == [[1]]
    assert task.train[0].output_grid == [[2]]
