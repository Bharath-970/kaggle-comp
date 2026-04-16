import json

from neurogolf.evaluate import evaluate_dataset, evaluate_task
from neurogolf.task_io import GridPair, TaskData


class EchoModel:
    def eval(self):
        return self

    def __call__(self, tensor):
        # Return identical logits to input one-hot tensor.
        return tensor


def test_evaluate_task_exact_for_identity_pairs() -> None:
    pair = GridPair(input_grid=[[1, 2], [3, 4]], output_grid=[[1, 2], [3, 4]])
    task = TaskData(train=(pair,), test=(pair,), arc_gen=(pair,))

    metrics = evaluate_task(model=EchoModel(), task_id="task999", task_data=task)

    assert metrics.solved is True
    assert metrics.train.exact_pairs == 1
    assert metrics.test.exact_pairs == 1
    assert metrics.arc_gen.exact_pairs == 1


def test_evaluate_dataset_skips_oversized_tasks(tmp_path) -> None:
    valid = {
        "train": [{"input": [[1]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[2]]}],
        "arc-gen": [{"input": [[3]], "output": [[3]]}],
    }

    oversized_grid = [[0] * 31 for _ in range(1)]
    invalid = {
        "train": [{"input": oversized_grid, "output": oversized_grid}],
        "test": [],
        "arc-gen": [],
    }

    (tmp_path / "task001.json").write_text(json.dumps(valid))
    (tmp_path / "task002.json").write_text(json.dumps(invalid))

    summary, results, skipped = evaluate_dataset(model=EchoModel(), dataset_root=tmp_path, verbose=False)

    assert summary.total_task_files == 2
    assert summary.evaluated_tasks == 1
    assert summary.skipped_tasks == 1
    assert summary.solved_tasks == 1
    assert len(results) == 1
    assert len(skipped) == 1


def test_evaluate_task_requires_all_splits_by_default() -> None:
    train_pair = GridPair(input_grid=[[1]], output_grid=[[2]])  # EchoModel will fail this.
    test_pair = GridPair(input_grid=[[3]], output_grid=[[3]])
    arc_pair = GridPair(input_grid=[[4]], output_grid=[[4]])
    task = TaskData(train=(train_pair,), test=(test_pair,), arc_gen=(arc_pair,))

    strict_metrics = evaluate_task(model=EchoModel(), task_id="task998", task_data=task)
    assert strict_metrics.solved is False

    test_only_metrics = evaluate_task(
        model=EchoModel(),
        task_id="task998",
        task_data=task,
        require_all_splits=False,
    )
    assert test_only_metrics.solved is True
