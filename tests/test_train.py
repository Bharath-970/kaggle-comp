from neurogolf.backbone import RegisterBackbone
from neurogolf.task_io import GridPair, TaskData
from neurogolf.train import TrainConfig, train_task_model


def test_train_task_model_smoke() -> None:
    pair = GridPair(input_grid=[[1, 0], [0, 1]], output_grid=[[1, 0], [0, 1]])
    task = TaskData(train=(pair,), test=(pair,), arc_gen=())

    model = RegisterBackbone(steps=1, scratch_channels=2, mask_channels=1, phase_channels=1, hidden_channels=8)
    config = TrainConfig(epochs=2, learning_rate=1e-2, batch_size=1, arcgen_train_sample=0)

    result = train_task_model(model=model, task=task, task_id="task999", config=config)

    assert result.task_id == "task999"
    assert result.train_samples == 1
    assert result.epochs == 2
    assert result.best_loss >= 0.0
