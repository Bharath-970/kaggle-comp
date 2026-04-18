"""Train an ensemble of models for a single ARC task."""

from __future__ import annotations

import torch
from typing import List
from .task_io import TaskData
from .train import train_task_model, TrainConfig
from .backbone import RegisterBackbone
from .ensemble import EnsembleSolver

def train_ensemble_for_task(
    task: TaskData,
    task_id: str,
    n_models: int = 3,
    config: TrainConfig = TrainConfig(),
    backbone_kwargs: dict | None = None,
) -> EnsembleSolver:
    """ Trains N models with different seeds and returns an ensemble. """
    models = []
    if backbone_kwargs is None:
        backbone_kwargs = {}
    
    for i in range(n_models):
        print(f"  Training ensemble branch {i+1}/{n_models}...", end=" ", flush=True)
        # Use a different seed for each branch
        branch_config = TrainConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            arcgen_train_sample=config.arcgen_train_sample,
            seed=config.seed + i,
            use_augmentation=config.use_augmentation,
            batch_size=config.batch_size,
            min_epochs=config.min_epochs,
            eval_interval=config.eval_interval,
            early_stop_patience=config.early_stop_patience,
            early_stop_delta=config.early_stop_delta,
            entropy_patience_bonus=config.entropy_patience_bonus,
            enable_dynamic_early_stop=config.enable_dynamic_early_stop,
        )
        
        model = RegisterBackbone(**backbone_kwargs)
        train_summary = train_task_model(model, task, task_id, branch_config)
        print(f"loss={train_summary.final_loss:.4f}")
        models.append(model)
        
    return EnsembleSolver(models)
