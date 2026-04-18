"""Local training helpers for RegisterBackbone on ARC tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Sequence

import numpy as np

from .grid_codec import encode_grid_to_tensor
from .task_io import GridPair, TaskData, load_task_json

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    F = None


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    arcgen_train_sample: int = 32
    batch_size: int = 16
    seed: int = 42
    use_augmentation: bool = True
    min_epochs: int = 24
    eval_interval: int = 8
    early_stop_patience: int = 20
    early_stop_delta: float = 5e-4
    entropy_patience_bonus: int = 4
    enable_dynamic_early_stop: bool = True


@dataclass(frozen=True)
class TaskTrainSummary:
    task_id: str
    train_samples: int
    final_loss: float
    best_loss: float
    epochs: int


@dataclass(frozen=True)
class SliceRunSummary:
    total_task_files: int
    selected_task_files: int
    trained_tasks: int
    skipped_tasks: int



def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for training.")


def _pick_device() -> "torch.device":
    _require_torch()
    # For small ARC models (30x30), CPU is faster and avoids synchronization bugs.
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _augment_grid(grid: list[list[int]], k: int, flip: bool) -> list[list[int]]:
    """Rotate k*90 degrees and optionally flip."""
    arr = np.array(grid)
    if k > 0:
        arr = np.rot90(arr, k=k)
    if flip:
        arr = np.fliplr(arr)
    return arr.tolist()


def _build_train_pairs(task: TaskData, config: TrainConfig) -> list[GridPair]:
    pairs = list(task.train)
    
    if config.use_augmentation:
        augmented = []
        for pair in pairs:
            # Add 4 rotations x 2 flips = 8 symmetries
            for k in range(4):
                for flip in [False, True]:
                    if k == 0 and not flip:
                        continue # Original is already in 'pairs'
                    augmented.append(GridPair(
                        input_grid=_augment_grid(pair.input_grid, k, flip),
                        output_grid=_augment_grid(pair.output_grid, k, flip)
                    ))
        pairs.extend(augmented)

    if config.arcgen_train_sample > 0 and len(task.arc_gen) > 0:
        rnd = random.Random(config.seed)
        sample_size = min(config.arcgen_train_sample, len(task.arc_gen))
        pairs.extend(rnd.sample(list(task.arc_gen), k=sample_size))
    return pairs


def _pairs_to_tensors(pairs: Sequence[GridPair], device: "torch.device") -> tuple["torch.Tensor", "torch.Tensor"]:
    inputs = []
    targets = []
    for pair in pairs:
        in_tensor = encode_grid_to_tensor(pair.input_grid)
        out_tensor = encode_grid_to_tensor(pair.output_grid)

        inputs.append(in_tensor[0])
        # Cross-entropy expects integer class labels per pixel.
        targets.append(np.argmax(out_tensor[0], axis=0).astype(np.int64))

    x = torch.from_numpy(np.stack(inputs)).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(np.stack(targets)).to(device=device, dtype=torch.long)
    return x, y


from .grid_codec import (
    encode_grid_to_tensor, decode_tensor_to_grid,
    get_color_normalization_map, apply_color_map
)

def _apply_augmentation(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies random symmetry on the fly. Color is handled by normalization."""
    # Symmetries (Rot 90s + Flip)
    k = random.randint(0, 3)
    flip = random.random() > 0.5
    
    x = torch.rot90(x, k=k, dims=(2, 3))
    y = torch.rot90(y, k=k, dims=(1, 2))
    if flip:
        x = torch.flip(x, dims=(3,))
        y = torch.flip(y, dims=(2,))
        
    return x, y


def train_task_model(
    model: "torch.nn.Module",
    task: TaskData,
    task_id: str,
    config: TrainConfig,
    device: "torch.device" | None = None,
) -> TaskTrainSummary:
    _require_torch()
    _seed_everything(config.seed)

    if device is None:
        device = _pick_device()

    # Derive a GLOBAL color map for this entire task
    all_inputs = [p.input_grid for p in task.train]
    # We also include test inputs to ensure the test colors are in the map
    all_inputs.extend([p.input_grid for p in task.test])
    global_cmap = get_color_normalization_map(all_inputs)
    setattr(model, "neurogolf_color_map", global_cmap)

    # Normalize all training pairs using the same GLOBAL map
    norm_pairs = []
    for pair in task.train:
        norm_in = apply_color_map(pair.input_grid, global_cmap)
        norm_out = apply_color_map(pair.output_grid, global_cmap)
        from .task_io import GridPair
        norm_pairs.append(GridPair(input_grid=norm_in, output_grid=norm_out))

    model.to(device)
    model.train()

    # Pre-encode normalized tensors
    base_x, base_y = _pairs_to_tensors(norm_pairs, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    n = base_x.shape[0]
    best_loss = float("inf")
    best_exact = 0.0
    best_entropy = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    final_loss = float("inf")
    patience_left = max(1, int(config.early_stop_patience))
    recent_losses: list[float] = []
    completed_epochs = 0

    for epoch in range(config.epochs):
        completed_epochs = epoch + 1
        # Each epoch, we create a fresh augmented batch
        # We augment every single training sample
        xb, yb = _apply_augmentation(base_x, base_y)
        
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        
        # Add L1 regularization for sparsity (Golf requirement)
        l1_reg = torch.tensor(0.0, device=device)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        loss = loss + 1e-5 * l1_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(base_x)
            eval_loss = F.cross_entropy(eval_logits, base_y)
            pred = torch.argmax(eval_logits, dim=1)
            pair_exact = float((pred == base_y).view(pred.shape[0], -1).all(dim=1).float().mean().item())
            probs = torch.softmax(eval_logits, dim=1).clamp_min(1e-8)
            entropy = float((-(probs * probs.log()).sum(dim=1).mean()).item())

        epoch_loss = float(eval_loss.item())
        final_loss = epoch_loss
        recent_losses.append(epoch_loss)

        improved_loss = epoch_loss + config.early_stop_delta < best_loss
        improved_exact = pair_exact > best_exact + 1e-6
        entropy_not_worse = entropy <= best_entropy + 1e-3

        if improved_loss or improved_exact:
            best_loss = min(best_loss, epoch_loss)
            best_exact = max(best_exact, pair_exact)
            if entropy_not_worse:
                best_entropy = min(best_entropy, entropy)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            if config.enable_dynamic_early_stop:
                patience_left = config.early_stop_patience
                if pair_exact >= 0.75 and entropy_not_worse:
                    patience_left += config.entropy_patience_bonus
        elif config.enable_dynamic_early_stop and epoch + 1 >= config.min_epochs:
            patience_left -= 1

        if pair_exact >= 1.0 and epoch + 1 >= max(8, config.min_epochs // 2):
            break

        if (
            config.enable_dynamic_early_stop
            and epoch + 1 >= config.min_epochs
            and (epoch + 1) % max(1, config.eval_interval) == 0
        ):
            plateau = len(recent_losses) >= 3 and (
                max(recent_losses[-3:]) - min(recent_losses[-3:]) < config.early_stop_delta
            )
            if patience_left <= 0 or (plateau and pair_exact >= 0.95):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if not np.isfinite(best_loss):
        best_loss = final_loss

    return TaskTrainSummary(
        task_id=task_id,
        train_samples=n,
        final_loss=final_loss,
        best_loss=best_loss,
        epochs=completed_epochs,
    )


def iter_task_files(dataset_root: str | Path) -> list[Path]:
    return sorted(Path(dataset_root).glob("task*.json"))


def select_task_files(task_files: list[Path], start_index: int, end_index: int | None) -> list[Path]:
    if start_index < 1:
        raise ValueError("start_index must be >= 1")

    start = start_index - 1
    if end_index is None:
        return task_files[start:]
    if end_index < start_index:
        raise ValueError("end_index must be >= start_index")
    return task_files[start:end_index]


def save_slice_training_report(
    report_path: str | Path,
    dataset_root: str | Path,
    summary: SliceRunSummary,
    train_summaries: list[TaskTrainSummary],
    eval_report_path: str | Path,
    skipped: list[dict[str, str]],
    config: TrainConfig,
) -> None:
    payload = {
        "dataset_root": str(dataset_root),
        "summary": asdict(summary),
        "train_config": asdict(config),
        "eval_report_path": str(eval_report_path),
        "train_summaries": [asdict(item) for item in train_summaries],
        "skipped": skipped,
    }

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_task_for_training(task_path: str | Path) -> TaskData:
    return load_task_json(task_path)
