from __future__ import annotations

"""Training API for the data-to-data flow experiments.

This layer is intentionally thin and focused. It keeps the training loop,
checkpoint logic, and early-stopping behavior outside the notebook while leaving
full evaluation and plotting in the notebook where they are most readable.

Core idea:
keep training configuration explicit
keep checkpoint handling reproducible
keep notebook logic focused on analysis and interpretation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from api.flow_model_api import (
    DataToDataRF,
    load_checkpoint,
    prepare_fashionmnist_setup,
    save_checkpoint,
)


# training specification for the data-to-data flow setup
@dataclass
class TrainingSpec:
    model_profile: str = "medium"
    source_class: int = 7
    target_class: int = 9
    root: str = "data"
    batch_size: int = 128
    test_batch_size: int = 128
    checkpoint_dir: str = "assets/fmnist_d2d_runs"
    train_epochs: int = 100
    quick_run: bool = False
    force_retrain: bool = False
    load_best_if_available: bool = True
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    lr: float = 2e-4
    model_name: str = "fmnist_d2d"


# small helper for predictable checkpoint paths
def checkpoint_paths(checkpoint_dir: str | Path, model_name: str = "fmnist_d2d") -> Dict[str, Path]:
    run_dir = Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "latest": run_dir / f"{model_name}_checkpoint_latest.pt",
        "best": run_dir / f"{model_name}_checkpoint_best.pt",
    }


# evaluate the mean flow-matching loss over a loader without gradient tracking
def evaluate_epoch_loss(model_rf: DataToDataRF, loader, device: torch.device, max_batches: Optional[int] = None):
    model_rf.model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for i, (x_a, x_b) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            x_a = x_a.to(device)
            x_b = x_b.to(device)
            loss = model_rf.forward(x_a, x_b)
            total += loss.item()
            count += 1

    return total / max(1, count)


# main training entry point for the research-style data-to-data flow setup
# notebooks can call this once and then focus on evaluation and interpretation
def train_data_to_data_flow(
    spec: Optional[TrainingSpec] = None,
    device: Optional[torch.device] = None,
    **kwargs,
):
    if spec is None:
        spec = TrainingSpec(**kwargs)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup = prepare_fashionmnist_setup(
        model_profile=spec.model_profile,
        source_class=spec.source_class,
        target_class=spec.target_class,
        root=spec.root,
        batch_size=spec.batch_size,
        test_batch_size=spec.test_batch_size,
        checkpoint_dir=spec.checkpoint_dir,
        train_epochs=spec.train_epochs,
        quick_run=spec.quick_run,
    )

    model = setup["model"].to(device)
    rf = setup["rf"]
    rf.model = model
    optimizer = setup["optimizer"]
    train_loader = setup["train_loader"]
    test_loader = setup["test_loader"]

    run_paths = checkpoint_paths(spec.checkpoint_dir, spec.model_name)
    latest_path = run_paths["latest"]
    best_path = run_paths["best"]

    train_loss_history = []
    val_loss_history = []
    loss_history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    if not spec.force_retrain and spec.load_best_if_available and best_path.exists():
        payload = load_checkpoint(model, optimizer, best_path, device)
        train_loss_history = payload.get("train_history", [])
        val_loss_history = payload.get("val_history", [])
        loss_history = payload.get("train_history", [])
        print(f"loaded best checkpoint from {best_path}")
    else:
        epochs = 8 if spec.quick_run else spec.train_epochs
        max_batches = 250 if spec.quick_run else None
        eval_batches = 20 if spec.quick_run else None

        for epoch in range(epochs):
            rf.model.train()
            running = 0.0
            steps_limit = min(len(train_loader), max_batches) if max_batches is not None else len(train_loader)

            for step, (x_a, x_b) in enumerate(train_loader):
                if step >= steps_limit:
                    break

                x_a = x_a.to(device)
                x_b = x_b.to(device)

                optimizer.zero_grad()
                loss = rf.forward(x_a, x_b)
                loss.backward()
                optimizer.step()

                running += loss.item()

            train_epoch_loss = running / max(1, steps_limit)
            val_epoch_loss = evaluate_epoch_loss(rf, test_loader, device, max_batches=eval_batches)

            train_loss_history.append(train_epoch_loss)
            val_loss_history.append(val_epoch_loss)
            loss_history.append(train_epoch_loss)

            config = {
                "dataset": "FashionMNIST",
                "source_class": spec.source_class,
                "target_class": spec.target_class,
                "channels": 1,
                "image_size": 32,
                "model_profile": spec.model_profile,
                "pairing_mode": "random",
                "conditioning": "none",
                "epochs": epochs,
            }

            save_checkpoint(model, optimizer, latest_path, config, train_loss_history, val_loss_history)

            improvement = best_val_loss - val_epoch_loss
            if improvement > spec.min_delta:
                best_val_loss = val_epoch_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, best_path, config, train_loss_history, val_loss_history)
                print(f"new best checkpoint at epoch {best_epoch}: val={best_val_loss:.5f}")
            else:
                epochs_no_improve += 1

            if spec.early_stopping and epochs_no_improve >= spec.patience:
                print(f"early stopping at epoch {epoch + 1} with no improvement")
                break

    return {
        "device": device,
        "setup": setup,
        "model": model,
        "rf": rf,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_history": train_loss_history,
        "val_history": val_loss_history,
        "loss_history": loss_history,
        "best_val_loss": best_val_loss,
        "latest_path": latest_path,
        "best_path": best_path,
        "config": {
            "source_class": spec.source_class,
            "target_class": spec.target_class,
            "model_profile": spec.model_profile,
            "batch_size": spec.batch_size,
            "checkpoint_dir": spec.checkpoint_dir,
            "train_epochs": spec.train_epochs,
            "quick_run": spec.quick_run,
        },
    }


__all__ = [
    "TrainingSpec",
    "checkpoint_paths",
    "evaluate_epoch_loss",
    "train_data_to_data_flow",
]
