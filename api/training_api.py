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

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
import torch

from api.checkpoint_management import CheckpointManager
from api.flow_model_api import (
    DataToDataRF,
    prepare_fashionmnist_setup,
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
    checkpoint_every: int = 20
    resume_from: Optional[str] = None
    mixed_precision: bool = False
    validation_fraction: float = 0.1
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if self.train_epochs < 1:
            raise ValueError("train_epochs must be at least 1")
        if self.resume_from is not None and self.force_retrain:
            raise ValueError("resume_from and force_retrain cannot be used together")


# small helper for predictable checkpoint paths
def checkpoint_paths(checkpoint_dir: str | Path, model_name: str = "fmnist_d2d") -> Dict[str, Path]:
    manager = CheckpointManager(checkpoint_dir)
    return {
        "latest": manager.get_checkpoint_path(model_name, "latest"),
        "best": manager.get_checkpoint_path(model_name, "best"),
    }


# evaluate the mean flow-matching loss over a loader without gradient tracking
def evaluate_epoch_loss(
    model_rf: DataToDataRF,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
    mixed_precision: bool = False,
):
    model_rf.model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for i, (x_a, x_b) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            x_a = x_a.to(device)
            x_b = x_b.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                loss = model_rf.forward(x_a, x_b)
            batch_size = x_a.size(0)
            total += loss.item() * batch_size
            count += batch_size

    return total / max(1, count)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def _capture_rng_state(loaders: Dict[str, Any]) -> Dict[str, Any]:
    numpy_name, numpy_values, numpy_position, numpy_has_gauss, numpy_cached_gaussian = (
        np.random.get_state()
    )
    loader_states = {}
    for name, loader in loaders.items():
        dataset = getattr(loader, "dataset", None)
        generator = getattr(loader, "generator", None)
        loader_states[name] = {
            "generator": generator.get_state() if generator is not None else None,
            "dataset": (
                dataset.get_rng_state()
                if dataset is not None and hasattr(dataset, "get_rng_state")
                else None
            ),
        }

    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_name,
            "values": numpy_values.tolist(),
            "position": numpy_position,
            "has_gauss": numpy_has_gauss,
            "cached_gaussian": numpy_cached_gaussian,
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": loader_states,
    }


def _restore_rng_state(state: Dict[str, Any], loaders: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            np.asarray(numpy_state["values"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    cuda_states = state["cuda"]
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Cannot exactly resume: checkpoint and current run use different "
                "numbers of CUDA devices."
            )
        torch.cuda.set_rng_state_all(cuda_states)

    loader_states = state["loaders"]
    for name, loader in loaders.items():
        saved_state = loader_states[name]
        generator = getattr(loader, "generator", None)
        dataset = getattr(loader, "dataset", None)
        if saved_state["generator"] is not None and generator is not None:
            generator.set_state(saved_state["generator"])
        if saved_state["dataset"] is not None and dataset is not None:
            dataset.set_rng_state(saved_state["dataset"])


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

    if spec.resume_from is not None and not Path(spec.resume_from).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {spec.resume_from}")

    seed_everything(spec.seed, spec.deterministic)

    setup = prepare_fashionmnist_setup(
        model_profile=spec.model_profile,
        source_class=spec.source_class,
        target_class=spec.target_class,
        root=spec.root,
        batch_size=spec.batch_size,
        test_batch_size=spec.test_batch_size,
        validation_fraction=spec.validation_fraction,
        seed=spec.seed,
        checkpoint_dir=spec.checkpoint_dir,
        train_epochs=spec.train_epochs,
        quick_run=spec.quick_run,
        lr=spec.lr,
    )

    model = setup["model"].to(device)
    rf = setup["rf"]
    rf.model = model
    optimizer = setup["optimizer"]
    train_loader = setup["train_loader"]
    validation_loader = setup["validation_loader"]
    test_loader = setup["test_loader"]

    checkpoint_manager = CheckpointManager(spec.checkpoint_dir)
    run_paths = checkpoint_paths(spec.checkpoint_dir, spec.model_name)
    latest_path = run_paths["latest"]
    best_path = run_paths["best"]

    train_loss_history = []
    val_loss_history = []
    loss_history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 0
    epoch_timings = []
    periodic_paths = []
    amp_enabled = spec.mixed_precision and device.type == "cuda"
    # The CUDA namespace also supports the project's minimum PyTorch 2.0.
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    config = asdict(spec)
    config.update(dataset="FashionMNIST", channels=1, image_size=32,
                  pairing_mode="random", conditioning="none")

    if spec.resume_from is not None:
        payload = checkpoint_manager.load_checkpoint(
            model, spec.resume_from, device=device, load_optimizer=True, optimizer=optimizer
        )
        state = payload.get("extra", {})
        required = {"epoch", "best_val_loss", "best_epoch", "epochs_no_improve", "scaler"}
        if not required.issubset(state) or "optimizer_state_dict" not in payload:
            raise ValueError(
                "This checkpoint has no complete resume state. Use a checkpoint created "
                "by the updated training API; legacy weights can only be loaded for evaluation."
            )
        saved_config = payload.get("config", {})
        for key in (
            "source_class",
            "target_class",
            "model_profile",
            "batch_size",
            "test_batch_size",
            "lr",
            "mixed_precision",
            "validation_fraction",
            "seed",
            "deterministic",
        ):
            saved_value = saved_config.get(key)
            if key == "seed" and saved_value is None:
                saved_value = saved_config.get("split_seed", 42)
            elif key == "deterministic" and saved_value is None:
                saved_value = config[key]
            if saved_value != config[key]:
                raise ValueError(f"Resume configuration differs for {key}")
        train_loss_history = list(payload["train_history"])
        val_loss_history = list(payload["val_history"])
        start_epoch = state["epoch"]
        if len(train_loss_history) != start_epoch or len(val_loss_history) != start_epoch:
            raise ValueError("Checkpoint epoch and loss histories do not match")
        loss_history = list(train_loss_history)
        best_val_loss = state["best_val_loss"]
        best_epoch = state["best_epoch"]
        epochs_no_improve = state["epochs_no_improve"]
        scaler.load_state_dict(state["scaler"])
        if "rng_state" in state:
            _restore_rng_state(
                state["rng_state"],
                {
                    "train": train_loader,
                    "validation": validation_loader,
                    "test": test_loader,
                },
            )
        else:
            print("warning: checkpoint has no RNG state; resumed run will not be exact")
        print(f"resuming from {spec.resume_from} after epoch {start_epoch}")

    if (spec.resume_from is None and not spec.force_retrain
            and spec.load_best_if_available and best_path.exists()):
        payload = checkpoint_manager.load_checkpoint(
            model, spec.model_name, checkpoint_type="best", device=device,
            load_optimizer=True, optimizer=optimizer,
        )
        train_loss_history = payload.get("train_history", [])
        val_loss_history = payload.get("val_history", [])
        loss_history = payload.get("train_history", [])
        state = payload.get("extra", {})
        best_val_loss = state.get("best_val_loss", min(val_loss_history, default=float("inf")))
        best_epoch = state.get(
            "best_epoch", val_loss_history.index(best_val_loss) + 1 if val_loss_history else -1
        )
        print(f"loaded best checkpoint from {best_path}")
    else:
        epochs = 8 if spec.quick_run else spec.train_epochs
        max_batches = 250 if spec.quick_run else None
        eval_batches = 20 if spec.quick_run else None

        print(
            f"Training on {device}"
            f" ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})"
            f" | AMP: {amp_enabled} | batches: {len(train_loader)} train,"
            f" {len(validation_loader)} val"
        )
        if spec.mixed_precision and not amp_enabled:
            print("CUDA AMP requested but unavailable on this device; using FP32.")

        for epoch in range(start_epoch, epochs):
            if spec.early_stopping and epochs_no_improve >= spec.patience:
                print("checkpoint already reached early stopping; increase patience to continue")
                break
            epoch_start = perf_counter()
            rf.model.train()
            running = 0.0
            steps_limit = min(len(train_loader), max_batches) if max_batches is not None else len(train_loader)

            for step, (x_a, x_b) in enumerate(train_loader):
                if step >= steps_limit:
                    break

                x_a = x_a.to(device)
                x_b = x_b.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    loss = rf.forward(x_a, x_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running += loss.item()

            train_epoch_loss = running / max(1, steps_limit)
            train_seconds = perf_counter() - epoch_start
            val_start = perf_counter()
            val_epoch_loss = evaluate_epoch_loss(
                rf, validation_loader, device,
                max_batches=eval_batches, mixed_precision=amp_enabled
            )
            val_seconds = perf_counter() - val_start

            train_loss_history.append(train_epoch_loss)
            val_loss_history.append(val_epoch_loss)
            loss_history.append(train_epoch_loss)

            improvement = best_val_loss - val_epoch_loss
            if improvement > spec.min_delta:
                best_val_loss = val_epoch_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            state = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "scaler": scaler.state_dict(),
                "rng_state": _capture_rng_state(
                    {
                        "train": train_loader,
                        "validation": validation_loader,
                        "test": test_loader,
                    }
                ),
            }
            save_start = perf_counter()
            checkpoint_manager.save_checkpoint(
                model, optimizer, None, spec.model_name,
                checkpoint_type="latest", config=config,
                train_history=train_loss_history, val_history=val_loss_history,
                extra=state, device_used=str(device),
            )
            if improvement > spec.min_delta:
                checkpoint_manager.save_checkpoint(
                    model, optimizer, None, spec.model_name,
                    checkpoint_type="best", config=config,
                    train_history=train_loss_history, val_history=val_loss_history,
                    extra=state, device_used=str(device),
                )
            if (epoch + 1) % spec.checkpoint_every == 0:
                periodic_path = checkpoint_manager.save_checkpoint(
                    model, optimizer, None, spec.model_name,
                    config=config, train_history=train_loss_history,
                    val_history=val_loss_history, extra=state,
                    epoch=epoch + 1, device_used=str(device),
                )
                periodic_paths.append(periodic_path)
                print(f"saved periodic checkpoint: {periodic_path}")
            save_seconds = perf_counter() - save_start
            epoch_timings.append({
                "epoch": epoch + 1, "train_seconds": train_seconds,
                "val_seconds": val_seconds, "checkpoint_seconds": save_seconds,
            })
            print(
                f"Epoch {epoch + 1}/{epochs} | train={train_epoch_loss:.6f}"
                f" | val={val_epoch_loss:.6f} | best={best_val_loss:.6f} @ {best_epoch}"
                f" | train {train_seconds:.1f}s, val {val_seconds:.1f}s,"
                f" save {save_seconds:.1f}s"
            )

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
        "validation_loader": validation_loader,
        "train_history": train_loss_history,
        "val_history": val_loss_history,
        "loss_history": loss_history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(train_loss_history),
        "epoch_timings": epoch_timings,
        "periodic_paths": periodic_paths,
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
    "seed_everything",
    "train_data_to_data_flow",
]
