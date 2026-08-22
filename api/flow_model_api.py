from __future__ import annotations

"""Minimal data-to-data flow API for FashionMNIST experiments.

This module keeps the original minimal rectified-flow idea, but wraps it in a
small reusable interface for experiments around source-target image transport.

Core idea:
keep the dataset configuration explicit
keep the model profile configurable
expose a tiny API for setup, checkpointing, and evaluation
stay lightweight enough for notebooks and quick experiments
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from dit import DiT_Llama


# small label metadata for a cleaner data-to-data setup
DEFAULT_FASHIONMNIST_CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


# dataset definition for the data-to-data flow setup
# source and target classes stay explicit for notebooks
@dataclass
class DatasetSpec:
    name: str = "fashionmnist"
    root: str = "data"
    image_size: int = 32
    channels: int = 1
    source_class: int = 7
    target_class: int = 9
    class_names: Dict[int, str] = field(default_factory=lambda: DEFAULT_FASHIONMNIST_CLASS_NAMES.copy())
    batch_size: int = 128
    test_batch_size: int = 128


# small preset table for model scaling
# different profiles map to compact DiT variants
MODEL_PROFILE_CONFIGS = {
    "small": {"dim": 64, "n_layers": 4, "n_heads": 4},
    "medium": {"dim": 128, "n_layers": 6, "n_heads": 8},
    "large": {"dim": 192, "n_layers": 8, "n_heads": 12},
}


# model wrapper for the flow experiment
# architecture choice and hyperparameters stay together
@dataclass
class ModelSpec:
    dataset_name: str = "fashionmnist"
    model_profile: str = "medium"
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    channels: int = 1
    image_size: int = 32
    num_classes: int = 1
    class_dropout_prob: float = 0.0
    lr: float = 2e-4
    pairing_mode: str = "random"
    checkpoint_dir: str = "assets/fmnist_d2d_runs"
    train_epochs: int = 100
    quick_run: bool = False

    def __post_init__(self):
        if self.model_profile not in MODEL_PROFILE_CONFIGS:
            raise ValueError(f"Unsupported model profile: {self.model_profile}")
        cfg = MODEL_PROFILE_CONFIGS[self.model_profile]
        if self.dim is None:
            self.dim = cfg["dim"]
        if self.n_layers is None:
            self.n_layers = cfg["n_layers"]
        if self.n_heads is None:
            self.n_heads = cfg["n_heads"]


# this dataset samples one source image and one target image
# pairing stays random by default for a minimal setup
class PairFashionMNISTDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.FashionMNIST,
        source_class: int,
        target_class: int,
        pairing_mode: str = "random",
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.source_class = int(source_class)
        self.target_class = int(target_class)
        self.pairing_mode = pairing_mode

        self.label_index = {i: [] for i in range(10)}
        for idx, (_, y) in enumerate(self.dataset):
            self.label_index[int(y)].append(idx)

        self.rng = random.Random(seed if seed is not None else 42)

    def __len__(self):
        return max(len(self.label_index[self.source_class]), len(self.label_index[self.target_class]))

    def __getitem__(self, index):
        idx_a = self.rng.choice(self.label_index[self.source_class])
        idx_b = self.rng.choice(self.label_index[self.target_class])

        x_a, _ = self.dataset[idx_a]
        x_b, _ = self.dataset[idx_b]
        return x_a, x_b


# core flow wrapper
# training objective and sampling loop stay here
class DataToDataRF:
    def __init__(self, model: nn.Module, pairing_mode: str = "random"):
        self.model = model
        self.pairing_mode = pairing_mode

    def _dummy_cond(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    @torch.no_grad()
    def _match_targets_by_nn(self, x_a: torch.Tensor, x_b: torch.Tensor):
        a_flat = x_a.flatten(start_dim=1)
        b_flat = x_b.flatten(start_dim=1)
        dist = torch.cdist(a_flat, b_flat, p=2)
        nn_idx = dist.argmin(dim=1)
        return x_b[nn_idx]

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        # flow learns a velocity field from source to target
        if self.pairing_mode == "nn_batch":
            x_b = self._match_targets_by_nn(x_a, x_b)

        b = x_a.size(0)
        cond = self._dummy_cond(b, x_a.device)
        t = torch.rand((b,), device=x_a.device)
        texp = t.view([b, *([1] * len(x_a.shape[1:]))])

        z_t = (1 - texp) * x_a + texp * x_b
        target_velocity = x_b - x_a
        pred_velocity = self.model(z_t, t, cond)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample(self, x_start: torch.Tensor, steps: int = 50):
        # sampling stays intentionally simple and notebook friendly
        b = x_start.size(0)
        cond = self._dummy_cond(b, x_start.device)
        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=x_start.device).view(
            [b, *([1] * len(x_start.shape[1:]))]
        )

        z = x_start.clone()
        trajectory = [z.clone()]
        for i in range(steps):
            t = torch.tensor([(i / steps)] * b, device=x_start.device)
            velocity = self.model(z, t, cond)
            z = z + dt_tensor * velocity
            trajectory.append(z.clone())
        return trajectory


def build_transform(image_size: int = 32):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad((32 - 28) // 2),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


# tiny factory for datasets and dataloaders
# notebooks get one entry point without training noise
def build_dataset_loaders(
    spec: DatasetSpec,
    batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    download: bool = True,
):
    batch_size = batch_size if batch_size is not None else spec.batch_size
    test_batch_size = test_batch_size if test_batch_size is not None else spec.test_batch_size

    transform = build_transform(spec.image_size)
    root_path = Path(spec.root)

    if spec.name.lower() == "fashionmnist":
        train_dataset = datasets.FashionMNIST(
            root=str(root_path),
            train=True,
            download=download,
            transform=transform,
        )
        test_dataset = datasets.FashionMNIST(
            root=str(root_path),
            train=False,
            download=download,
            transform=transform,
        )
    elif spec.name.lower() == "mnist":
        train_dataset = datasets.MNIST(
            root=str(root_path),
            train=True,
            download=download,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root=str(root_path),
            train=False,
            download=download,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {spec.name}")

    train_pair_dataset = PairFashionMNISTDataset(
        train_dataset,
        source_class=spec.source_class,
        target_class=spec.target_class,
        pairing_mode="random",
    )
    test_pair_dataset = PairFashionMNISTDataset(
        test_dataset,
        source_class=spec.source_class,
        target_class=spec.target_class,
        pairing_mode="random",
    )

    train_loader = DataLoader(train_pair_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_pair_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)

    train_label_index = {i: [] for i in range(10)}
    for idx, (_, y) in enumerate(train_dataset):
        train_label_index[int(y)].append(idx)

    test_label_index = {i: [] for i in range(10)}
    for idx, (_, y) in enumerate(test_dataset):
        test_label_index[int(y)].append(idx)

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_label_index": train_label_index,
        "test_label_index": test_label_index,
    }


def build_flow_model(spec: ModelSpec):
    model = DiT_Llama(
        spec.channels,
        spec.image_size,
        dim=spec.dim,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        num_classes=spec.num_classes,
        class_dropout_prob=spec.class_dropout_prob,
    )
    rf = DataToDataRF(model, pairing_mode=spec.pairing_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.lr)
    return model, rf, optimizer


# small helper for predictable checkpoint names
def checkpoint_path_for_run(run_dir: str | Path, profile: str, pairing_mode: str, suffix: str) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"{suffix}_{profile}_{pairing_mode}.pt"


# save a full training snapshot for reproducible runs
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    config: Dict[str, Any],
    train_history: Optional[List[float]] = None,
    val_history: Optional[List[float]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_history": train_history or [],
        "val_history": val_history or [],
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


# restore a saved model state on a target device
def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def build_default_fashionmnist_config(
    model_profile: str = "medium",
    source_class: int = 7,
    target_class: int = 9,
    checkpoint_dir: str = "assets/fmnist_d2d_runs",
    train_epochs: int = 100,
    quick_run: bool = False,
):
    return ModelSpec(
        dataset_name="fashionmnist",
        model_profile=model_profile,
        channels=1,
        image_size=32,
        num_classes=1,
        class_dropout_prob=0.0,
        lr=2e-4,
        pairing_mode="random",
        checkpoint_dir=checkpoint_dir,
        train_epochs=train_epochs,
        quick_run=quick_run,
    )


# convenience constructor for the FashionMNIST data-to-data setup
# this is the main notebook entry point
def prepare_fashionmnist_setup(
    model_profile: str = "medium",
    source_class: int = 7,
    target_class: int = 9,
    root: str = "data",
    batch_size: int = 128,
    test_batch_size: int = 128,
    checkpoint_dir: str = "assets/fmnist_d2d_runs",
    train_epochs: int = 100,
    quick_run: bool = False,
):
    spec = DatasetSpec(
        name="fashionmnist",
        root=root,
        image_size=32,
        channels=1,
        source_class=source_class,
        target_class=target_class,
        class_names=DEFAULT_FASHIONMNIST_CLASS_NAMES.copy(),
        batch_size=batch_size,
        test_batch_size=test_batch_size,
    )

    model_spec = ModelSpec(
        dataset_name="fashionmnist",
        model_profile=model_profile,
        channels=1,
        image_size=32,
        num_classes=1,
        class_dropout_prob=0.0,
        lr=2e-4,
        pairing_mode="random",
        checkpoint_dir=checkpoint_dir,
        train_epochs=train_epochs,
        quick_run=quick_run,
    )

    loaders = build_dataset_loaders(spec, batch_size=batch_size, test_batch_size=test_batch_size)
    model, rf, optimizer = build_flow_model(model_spec)

    return {
        "dataset_spec": spec,
        "model_spec": model_spec,
        "train_loader": loaders["train_loader"],
        "test_loader": loaders["test_loader"],
        "train_dataset": loaders["train_dataset"],
        "test_dataset": loaders["test_dataset"],
        "train_label_index": loaders["train_label_index"],
        "test_label_index": loaders["test_label_index"],
        "model": model,
        "rf": rf,
        "optimizer": optimizer,
    }


def evaluate_on_loader(model_rf: DataToDataRF, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None):
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
    return total / max(1, count), count


__all__ = [
    "DatasetSpec",
    "ModelSpec",
    "DataToDataRF",
    "PairFashionMNISTDataset",
    "DEFAULT_FASHIONMNIST_CLASS_NAMES",
    "MODEL_PROFILE_CONFIGS",
    "build_dataset_loaders",
    "build_flow_model",
    "save_checkpoint",
    "load_checkpoint",
    "prepare_fashionmnist_setup",
    "evaluate_on_loader",
]
