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

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from api.checkpoint_management import CheckpointManager
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
    validation_fraction: float = 0.1
    seed: int = 42
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
        seed: int = 42,
    ):
        self.dataset = dataset
        self.source_class = int(source_class)
        self.target_class = int(target_class)
        self.pairing_mode = pairing_mode

        if isinstance(self.dataset, Subset):
            targets = self.dataset.dataset.targets[self.dataset.indices]
        else:
            targets = self.dataset.targets
        self.label_index = {i: [] for i in range(10)}
        for idx, label in enumerate(targets):
            self.label_index[int(label)].append(idx)

        self.rng = random.Random(seed)

    def __len__(self):
        return max(len(self.label_index[self.source_class]), len(self.label_index[self.target_class]))

    def __getitem__(self, index):
        idx_a = self.rng.choice(self.label_index[self.source_class])
        idx_b = self.rng.choice(self.label_index[self.target_class])

        x_a, _ = self.dataset[idx_a]
        x_b, _ = self.dataset[idx_b]
        return x_a, x_b

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        self.rng.setstate(state)


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
    if image_size < 1:
        raise ValueError("image_size must be a positive integer")

    image_transforms = [transforms.ToTensor()]
    if image_size >= 28:
        padding = image_size - 28
        left = top = padding // 2
        right = bottom = padding - padding // 2
        if padding:
            image_transforms.append(transforms.Pad((left, top, right, bottom)))
    else:
        image_transforms.append(transforms.Resize((image_size, image_size)))
    image_transforms.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(image_transforms)


# tiny factory for datasets and dataloaders
# notebooks get one entry point without training noise
def build_dataset_loaders(
    spec: DatasetSpec,
    batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    download: bool = True,
):
    if not 0.0 < spec.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
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

    labels = train_dataset.targets.tolist()
    indices_by_label = {label: [] for label in range(10)}
    for idx, label in enumerate(labels):
        indices_by_label[int(label)].append(idx)

    generator = torch.Generator().manual_seed(spec.seed)
    train_indices = []
    validation_indices = []
    for label_indices in indices_by_label.values():
        permutation = torch.randperm(len(label_indices), generator=generator).tolist()
        shuffled_indices = [label_indices[index] for index in permutation]
        validation_count = max(1, round(len(shuffled_indices) * spec.validation_fraction))
        if len(shuffled_indices) < 2:
            validation_count = 0
        validation_indices.extend(shuffled_indices[:validation_count])
        train_indices.extend(shuffled_indices[validation_count:])

    train_split = Subset(train_dataset, train_indices)
    validation_split = Subset(train_dataset, validation_indices)
    train_pair_dataset = PairFashionMNISTDataset(
        train_split,
        source_class=spec.source_class,
        target_class=spec.target_class,
        pairing_mode="random",
        seed=spec.seed,
    )
    validation_pair_dataset = PairFashionMNISTDataset(
        validation_split,
        source_class=spec.source_class,
        target_class=spec.target_class,
        pairing_mode="random",
        seed=spec.seed + 1,
    )
    test_pair_dataset = PairFashionMNISTDataset(
        test_dataset,
        source_class=spec.source_class,
        target_class=spec.target_class,
        pairing_mode="random",
        seed=spec.seed + 2,
    )

    train_loader = DataLoader(
        train_pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(spec.seed + 3),
    )
    validation_loader = DataLoader(
        validation_pair_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(spec.seed + 4),
    )
    test_loader = DataLoader(
        test_pair_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(spec.seed + 5),
    )

    train_label_index = {i: [] for i in range(10)}
    for idx, label in enumerate(train_dataset.targets):
        train_label_index[int(label)].append(idx)

    test_label_index = {i: [] for i in range(10)}
    for idx, label in enumerate(test_dataset.targets):
        test_label_index[int(label)].append(idx)

    return {
        "train_dataset": train_dataset,
        "validation_dataset": validation_split,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
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


# Compatibility helpers; CheckpointManager owns checkpoint serialization.
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
    manager = CheckpointManager(checkpoint_path.parent)
    return manager.save_checkpoint(
        model,
        optimizer,
        None,
        str(checkpoint_path),
        config=config,
        train_history=train_history,
        val_history=val_history,
        extra=extra,
    )


# restore a saved model state on a target device
def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], checkpoint_path: str | Path, device: torch.device):
    manager = CheckpointManager(Path(checkpoint_path).parent)
    return manager.load_checkpoint(
        model,
        str(checkpoint_path),
        device=device,
        load_optimizer=optimizer is not None,
        optimizer=optimizer,
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
    lr: float = 2e-4,
    validation_fraction: float = 0.1,
    seed: int = 42,
):
    spec = DatasetSpec(
        name="fashionmnist",
        root=root,
        image_size=32,
        channels=1,
        source_class=source_class,
        target_class=target_class,
        validation_fraction=validation_fraction,
        seed=seed,
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
        lr=lr,
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
        "validation_loader": loaders["validation_loader"],
        "test_loader": loaders["test_loader"],
        "train_dataset": loaders["train_dataset"],
        "test_dataset": loaders["test_dataset"],
        "train_label_index": loaders["train_label_index"],
        "test_label_index": loaders["test_label_index"],
        "model": model,
        "rf": rf,
        "optimizer": optimizer,
    }


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
]
