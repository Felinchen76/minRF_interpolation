from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from api.checkpoint_management import CheckpointManager
from api.evaluation_api import (
    evaluate_trajectory_metrics,
    global_shift_trajectory,
    identity_trajectory,
    linear_interpolation,
    nn_matched_linear_trajectory,
    trajectory_pixel_smoothness,
)
from api.flow_model_api import ModelSpec, build_flow_model


METHOD_ORDER = (
    "rectified_flow",
    "identity",
    "global_shift",
    "linear_oracle",
    "nn_linear_oracle",
)


def load_fixed_flow(
    checkpoint_path: str | Path,
    device: torch.device,
    expected_source: int = 7,
    expected_target: int = 9,
    expected_profile: str = "medium",
):
    """Load + validate the preselected RF checkpoint; never train or select a model"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Fixed RF checkpoint not found: {checkpoint_path}. "
            "Run notebook 02 and select the checkpoint before evaluation"
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    metadata = payload.get("metadata")
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("Fixed RF checkpoint is missing its training config or metadata")

    expected = {
        "source_class": expected_source,
        "target_class": expected_target,
        "model_profile": expected_profile,
    }
    for key, value in expected.items():
        if config.get(key) != value or metadata.get(key) != value:
            raise ValueError(
                f"Fixed checkpoint {key} mismatch: "
                f"config={config.get(key)!r}, metadata={metadata.get(key)!r}, "
                f"expected={value!r}"
            )

    model_spec = ModelSpec(
        model_profile=expected_profile,
        pairing_mode=config.get("pairing_mode", "random"),
    )
    model, flow_model, _ = build_flow_model(model_spec)
    model.to(device)
    loaded = CheckpointManager(checkpoint_path.parent).load_checkpoint(
        model,
        str(checkpoint_path),
        device=device,
    )
    model.eval()
    flow_model.model = model
    metadata = loaded["metadata"]
    metadata_dict = metadata.to_dict() if hasattr(metadata, "to_dict") else metadata
    return flow_model, metadata_dict, config


def build_reference_banks(
    training_dataset,
    training_label_index: Dict[int, List[int]],
    labels: Tuple[int, ...],
) -> Dict[int, torch.Tensor]:
    """Build class image banks only from official training split"""
    banks = {}
    for label in labels:
        if label not in training_label_index or not training_label_index[label]:
            raise ValueError(f"No training examples found for class {label}")
        banks[label] = torch.stack([
            training_dataset[index][0] for index in training_label_index[label]
        ])
    return banks


def sample_class_pairs(
    label_index: Dict[int, List[int]],
    source_label: int,
    target_label: int,
    n_pairs: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """Choose reproducible, without-replacement source + target test examples"""
    if source_label == target_label:
        raise ValueError("source_label and target_label must differ")
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    source_indices = label_index.get(source_label, [])
    target_indices = label_index.get(target_label, [])
    if n_pairs > min(len(source_indices), len(target_indices)):
        raise ValueError("n_pairs exceeds the number of available class examples")

    rng = random.Random(seed)
    sources = rng.sample(source_indices, n_pairs)
    targets = rng.sample(target_indices, n_pairs)
    return list(zip(sources, targets))


@torch.no_grad()
def evaluate_class_pair_cohort(
    flow_model,
    eval_model,
    test_dataset,
    test_label_index: Dict[int, List[int]],
    reference_banks: Dict[int, torch.Tensor],
    source_label: int,
    target_label: int,
    n_pairs: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate RF + all four agreed baselines on one paired test cohort"""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if source_label not in reference_banks or target_label not in reference_banks:
        raise ValueError("Reference banks must include both source and target classes")

    pair_indices = sample_class_pairs(
        test_label_index, source_label, target_label, n_pairs, seed
    )
    source_images = torch.stack([
        test_dataset[source_index][0] for source_index, _ in pair_indices
    ]).to(device)
    target_images = torch.stack([
        test_dataset[target_index][0] for _, target_index in pair_indices
    ]).to(device)
    source_bank = reference_banks[source_label].to(device)
    target_bank = reference_banks[target_label].to(device)
    source_mean = source_bank.mean(dim=0, keepdim=True)
    target_mean = target_bank.mean(dim=0, keepdim=True)

    trajectories = {
        "rectified_flow": flow_model.sample(source_images, steps=steps),
        "identity": identity_trajectory(source_images, steps),
        "global_shift": global_shift_trajectory(
            source_images, source_mean, target_mean, steps
        ),
        "linear_oracle": linear_interpolation(source_images, target_images, steps),
    }
    trajectories["nn_linear_oracle"], _ = nn_matched_linear_trajectory(
        source_images, target_bank, steps, device=device, metric="l2"
    )

    method_results = {}
    for method, trajectory in trajectories.items():
        metrics = evaluate_trajectory_metrics(
            trajectory,
            eval_model,
            target_label=target_label,
            source_label=source_label,
            device=device,
        )
        method_results[method] = {
            "metrics": metrics.to_dict(),
            "pixel_smoothness": trajectory_pixel_smoothness(trajectory),
        }

    return {
        "pair_indices": [
            {"source_index": source_index, "target_index": target_index}
            for source_index, target_index in pair_indices
        ],
        "trajectories": trajectories,
        "methods": method_results,
        "config": {
            "source_label": source_label,
            "target_label": target_label,
            "n_pairs": n_pairs,
            "steps": steps,
            "seed": seed,
            "methods": list(METHOD_ORDER),
        },
    }


def bootstrap_mean_interval(
    values,
    seed: int = 42,
    resamples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean + percentile 95% bootstrap interval over sample axis"""
    observations = np.asarray(values, dtype=float)
    if observations.ndim < 1 or observations.shape[0] == 0:
        raise ValueError("values must contain at least one sample")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, observations.shape[0], (resamples, observations.shape[0]))
    bootstrap_means = observations[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5], axis=0)
    mean = observations.mean(axis=0)
    return np.asarray(mean), np.asarray(lower), np.asarray(upper)
