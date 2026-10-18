"""
Evaluation API for Data-to-Data Rectified Flow on FashionMNIST

This module provides a comprehensive framework for evaluating semantic meaningfulness
of interpolation trajectories. It implements:

1. Baseline trajectory generators:
   - Identity: no transport (x_t = x_0)
   - Global Shift: class-level mean transport
   - Linear Pixel: direct pixel interpolation (oracle-style)
   - NN-matched Linear: linear interpolation to nearest neighbor (oracle-style)

2. Classifier-based semantic criteria:
   - Classifier-confidence proxy (not a data-distribution plausibility score)
   - Semantic Progression: how consistently semantic properties change
   - Target Alignment & Consistency: movement toward target
   - Source-class confidence change
   - Smoothness: local continuity along trajectory

3. Quantitative metrics computed along trajectories as functions of time t

4. Comparative analysis between RF and baseline methods

All evaluation functions are batched and GPU-accelerated. Metrics are computed
as curves (over trajectory steps) rather than single aggregate values, enabling
detailed path-quality analysis
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# CLASSIFIER FOR SEMANTIC EVALUATION

class FashionMNISTEvalClassifier(nn.Module):
    """
    Lightweight semantic classifier trained on FashionMNIST.
    
    Used as a semantic proxy to measure:
    - target-class confidence (for target alignment)
    - class-wise probabilities (for semantic progression)
    - trajectory diversity (entropy along path)
    
    Architecture: simple CNN, trained independently on full FashionMNIST.
    This keeps evaluation orthogonal to flow model being evaluated.
    """
    
    def __init__(self, image_size: int = 32, channels: int = 1):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        
        # small CNN: 3 conv layers + 2 FC layers
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x -> [batch, 10] logits.
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


# BASELINE TRAJECTORY GENERATORS

def _validate_steps(steps: int) -> None:
    if steps < 1:
        raise ValueError("steps must be at least 1")


@torch.no_grad()
def identity_trajectory(x_source: torch.Tensor, steps: int) -> List[torch.Tensor]:
    """
    Identity baseline: no transport.
    
    For all t ∈ [0, 1], x_t = x_0.
    Represents absence of any learned transformation.
    
    Args:
        x_source: [batch, channels, height, width]
        steps: number of time steps (inclusive, from 0 to 1)
    
    Returns:
        List of (steps+1) tensors, each [batch, channels, height, width]
    """
    _validate_steps(steps)
    trajectory = []
    for _ in range(steps + 1):
        trajectory.append(x_source.clone())
    return trajectory


@torch.no_grad()
def global_shift_trajectory(
    x_source: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    steps: int
) -> List[torch.Tensor]:
    """
    Global Shift baseline: class-level mean transport
    
    Uses only class statistics (mean images) without access to specific target sample.
    For t ∈ [0, 1]: x_t = x_0 + t * (μ_target - μ_source)
    
    This is a non-oracle baseline that does not use specific target image.
    It provides context for interpreting whether flow model learns anything
    beyond simple per-class statistics.
    
    Args:
        x_source: [batch, channels, height, width]
        source_mean: [1, channels, height, width] mean of source class
        target_mean: [1, channels, height, width] mean of target class
        steps: number of time steps
    
    Returns:
        List of (steps+1) tensors
    """
    _validate_steps(steps)
    trajectory = []
    diff = target_mean - source_mean
    
    for t in range(steps + 1):
        alpha = t / steps
        x_t = x_source + alpha * diff
        trajectory.append(x_t)
    
    return trajectory


@torch.no_grad()
def linear_interpolation(
    x_source: torch.Tensor,
    x_target: torch.Tensor,
    steps: int
) -> List[torch.Tensor]:
    """
    Linear pixel-space interpolation (oracle-style baseline).
    
    Directly interpolates between two specific images in pixel space.
    For t ∈ [0, 1]: x_t = (1 - t) * x_0 + t * x_1
    
    Oracle-style because it has access to specific target image x_1.
    Provides a strong reference for endpoint similarity but may not reflect
    path quality of learned flow.
    
    Args:
        x_source: [batch, channels, height, width]
        x_target: [batch, channels, height, width] (same batch size as source)
        steps: number of time steps
    
    Returns:
        List of (steps+1) tensors
    """
    _validate_steps(steps)
    trajectory = []
    
    for t in range(steps + 1):
        alpha = t / steps
        x_t = (1.0 - alpha) * x_source + alpha * x_target
        trajectory.append(x_t)
    
    return trajectory


@torch.no_grad()
def nn_matched_linear_trajectory(
    x_source: torch.Tensor,
    target_class_dataset: torch.Tensor,
    steps: int,
    device: torch.device,
    metric: str = "l2"
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    NN-matched Linear interpolation (oracle-style baseline).
    
    Instead of using an arbitrary target image, selects nearest neighbor
    from target class (in pixel space) and interpolates to that.
    
    For t ∈ [0, 1]: x_t = (1 - t) * x_0 + t * x_1_NN
    
    Oracle-style but with more realistic target selection.
    Serves as a stronger reference for assessing whether simple geometry
    produces semantic transitions when source/target are well-matched.
    
    Args:
        x_source: [batch, channels, height, width]
        target_class_dataset: [n_target, channels, height, width] all target samples
        steps: number of time steps
        device: torch device
        metric: distance metric ("l2" or "cosine")
    
    Returns:
        (trajectory: List of tensors, nearest_neighbor: selected target image)
    """
    _validate_steps(steps)
    # flatten for distance computation
    x_source_flat = x_source.view(x_source.size(0), -1)
    target_flat = target_class_dataset.view(target_class_dataset.size(0), -1)
    
    # compute pairwise distances
    if metric == "l2":
        distances = torch.cdist(x_source_flat.to(device), target_flat.to(device), p=2)
    elif metric == "cosine":
        distances = torch.cdist(x_source_flat.to(device), target_flat.to(device), p=2)
        # Normalize for cosine
        distances = 1.0 - F.cosine_similarity(x_source_flat.to(device).unsqueeze(1),
                                               target_flat.to(device).unsqueeze(0), dim=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # select nearest neighbor for each sample in batch
    nn_indices = distances.argmin(dim=1)
    x_target_nn = target_class_dataset[nn_indices].to(x_source.device)
    
    # linear interpolation to NN
    trajectory = linear_interpolation(x_source, x_target_nn, steps)
    
    return trajectory, x_target_nn


# QUANTITATIVE METRICS FOR EVALUATION CRITERIA

@torch.no_grad()
def classifier_logits(eval_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute FashionMNIST classifier logits
    
    Args:
        eval_model: trained FashionMNISTEvalClassifier
        x: [batch, channels, height, width]
    
    Returns:
        [batch, 10] logits
    """
    return eval_model(x)


@torch.no_grad()
def classifier_probs(eval_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute FashionMNIST classifier probabilities (softmax)
    
    Args:
        eval_model: trained FashionMNISTEvalClassifier
        x: [batch, channels, height, width]
    
    Returns:
        [batch, 10] probabilities
    """
    logits = classifier_logits(eval_model, x)
    return F.softmax(logits, dim=1)


@dataclass
class TrajectoryMetrics:
    """
    Container for quantitative metrics along a trajectory.
    
    Each metric is a curve (list/array) indexed by time step t.
    This allows analysis of progression along the path, not just at endpoints
    """
    
    # semantic progression: how does target-class confidence evolve?
    target_confidence: List[float]  # P(target_class | x_t) for each t
    
    # target alignment & consistency: monotonic movement toward target?
    target_confidence_monotonic: bool
    target_confidence_mean_change: float  # average step-wise change
    
    # semantic progression: class probability entropy (diversity)
    entropy_curve: List[float]  # entropy H(p(x_t)) for each t
    
    # classifier-confidence proxy; not a distributional plausibility guarantee
    max_class_confidence: List[float]  # max_k P(k | x_t)
    
    # source-class confidence and its non-negative endpoint drop
    source_confidence: List[float]  # P(source_class | x_t)
    source_confidence_drop: float  # non-negative drop from initial to final confidence
    
    # smoothness: local changes in class probabilities
    prob_gradient: List[float]  # ||p(x_t) - p(x_{t-1})||_2
    
    # endpoint metrics (for comparison)
    endpoint_target_confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/saving"""
        return {
            "target_confidence": self.target_confidence,
            "target_confidence_monotonic": self.target_confidence_monotonic,
            "target_confidence_mean_change": self.target_confidence_mean_change,
            "entropy_curve": self.entropy_curve,
            "max_class_confidence": self.max_class_confidence,
            "source_confidence": self.source_confidence,
            "source_confidence_drop": self.source_confidence_drop,
            "prob_gradient": self.prob_gradient,
            "endpoint_target_confidence": self.endpoint_target_confidence,
        }


@torch.no_grad()
def evaluate_trajectory_metrics(
    trajectory: List[torch.Tensor],
    eval_model: nn.Module,
    target_label: int,
    source_label: Optional[int] = None,
    device: Optional[torch.device] = None
) -> TrajectoryMetrics:
    """
    Compute quantitative metrics for a trajectory against multiple evaluation criteria.
    
    Implements quantitative versions of:
    - Semantic Progression: via entropy and target-class confidence changes
    - Target Alignment & Consistency: monotonicity of target confidence
    - Source-class confidence drop between trajectory endpoints
    - Classifier-confidence proxy (not a data-distribution or OOD score)
    - Smoothness: probability gradient between steps
    
    Args:
        trajectory: List of [batch, channels, height, width] tensors
        eval_model: trained FashionMNISTEvalClassifier
        target_label: target class index (0-9 for FashionMNIST)
        source_label: source class index (optional, for preservation analysis)
        device: torch device (if None, uses trajectory device)
    
    Returns:
        TrajectoryMetrics dataclass with curves and scalar summaries
    """
    if len(trajectory) < 2:
        raise ValueError("trajectory must contain at least two points")
    if device is None:
        device = trajectory[0].device
    
    # compute class probabilities for all steps
    probs_list = []
    for x_t in trajectory:
        probs = classifier_probs(eval_model, x_t.to(device))
        probs_list.append(probs)
    
    # stack into [steps, batch, 10]
    probs = torch.stack(probs_list, dim=0)  # [steps, batch, 10]
    mean_probs = probs.mean(dim=1)
    
    # ========== SEMANTIC PROGRESSION ==========
    # target-class confidence curve
    target_confidence = mean_probs[:, target_label].cpu().numpy().tolist()
    
    # entropy curve (diversity of predictions)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=2).mean(dim=1)
    entropy_curve = entropy.cpu().numpy().tolist()
    
    # ========== target alignment & consistency ==========
    # check monotonicity: does target confidence increase consistently?
    target_conf_tensor = mean_probs[:, target_label]
    diffs = target_conf_tensor[1:] - target_conf_tensor[:-1]
    target_confidence_monotonic = bool((diffs >= -0.01).all().item())  # allow small jitter
    target_confidence_mean_change = float(diffs.mean().item())
    
    # ========== classifier-confidence proxy ==========
    max_class_confidence = probs.max(dim=2).values.mean(dim=1).cpu().numpy().tolist()
    
    # ========== PRESERVATION ==========
    source_confidence = []
    source_confidence_drop = 0.0
    if source_label is not None:
        source_confidence = mean_probs[:, source_label].cpu().numpy().tolist()
        # Measure the non-negative drop in source-class confidence.
        initial_source_conf = source_confidence[0]
        final_source_conf = source_confidence[-1]
        source_confidence_drop = float(max(0, initial_source_conf - final_source_conf))
    
    # ========== SMOOTHNESS ==========
    # L2 distance in probability space between consecutive steps
    prob_gradient = (
        (probs[1:] - probs[:-1]).norm(p=2, dim=2).mean(dim=1).cpu().numpy().tolist()
    )
    
    # ========== ENDPOINT ==========
    endpoint_target_confidence = float(target_confidence[-1])
    
    return TrajectoryMetrics(
        target_confidence=target_confidence,
        target_confidence_monotonic=target_confidence_monotonic,
        target_confidence_mean_change=target_confidence_mean_change,
        entropy_curve=entropy_curve,
        max_class_confidence=max_class_confidence,
        source_confidence=source_confidence if source_label is not None else [],
        source_confidence_drop=source_confidence_drop,
        prob_gradient=prob_gradient,
        endpoint_target_confidence=endpoint_target_confidence,
    )


# PIXEL-SPACE SMOOTHNESS METRICS

@torch.no_grad()
def trajectory_pixel_smoothness(trajectory: List[torch.Tensor]) -> Dict[str, float]:
    """
    Evaluate pixel-space smoothness of a trajectory
    
    Smoothness criterion: no abrupt local changes or discontinuities
    
    Measures:
    - Mean L2 distance between consecutive frames
    - Max L2 distance (largest jump)
    - Variance of step sizes (consistency)
    
    Args:
        trajectory: List of [batch, channels, height, width]
    
    Returns:
        Dict with smoothness metrics
    """
    if len(trajectory) < 2:
        raise ValueError("trajectory must contain at least two points")
    step_sizes = []
    
    for i in range(1, len(trajectory)):
        x_prev = trajectory[i-1]
        x_curr = trajectory[i]
        
        # L2 distance in pixel space
        delta = (x_curr - x_prev).flatten(start_dim=1).norm(p=2, dim=1).mean().item()
        step_sizes.append(delta)
    
    step_sizes = np.array(step_sizes)
    
    return {
        "mean_step_size": float(np.mean(step_sizes)),
        "max_step_size": float(np.max(step_sizes)),
        "step_size_std": float(np.std(step_sizes)),
        "step_sizes": step_sizes.tolist(),
    }


# ROBUSTNESS EVALUATION

@torch.no_grad()
def intraclass_robustness_test(
    flow_model,
    eval_model: nn.Module,
    test_dataset,
    test_label_index: Dict[int, List[int]],
    source_label: int,
    target_label: int,
    n_pairs: int = 10,
    steps: int = 30,
    device: Optional[torch.device] = None
) -> Dict[str, List]:
    """
    Robustness evaluation: how consistent is semantic behavior across different
    source-target pairs of the same classes?
    
    For a given source-target class pair, sample multiple pairs of actual images
    from those classes and evaluate whether semantic properties are consistent
    (not just successful for one lucky pair).
    
    Args:
        flow_model: trained flow model with a .sample() method
        eval_model: FashionMNISTEvalClassifier
        test_dataset: full FashionMNIST test set
        test_label_index: Dict[int, List[int]] mapping class -> image indices
        source_label: source class
        target_label: target class
        n_pairs: number of source-target pairs to test
        steps: trajectory steps
        device: torch device
    
    Returns:
        Dict with results per method:
        {
            'method_name': [metrics_per_pair],
            ...
        }
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1")
    _validate_steps(steps)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        'identity': [],
        'linear_oracle': [],
        'flow': [],
    }
    
    for _ in range(n_pairs):
        # Sample random pair
        idx_source = random.choice(test_label_index[source_label])
        idx_target = random.choice(test_label_index[target_label])
        
        x_source, _ = test_dataset[idx_source]
        x_target, _ = test_dataset[idx_target]
        
        x_source = x_source.unsqueeze(0).to(device)
        x_target = x_target.unsqueeze(0).to(device)
        
        # generate baselines
        traj_identity = identity_trajectory(x_source, steps)
        traj_linear = linear_interpolation(x_source, x_target, steps)
        
        traj_flow = flow_model.sample(x_source, steps=steps)
        
        # evaluate each
        for name, traj in [
            ('identity', traj_identity),
            ('linear_oracle', traj_linear),
            ('flow', traj_flow),
        ]:
            metrics = evaluate_trajectory_metrics(
                traj, eval_model, target_label, source_label, device
            )
            results[name].append({
                'endpoint_acc': metrics.endpoint_target_confidence,
                'monotonic': metrics.target_confidence_monotonic,
                'mean_change': metrics.target_confidence_mean_change,
            })
    
    return results


# COMPARATIVE ANALYSIS

@torch.no_grad()
def compare_trajectories(
    trajectories: Dict[str, List[torch.Tensor]],
    eval_model: nn.Module,
    target_label: int,
    source_label: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[str, TrajectoryMetrics]:
    """
    Evaluate multiple trajectories (e.g., different methods) using the same
    metrics for direct comparison

    Args:
        trajectories: Dict[method_name, trajectory]
        eval_model: FashionMNISTEvalClassifier
        target_label: target class
        source_label: source class (optional)
        device: torch device
    
    Returns:
        Dict[method_name, TrajectoryMetrics]
    """
    results = {}
    for name, traj in trajectories.items():
        metrics = evaluate_trajectory_metrics(traj, eval_model, target_label, source_label, device)
        results[name] = metrics
    
    return results


def summarize_comparison(comparison_results: Dict[str, TrajectoryMetrics]) -> Dict[str, Dict]:
    """
    Create a summary table from comparison results
    
    Extracts key scalar metrics for easy comparison across methods
    
    Args:
        comparison_results: output of compare_trajectories
    
    Returns:
        Dict[method_name, Dict[metric_name, value]]
    """
    summary = {}
    
    for method_name, metrics in comparison_results.items():
        summary[method_name] = {
            'endpoint_target_conf': metrics.endpoint_target_confidence,
            'target_conf_monotonic': metrics.target_confidence_monotonic,
            'target_conf_mean_change': metrics.target_confidence_mean_change,
            'source_confidence_drop': metrics.source_confidence_drop,
            'mean_entropy': np.mean(metrics.entropy_curve) if metrics.entropy_curve else 0.0,
            'mean_max_class_conf': np.mean(metrics.max_class_confidence),
            'mean_prob_gradient': np.mean(metrics.prob_gradient) if metrics.prob_gradient else 0.0,
        }
    
    return summary
