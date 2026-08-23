from __future__ import annotations

"""Evaluation suite for data-to-data flow analysis.

This module provides reusable evaluation utilities across the experimental phases.
It keeps metrics computation, result collection, and formatting in one place while
leaving experiment-specific logic and visualization in notebooks.

Core idea:
keep evaluation metrics focused and well-defined
keep result aggregation and I/O separate from computation
stay lightweight enough for quick metric lookups in notebooks
support multiple experimental phases with minimal code duplication
"""

import json
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# small preset table for FashionMNIST class labels
# used consistently across notebooks for pair formatting and results display
FMNIST_CLASS_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
    8: "Bag", 9: "Ankle Boot",
}


def format_pair_name(source: int, target: int) -> str:
    """format a class pair with readable names"""
    return f"{FMNIST_CLASS_NAMES[source]}({source}) → {FMNIST_CLASS_NAMES[target]}({target})"


# standard small CNN used as external classifier for trajectory evaluation
# shared architecture across all phases for consistent comparison
def build_eval_classifier() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def train_eval_classifier(
    device: torch.device,
    data_root: str = "data",
    epochs: int = 5,
    save_path: Optional[str] = None,
) -> nn.Sequential:
    """train the shared eval classifier on full FashionMNIST
    returns a trained model ready for evaluation use"""
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

    model = build_eval_classifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
        print(f"  Classifier epoch {epoch + 1}/{epochs} — acc: {correct / total:.3f}")

    model.eval()

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"  ok Classifier saved: {save_path}")

    return model


def load_eval_classifier(path: str, device: torch.device) -> nn.Sequential:
    """load a previously trained eval classifier"""
    model = build_eval_classifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
# computes trajectory metrics and aggregates over random samples
class FlowEvaluator:
    def __init__(self, eval_model, test_dataset, test_label_index, device):
        self.eval_model = eval_model
        self.test_dataset = test_dataset
        self.test_label_index = test_label_index
        self.device = device

    @torch.no_grad()
    def compute_monotonicity(self, trajectory: torch.Tensor, target_label: int) -> float:
        """compute % of steps where target class confidence increases"""
        probs = []
        for z in trajectory:
            logits = self.eval_model(z.to(self.device))
            p = torch.softmax(logits, dim=1)[0].detach().cpu()
            probs.append(p)

        probs = torch.stack(probs, dim=0)
        target_conf = probs[:, target_label]

        diff = target_conf[1:] - target_conf[:-1]
        monotonicity = float((diff > 0).float().mean().item())

        return monotonicity

    @torch.no_grad()
    def compute_smoothness(self, trajectory: torch.Tensor) -> float:
        """compute smoothness as 1 / (1 + mean_step_diff)
        higher values indicate smoother transitions"""
        diffs = []
        for i in range(len(trajectory) - 1):
            diff = torch.norm(trajectory[i] - trajectory[i + 1]).item()
            diffs.append(diff)

        if not diffs:
            return 1.0

        smoothness = 1.0 / (1.0 + np.mean(diffs))
        return float(smoothness)

    @torch.no_grad()
    def evaluate_pair(
        self, flow_model, x_start: torch.Tensor, target_label: int, steps: int = 50
    ) -> Dict[str, float]:
        """evaluate flow for a single source sample"""
        x_start = x_start.to(self.device)
        traj = flow_model.sample(x_start, steps=steps)

        metrics = {
            "monotonicity": self.compute_monotonicity(traj, target_label),
            "smoothness": self.compute_smoothness(torch.stack(traj)),
        }

        return metrics

    @torch.no_grad()
    def evaluate_classepair(
        self, flow_model, source_class: int, target_class: int, n_pairs: int = 20, steps: int = 50
    ) -> Dict[str, Any]:
        """evaluate flow over multiple random samples from a class pair
        returns aggregated statistics and individual sample results"""
        results = []

        for _ in range(n_pairs):
            idx_a = np.random.choice(self.test_label_index[source_class])
            x_a, _ = self.test_dataset[idx_a]
            x_a = x_a.unsqueeze(0).to(self.device)

            metrics = self.evaluate_pair(flow_model, x_a, target_class, steps=steps)
            results.append(metrics)

        # aggregate metrics across samples
        all_mono = [r["monotonicity"] for r in results]
        all_smooth = [r["smoothness"] for r in results]

        summary = {
            "monotonicity_mean": float(np.mean(all_mono)),
            "monotonicity_std": float(np.std(all_mono)),
            "monotonicity_min": float(np.min(all_mono)),
            "monotonicity_max": float(np.max(all_mono)),
            "smoothness_mean": float(np.mean(all_smooth)),
            "smoothness_std": float(np.std(all_smooth)),
            "n_pairs": n_pairs,
            "samples": results,
        }

        return summary


# result collection and output formatting for experiment phases
# keeps JSON serialization and table rendering separate from metrics
class ResultsCollector:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = OrderedDict()

    def add_result(self, key: str, metrics: Dict[str, float]):
        """add a single result entry"""
        self.results[key] = metrics

    def add_pair_result(
        self, source: int, target: int, metrics: Dict[str, Any], class_names: Dict[int, str] | None = None
    ):
        """add a class-pair result with human-readable formatting"""
        pair_key = f"{source:02d}→{target:02d}"
        if class_names:
            pair_key = f"{class_names[source]}({source})→{class_names[target]}({target})"

        self.add_result(pair_key, metrics)

    def to_json(self, filename: str = "results.json") -> Path:
        """serialize results to JSON, converting numpy/torch types to native Python"""
        output_path = self.output_dir / filename

        clean_results = {}
        for key, metrics in self.results.items():
            clean_results[key] = {
                k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                for k, v in metrics.items()
                if not isinstance(v, list)
            }

        with open(output_path, "w") as f:
            json.dump(clean_results, f, indent=2)

        return output_path

    def to_table_string(self, metrics_to_show: List[str] | None = None) -> str:
        """format results as a readable ASCII table"""
        if not metrics_to_show:
            metrics_to_show = ["monotonicity_mean", "smoothness_mean"]

        lines = []
        header = "Pair" + "".join(f" | {m:>15}" for m in metrics_to_show)
        lines.append(header)
        lines.append("-" * len(header))

        for pair, metrics in self.results.items():
            row = pair
            for m in metrics_to_show:
                val = metrics.get(m, np.nan)
                if isinstance(val, float):
                    row += f" | {val:>15.4f}"
                else:
                    row += f" | {str(val):>15}"
            lines.append(row)

        return "\n".join(lines)

    def print_summary(self, metrics_to_show: List[str] | None = None):
        """print a formatted summary table to stdout"""
        print(self.to_table_string(metrics_to_show))
        print(f"\nok Results saved to {self.output_dir}")
