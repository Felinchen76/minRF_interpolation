"""
Checkpoint and Results Management System

This module provides systematic tracking of:
1. Model checkpoints (training progress, best models)
2. Evaluation results (metrics, plots, summaries)
3. Experiment metadata (when, how, what settings)
4. Robustness testing across multiple pairs

Goals:
- Avoid retraining with checkpoint resumption
- Systematically document all results
- Track experiment configuration and outcomes
- Easy loading and comparison of results
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# CHECKPOINT MANAGEMENT

@dataclass
class CheckpointMetadata:
    """
    Metadata for a checkpoint.
    
    Stored alongside checkpoint file to document:
    - Training configuration (model, data, hyperparameters)
    - Training progress (epochs, best loss, date)
    - Source/target setup
    - Associated evaluation results
    """
    
    # training config
    model_profile: str                  # 'small', 'medium', 'large'
    source_class: int                   
    target_class: int                   
    batch_size: int
    learning_rate: float
    train_epochs: int
    early_stopping_patience: int
    
    # training progress
    epochs_completed: int               # how many epochs actually trained
    best_validation_loss: float
    final_training_loss: float
    
    # metadata
    created_at: str                     # ISO timestamp
    device_used: str                    # 'cuda', 'cpu'
    comment: Optional[str] = None       # user notes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointMetadata:
        """Create from dict"""
        return cls(**data)
    
    def save(self, metadata_path: Path):
        """Save metadata to JSON file"""
        with open(metadata_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, metadata_path: Path) -> CheckpointMetadata:
        """Load metadata from JSON file"""
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and metadata
    
    Features:
    - Structured checkpoint directory organization
    - Metadata tracking (config, progress, notes)
    - Best checkpoint detection (by validation loss)
    - Easy resumption from checkpoint
    - Prevent unnecessary retraining
    """
    
    def __init__(self, checkpoint_dir: str | Path):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Base directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, run_name: str, checkpoint_type: str = 'best') -> Path:
        """
        Get path to a checkpoint
        
        Args:
            run_name: Name of run (e.g., 'fmnist_d2d_7_9_medium')
            checkpoint_type: 'best' or 'latest'
        
        Returns:
            Path to checkpoint file
        """
        if checkpoint_type not in {'best', 'latest'}:
            raise ValueError("checkpoint_type must be 'best' or 'latest'")
        suffix = f'checkpoint_{checkpoint_type}'
        return self.checkpoint_dir / f"{run_name}_{suffix}.pt"

    def get_periodic_checkpoint_path(self, run_name: str, epoch: int) -> Path:
        """Get the path for a numbered training snapshot."""
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        return self.checkpoint_dir / f"{run_name}_checkpoint_epoch_{epoch:04d}.pt"

    def get_metadata_path(self, run_name: str) -> Path:
        """Get path to metadata file"""
        return self.checkpoint_dir / f"{run_name}_metadata.json"

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        metadata: Optional[CheckpointMetadata],
        run_name: str,
        checkpoint_type: str = 'best',
        *,
        config: Optional[Dict[str, Any]] = None,
        train_history: Optional[list[float]] = None,
        val_history: Optional[list[float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        device_used: str = 'unknown',
    ) -> Path:
        """
        Save a model checkpoint and, when supplied, complete training state.
        
        Args:
            model: model to save
            optimizer: optimizer state (optional)
            metadata: checkpoint metadata; derived from config and extra for training snapshots
            run_name: name of run
            checkpoint_type: 'best' or 'latest'
            config: training configuration for resumable checkpoints
            train_history: per-epoch training losses
            val_history: per-epoch validation losses
            extra: training progress and scaler state
            epoch: save a numbered snapshot at this epoch instead of best/latest
            device_used: device used for training, included in derived metadata
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = (
            self.get_periodic_checkpoint_path(run_name, epoch)
            if epoch is not None
            else self.get_checkpoint_path(run_name, checkpoint_type)
        )
        metadata_path = self.get_metadata_path(run_name)

        train_history = train_history or []
        val_history = val_history or []
        extra = extra or {}
        if metadata is None:
            if config is None or not extra or not train_history:
                raise ValueError(
                    "metadata is required unless config, training histories, and extra state are provided"
                )
            metadata = CheckpointMetadata(
                model_profile=str(config["model_profile"]),
                source_class=int(config["source_class"]),
                target_class=int(config["target_class"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["lr"]),
                train_epochs=int(config["train_epochs"]),
                early_stopping_patience=int(config["patience"]),
                epochs_completed=int(extra["epoch"]),
                best_validation_loss=float(extra["best_val_loss"]),
                final_training_loss=float(train_history[-1]),
                created_at=datetime.now().isoformat(timespec="seconds"),
                device_used=device_used,
            )

        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata.to_dict(),
        }
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        if config is not None:
            checkpoint_data.update(
                config=config,
                train_history=train_history,
                val_history=val_history,
                extra=extra,
            )

        with tempfile.NamedTemporaryFile(
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(checkpoint_data, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        metadata.save(metadata_path)
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        run_name: str,
        checkpoint_type: str = 'best',
        device: Optional[torch.device] = None,
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint and metadata
        
        Args:
            model: model to load state into
            run_name: run name, or a direct path to a checkpoint file
            checkpoint_type: 'best' or 'latest'
            device: device to load to ('cuda', 'cpu', etc.)
            load_optimizer: whether to load optimizer state
            optimizer: optimizer to load state into (required if load_optimizer=True)
        
        Returns:
            Checkpoint payload plus the loaded 'model' and 'metadata' entries.
            Training checkpoints also include config, histories, and resume state.
            Metadata is None only for older checkpoints without embedded or sidecar metadata.
        """
        candidate_path = Path(run_name)
        checkpoint_path = (
            candidate_path
            if candidate_path.is_file()
            else self.get_checkpoint_path(run_name, checkpoint_type)
        )
        if candidate_path.is_file():
            run_stem = candidate_path.stem
            for suffix in ("_checkpoint_best", "_checkpoint_latest"):
                if run_stem.endswith(suffix):
                    run_stem = run_stem.removesuffix(suffix)
                    break
            else:
                run_stem = run_stem.rsplit("_checkpoint_epoch_", 1)[0]
            metadata_path = self.get_metadata_path(run_stem)
        else:
            metadata_path = self.get_metadata_path(run_name)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata_data = checkpoint.get('metadata')
        if metadata_data is not None:
            metadata = CheckpointMetadata.from_dict(metadata_data)
        elif metadata_path.is_file():
            metadata = CheckpointMetadata.load(metadata_path)
        else:
            metadata = None

        result = {
            **checkpoint,
            'model': model,
            'metadata': metadata,
        }
        if load_optimizer:
            if optimizer is None:
                raise ValueError("optimizer required when load_optimizer=True")
            if 'optimizer_state_dict' not in checkpoint:
                raise ValueError(f"Checkpoint has no optimizer state: {checkpoint_path}")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            result['optimizer'] = optimizer
        return result
    
    def load_metadata(self, run_name: str) -> CheckpointMetadata:
        """Load just metadata without loading model"""
        metadata_path = self.get_metadata_path(run_name)
        return CheckpointMetadata.load(metadata_path)
    
    def checkpoint_exists(self, run_name: str, checkpoint_type: str = 'best') -> bool:
        """Check if a checkpoint exists"""
        return self.get_checkpoint_path(run_name, checkpoint_type).exists()
    
    def list_checkpoints(self) -> Dict[str, Path]:
        """List all available checkpoints"""
        checkpoints = {}
        for checkpoint_file in self.checkpoint_dir.glob('*_checkpoint_best.pt'):
            run_name = checkpoint_file.stem.replace('_checkpoint_best', '')
            checkpoints[run_name] = checkpoint_file
        return checkpoints


# RESULTS MANAGEMENT

@dataclass
class EvaluationResults:
    """
    Container for evaluation results from a single run
    
    Stores:
    - Quantitative metrics (all 5 criteria)
    - Qualitative data (trajectory visualizations)
    - Robustness test results
    - Configuration used for evaluation
    """
    
    # configuration
    model_checkpoint: str
    eval_date: str                      # ISO timestamp
    source_class: int
    target_class: int
    n_test_pairs: int                   # number of pairs tested
    trajectory_steps: int
    
    # quantitative results
    metrics_summary: Dict[str, Dict[str, float]]  # {method: {metric: value}}
    
    # robustness results
    robustness_data: Optional[Dict[str, Any]] = None
    
    # file paths
    plot_paths: Optional[Dict[str, str]] = None  # {plot_name: file_path}
    comment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationResults:
        """Create from dict"""
        return cls(**data)
    
    def save(self, results_path: Path):
        """Save results to JSON"""
        with open(results_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Results saved: {results_path}")
    
    @classmethod
    def load(cls, results_path: Path) -> EvaluationResults:
        """Load results from JSON"""
        with open(results_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ResultsManager:
    """
    Manages systematic storage and retrieval of evaluation results
    
    Features:
    - Timestamped result directories
    - Metadata about evaluation conditions
    - Easy comparison across multiple evaluations
    - Organized plot storage
    """
    
    def __init__(self, results_dir: str | Path):
        """
        Initialize results manager
        
        Args:
            results_dir: Base directory for storing results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run_directory(
        self,
        source_class: int,
        target_class: int,
        model_profile: str
    ) -> Path:
        """
        Create a timestamped directory for a new evaluation run
        
        Returns:
            Path to new directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"eval_{source_class}_{target_class}_{model_profile}_{timestamp}"
        run_dir = self.results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results directory created: {run_dir}")
        return run_dir
    
    def save_results(
        self,
        evaluation_results: EvaluationResults,
        run_directory: Path,
        results_filename: str = 'evaluation_results.json'
    ) -> Path:
        """
        Save evaluation results to JSON in run directory
        
        Args:
            evaluation_results: EvaluationResults object
            run_directory: Directory to save to
            results_filename: Name of results file
        
        Returns:
            Path to saved results file
        """
        results_path = run_directory / results_filename
        evaluation_results.save(results_path)
        return results_path
    
    def save_plots(
        self,
        plot_files: Dict[str, Path],
        run_directory: Path
    ) -> Dict[str, Path]:
        """
        Copy plots to run directory and return their paths
        
        Args:
            plot_files: Dict of {plot_name: source_path}
            run_directory: Destination directory
        
        Returns:
            Dict of {plot_name: destination_path}
        """
        saved_paths = {}
        for plot_name, source_path in plot_files.items():
            source_path = Path(source_path)
            if source_path.exists():
                dest_path = run_directory / source_path.name
                shutil.copy(source_path, dest_path)
                saved_paths[plot_name] = str(dest_path)
                print(f"Plot saved: {dest_path}")
            else:
                print(f"Warning: Plot not found: {source_path}")
        
        return saved_paths
    
    def load_results(
        self,
        run_directory: Path,
        results_filename: str = 'evaluation_results.json'
    ) -> EvaluationResults:
        """Load evaluation results from run directory"""
        results_path = run_directory / results_filename
        return EvaluationResults.load(results_path)
    
    def list_runs(self) -> list[Path]:
        """List all evaluation run directories"""
        return sorted(self.results_dir.glob('eval_*'), reverse=True)
    
    def get_latest_run(self) -> Optional[Path]:
        """Get most recent evaluation run"""
        runs = self.list_runs()
        return runs[0] if runs else None


# EXPERIMENT TRACKING


@dataclass
class ExperimentLog:
    """Log entry for an experiment (train + evaluate)

    Tracks full pipeline from training through evaluation
    """
    
    experiment_id: str                  # unique identifier
    timestamp: str                      # ISO timestamp
    source_class: int                   # configuration
    target_class: int                   # configuration
    model_profile: str
    training_status: str                # training: 'completed', 'resumed', 'skipped'
    evaluation_status: str              # evaluation: 'completed', 'pending', 'skipped'
    checkpoint_path: Optional[str] = None  # optional fields after required
    results_directory: Optional[str] = None
    notes: Optional[str] = None
    
    def save(self, log_file: Path):
        """Append this entry to experiment log"""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        entry = json.dumps(asdict(self), indent=2)
        
        if log_file.exists():
            # append to existing log
            with open(log_file, 'a') as f:
                f.write('\n' + entry + '\n')
        else:
            # create new log
            with open(log_file, 'w') as f:
                f.write(entry + '\n')
        
        print(f"Experiment logged: {log_file}")
