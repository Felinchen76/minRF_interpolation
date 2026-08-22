# minRF Interpolation

This repository started as a fork of the minimal rectified-flow implementation from [minRF](https://github.com/cloneofsimo/minRF). The fork keeps the original compact training logic and DiT-based flow setup, but shifts the focus toward a different research question:

- data-to-data transport between image classes
- interpolation quality and trajectory structure
- minimal reusable APIs for experiments in notebooks
- evaluation logic that distinguishes real flow behavior from endpoint-only shortcuts

The main extension in this fork is a small research API around the original architecture. It is designed for FashionMNIST source-target transport experiments, especially Sneaker → Ankle Boot.

## What is new in this fork

- a reusable model and dataset setup in [api/flow_model_api.py](api/flow_model_api.py)
- a lightweight training wrapper in [api/training_api.py](api/training_api.py)
- notebook-first evaluation logic for trajectory quality and semantic transitions
- checkpoint save/load helpers for reproducible experiments
- notebook-friendly setup functions for local runs and Colab

## Current research focus

This project is built around a simple but important idea:

- a flow model should not only land near the target distribution
- it should also move along a semantically meaningful trajectory
- for data-to-data interpolation, the path structure matters as much as the endpoint

The current setup is centered on FashionMNIST with a source-target pair such as:

- source: Sneaker (7)
- target: Ankle Boot (9)

The refactored evaluation notebook compares:

- identity baseline
- global shift baseline
- linear interpolation oracle
- NN-matched linear interpolation oracle
- learned flow trajectory

This makes the project more suitable for research-oriented notebook analysis and for comparing learned transport against simple geometric baselines.

## Project structure

- [dit.py](dit.py): minimal DiT-style transformer backbone
- [rf.py](rf.py): original rectified-flow training logic
- [api/flow_model_api.py](api/flow_model_api.py): reusable model, dataset, and checkpoint API
- [api/training_api.py](api/training_api.py): training wrapper for the data-to-data flow setup
- [notebooks/05_data_to_data_fm_fashionmnist.ipynb](notebooks/05_data_to_data_fm_fashionmnist.ipynb): training/reference notebook
- [notebooks/06_evaluation_refined.ipynb](notebooks/06_evaluation_refined.ipynb): refined evaluation notebook
- [assets/fmnist_d2d_runs](assets/fmnist_d2d_runs): saved checkpoints for the FashionMNIST flow runs

## Minimal usage

The project is intentionally built to be simple in notebooks:

```python
from api.flow_model_api import prepare_fashionmnist_setup, load_checkpoint

setup = prepare_fashionmnist_setup(model_profile='medium', source_class=7, target_class=9)
model = setup['model']
rf_d2d = setup['rf']

ckpt_path = 'assets/fmnist_d2d_runs/fmnist_d2d_checkpoint_best.pt'
payload = load_checkpoint(model, None, ckpt_path, device)
rf_d2d.model = model
```

## Training entry point

Training is kept out of the notebook when the workflow is stable. The reusable training wrapper is in [api/training_api.py](api/training_api.py):

```python
from api.training_api import TrainingSpec, train_data_to_data_flow

spec = TrainingSpec(
    model_profile='medium',
    source_class=7,
    target_class=9,
    checkpoint_dir='assets/fmnist_d2d_runs',
    train_epochs=100,
    quick_run=False,
)

result = train_data_to_data_flow(spec=spec, device=device)
model = result['model']
rf_d2d = result['rf']
```

This keeps the training loop reproducible and leaves the notebook focused on evaluation, path diagnostics, and interpretation.

## Colab quick start

For Colab, the main thing is to make sure the repository root is on `sys.path` before importing the API modules:

```python
import sys
from pathlib import Path

repo_root = Path('/content/minRF_interpolation').resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from api.flow_model_api import prepare_fashionmnist_setup, load_checkpoint
```

This is usually the cleanest way to keep the repo importable without copying the source files into the notebook itself.

## Notes on the fork

This repository keeps the original minimal-implementation philosophy from the upstream project, but the actual contribution here is not a large framework rewrite. It is a smaller, research-focused extension:

- keep the model compact
- keep training logic readable
- move setup and model logic into reusable API modules
- keep evaluation notebook-driven because the core research question is path quality and semantic transitions

## Credits

This project is a fork of the minimal rectified-flow work by Simo Ryu. The core architecture and initial ideas remain from that codebase, while the data-to-data adaptation, evaluation framing, and API cleanup are the project-specific additions in this fork.

## Citation

If you use this repo in its forked or adapted form, please cite the upstream project and also refer to this fork as a custom research extension.

```bibtex
@misc{ryu2024minrf,
  author       = {Simo Ryu},
  title        = {minRF: Minimal Implementation of Scalable Rectified Flow Transformers},
  year         = 2024,
  publisher    = {GitHub},
  url          = {https://github.com/cloneofsimo/minRF}
}
```
