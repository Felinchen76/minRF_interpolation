from __future__ import annotations

import os
from pathlib import Path


def find_repository_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or Path.cwd()).resolve()
    repository_name = "minRF_interpolation"
    candidates = [start_path, start_path / repository_name]
    candidates.extend(
        candidate
        for parent in start_path.parents
        for candidate in (parent, parent / repository_name)
    )
    candidates.extend(
        (Path("/content/minRF_interpolation"), Path("/content/minrf/minRF_interpolation"))
    )

    for candidate in candidates:
        if (candidate / "api" / "flow_model_api.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate the minRF_interpolation repository. "
        "Run notebooks/00_colab_setup.ipynb first."
    )


def find_checkpoint_directory(
    repository_root: str | Path,
    run_directory: str = "fmnist_d2d_runs",
) -> Path:
    repository_root = Path(repository_root).resolve()
    candidates = [
        repository_root / "assets" / run_directory,
    ]
    runtime_root = os.environ.get("MINRF_RUNTIME_ROOT")
    if runtime_root:
        candidates.append(Path(runtime_root) / "assets" / run_directory)
    candidates.append(repository_root.parent / "assets" / run_directory)
    candidates.extend(
        parent / "minrf" / "assets" / run_directory
        for parent in (repository_root.parent, repository_root.parent.parent)
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if any(candidate.glob("*_checkpoint_best.pt")):
            return candidate

    return candidates[0].resolve()
