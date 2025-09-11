#!/usr/bin/env python3
"""
Failure Analysis: T250 Validation Prediction Scatter Heatmap.

This script is a specialized version of `plot_validation_failure_map.py`
_specifically for analyzing T250 validation runs_.

Workflow:
1.  Reads a `validation_results_*.json` summary from a specified T250 validation
    experiment directory.
2.  From the summary, it identifies the model and dataset used.
3.  It loads the specified validation dataset and the trained model from the
    T250 experiment directories.
4.  It re-runs predictions for every sample in the validation set to get the
    per-sample prediction error.
5.  It generates a scatter plot of all predicted coordinates, colored by error
    percentile (deciles) from green (best) to red (worst).
6.  A legend provides the mean error for each decile.

The output plot is saved to `experiments/t250_cv_full/analysis/failure_analysis/`,
providing a visual map of where the T250 model performs well and where it fails.

Usage:
    # This script requires a hardcoded path to a T250 validation run directory.
    # Update the `validation_dir` variable in the `main` function.
    python scripts/failure_analysis/plot_validation_failure_map_t250.py
"""

import json
import sys
from pathlib import Path
from typing import Tuple, List

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so imports like 'src.*' resolve
sys.path.append(str(Path(__file__).parent.parent.parent))

# Local imports
from src.models.wave_source_resnet import create_wave_source_model
from src.common.normalization import load_training_stats
from src.common.paths import get_experiments_dir, get_data_dir


def read_validation_summary(validation_dir: Path) -> dict:
    """Load validation_results_*.json from the validation directory."""
    candidates = list(validation_dir.glob("validation_results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No validation_results_*.json in {validation_dir}")
    with open(candidates[0], "r") as f:
        return json.load(f)


def resolve_model_path(t250_root: Path, model_name: str) -> Path:
    """Find the trained model file under experiments/t250_cv_full/data/models."""
    models_dir = t250_root / "data" / "models"
    candidate = models_dir / model_name
    if candidate.exists():
        return candidate
    # Try with possible extensions variations
    for ext in ("", ".pth", ".pt"):
        cand = models_dir / f"{model_name}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Model not found under {models_dir}: {model_name}")


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_wave_source_model(grid_size=128)
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def get_coords_key(f: h5py.File) -> str:
    if "source_coords" in f:
        return "source_coords"
    if "coordinates" in f:
        return "coordinates"
    raise KeyError("No 'source_coords' or 'coordinates' found in HDF5")


def run_validation_predictions(
    model: torch.nn.Module,
    dataset_path: Path,
    wave_mean: float,
    wave_std: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds [N,2], targets [N,2], errors [N])."""
    with h5py.File(str(dataset_path), "r") as f:
        waves = f["wave_fields"]  # [N, H, W]
        coords_key = get_coords_key(f)
        targets = np.asarray(f[coords_key], dtype=np.float32)  # [N, 2]
        n = waves.shape[0]
        preds = np.zeros_like(targets)
        for i in range(n):
            wf = np.asarray(waves[i], dtype=np.float32)
            # Normalize with training stats
            wf_norm = (wf - wave_mean) / wave_std
            inp = torch.from_numpy(wf_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)[0].cpu().numpy()
            preds[i] = out
        errors = np.sqrt(((preds - targets) ** 2).sum(axis=1))
        return preds, targets, errors


def color_for_percentile(p: float) -> Tuple[float, float, float]:
    """Green (best) → Red (worst) over deciles. p in [0,1]."""
    # Simple linear blend from green to red
    r = p
    g = 1.0 - p
    b = 0.0
    return (r, g, b)


def plot_failure_map(
    preds: np.ndarray,
    targets: np.ndarray,
    errors: np.ndarray,
    grid_size: int,
    out_path: Path,
    title: str,
) -> None:
    """Scatter all predicted points on grid, colored by error percentile deciles."""
    n = len(errors)
    # Percentile rank per sample
    ranks = errors.argsort().argsort().astype(np.float32)
    percentiles = ranks / max(1, n - 1)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect("equal")
    ax.set_facecolor("lightgray")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Draw predictions
    for (x, y), p in zip(preds, percentiles):
        c = color_for_percentile(float(p))
        ax.plot(
            x,
            y,
            "o",
            markersize=4,
            markerfacecolor=c,
            markeredgecolor="black",
            markeredgewidth=0.2,
            alpha=0.9,
        )

    # Legend like color bar (deciles) with mean error for each group
    from matplotlib.patches import Patch

    patches: List[Patch] = []
    for d in range(10):
        c = color_for_percentile(d / 9.0)
        # Calculate mean error for this decile
        decile_start = d / 10.0
        decile_end = (d + 1) / 10.0
        decile_mask = (percentiles >= decile_start) & (percentiles < decile_end)
        if d == 9:  # Last decile includes the maximum percentile
            decile_mask = (percentiles >= decile_start) & (percentiles <= decile_end)
        decile_errors = errors[decile_mask]
        mean_error = decile_errors.mean() if len(decile_errors) > 0 else 0.0
        label = f"{d*10}-{(d+1)*10}% (μ={mean_error:.1f}px)"
        patches.append(Patch(color=c, label=label))
    ax.legend(
        handles=patches,
        title="Error percentile",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # --- Input Configuration ---
    # NOTE: This path must be manually updated to point to a specific T250 validation run.
    validation_run_name = "validation_20250723_2352"

    t250_root = get_experiments_dir() / "t250_cv_full"
    validation_dir = t250_root / "extra_validation" / validation_run_name
    analysis_dir = t250_root / "analysis" / "failure_analysis"

    print(f"Analyzing T250 validation run: {validation_dir}")

    # --- Main Script ---
    summary = read_validation_summary(validation_dir)
    # The dataset path in the summary might be relative; ensure it's rooted correctly.
    dataset_name = Path(
        summary.get("dataset_info", {}).get(
            "dataset_path", "wave_dataset_T250_validation.h5"
        )
    ).name
    dataset_path = get_data_dir() / dataset_name

    model_name = summary.get("model_info", {}).get("model_name", "")
    if not model_name:
        raise ValueError("model_name missing in validation summary")

    # Resolve TRAINING normalization stats by explicitly using the 'T250' tag
    wave_mean, wave_std = load_training_stats("T250")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model_path(t250_root, model_name)
    model = load_model(model_path, device)

    # Predict
    preds, targets, errors = run_validation_predictions(
        model, dataset_path, wave_mean, wave_std, device
    )

    # Plot
    out_png = analysis_dir / f"failure_map_{validation_dir.name}.png"
    title = f"T250 Validation Failure Map ({validation_dir.name})\nMean error: {errors.mean():.2f} px"
    plot_failure_map(
        preds, targets, errors, grid_size=128, out_path=out_png, title=title
    )

    print(f"✅ T250 Failure map saved: {out_png}")


if __name__ == "__main__":
    main()
