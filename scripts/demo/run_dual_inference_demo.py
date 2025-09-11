#!/usr/bin/env python3
"""
Run a dual inference demo: automatically run three random samples for T250 and T500,
using models in models/best_T250.pth and models/best_T500.pth, with correct
training normalization and a single figure containing both results.

Usage (Windows):
  .\venv\Scripts\python.exe scripts\demo\run_dual_inference_demo.py
"""

import sys
from pathlib import Path
import random

# Ensure project root on path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.common.paths import get_data_dir
from src.common.normalization import (
    load_training_stats,
    infer_dataset_tag,
)
from src.inference.inference import WaveSourceInference


def pick_random_indices(h5_path: Path, num_samples: int) -> list[int]:
    with h5py.File(h5_path, "r") as f:
        wave_key = "wave_fields" if "wave_fields" in f else "inputs"
        total = len(f[wave_key])
    num = min(num_samples, total)
    return random.sample(range(total), k=num)


def load_sample_by_index(h5_path: Path, idx: int):
    with h5py.File(h5_path, "r") as f:
        wave_key = "wave_fields" if "wave_fields" in f else "inputs"
        coords_key = (
            "source_coords"
            if "source_coords" in f
            else ("coordinates" if "coordinates" in f else None)
        )
        x = f[wave_key][idx][...]
        y = f[coords_key][idx][...] if coords_key is not None else None
    return np.asarray(x).squeeze(), (np.asarray(y) if y is not None else None)


def resolve_dataset_path(tag: str) -> Path | None:
    # Prefer validation dataset under data/
    val_name = f"wave_dataset_{tag}_validation.h5"
    val_path = get_data_dir() / val_name
    if val_path.exists():
        return val_path

    # Fallback to tag-specific analysis under data/<TAG>/analysis.h5
    data_tag = get_data_dir() / tag / "analysis.h5"
    if data_tag.exists():
        return data_tag

    # Final fallback: any file under data/ containing the tag
    tag_lower = tag.lower()
    generic_candidates = list(get_data_dir().glob(f"**/*{tag_lower}*.h5"))
    if generic_candidates:
        return generic_candidates[0]

    return None


def run_dual(num_samples: int = 3):
    tasks = [
        ("T250", Path("models") / "best_T250.pth"),
        ("T500", Path("models") / "best_T500.pth"),
    ]

    # Create single figure: rows = num_samples, cols = 2 (T250, T500)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for col, (tag, model_path) in enumerate(tasks):
        if not model_path.exists():
            for row in range(num_samples):
                ax = axes[row, col]
                ax.set_title(f"{tag}: missing model")
                ax.axis("off")
            print(f"[{tag}] Missing model: {model_path}")
            continue

        ds_path = resolve_dataset_path(tag)
        if ds_path is None:
            for row in range(num_samples):
                ax = axes[row, col]
                ax.set_title(f"{tag}: missing dataset")
                ax.axis("off")
            print(f"[{tag}] Missing dataset for tag")
            continue

        # Minimal logs
        print(f"[{tag}] model:   {model_path}")
        print(f"[{tag}] dataset: {ds_path}")

        # Remove checkpoint tag mismatch warning; we always use explicit tag
        # model_tag = infer_dataset_tag(str(model_path))
        # if model_tag != tag:
        #     print(f"[{tag}] WARN: checkpoint tag inferred as {model_tag}; using {tag}.")

        # Load training normalization (log) and inference helper (normalizes internally)
        mean, std = load_training_stats(tag)
        print(f"[{tag}] norm: mean={mean:.6f}, std={std:.6f}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        infer = WaveSourceInference(model_path, device=device, dataset_tag=tag)

        # Choose indices and predict
        indices = pick_random_indices(ds_path, num_samples)
        for row, idx in enumerate(indices):
            x, y = load_sample_by_index(ds_path, idx)
            # If GT coords appear normalized [0,1], rescale to pixels (grid_size-1)
            if y is not None and float(np.max(y)) <= 2.0:
                grid_size = getattr(infer.model, "grid_size", 128)
                y = y * (grid_size - 1)

            px, py = infer.predict_source(x)
            err = (
                float(np.sqrt((y[0] - px) ** 2 + (y[1] - py) ** 2)) if y is not None else 0.0
            )

            ax = axes[row, col]
            ax.imshow(x, cmap="RdBu_r", origin="lower")
            if y is not None:
                ax.plot(y[0], y[1], "yo", markersize=8, markeredgecolor="k", label="GT")
            ax.plot(px, py, "g^", markersize=10, markeredgecolor="w", label="Pred")
            ax.set_title(f"{tag}: sample={idx}  err={err:.2f}px")
            ax.legend(loc="upper right")
            ax.axis("off")

            print(f"[{tag}] sample={idx} true={tuple(y) if y is not None else None} pred=({px:.2f},{py:.2f}) err={err:.2f}px")

    plt.tight_layout()
    plt.show()


def main():
    run_dual(num_samples=3)


if __name__ == "__main__":
    main()
