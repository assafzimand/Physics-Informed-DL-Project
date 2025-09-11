#!/usr/bin/env python3
"""
Multi-Sample Prediction Visualization Script for T250 Model.

This script generates a detailed, multi-panel visualization comparing the
predictions of the best-trained T250 model against the ground truth for a
selection of samples from the T250 validation dataset.

Workflow:
1.  Finds the best-performing model from the T250 cross-validation experiment
    directory (`experiments/t250_cv_full`).
2.  Loads this model and the T250 validation dataset. Crucially, it loads the
    dataset with normalization disabled and then applies the correct TRAINING
    normalization statistics before feeding samples to the model.
3.  Prompts the user to select which samples to visualize (e.g., first 5,
    a specific list, or a random selection).
4.  For each selected sample, it generates a 3-panel plot:
    -   The raw wave field.
    -   The wave field with true and predicted source locations overlaid.
    -   A coordinate-space plot with detailed error metrics.
5.  The final composite plot is saved to a timestamped directory under
    `experiments/t250_visualization/`.

This tool is useful for qualitative assessment of the model's performance on
specific examples.

Usage:
    python scripts/visualization/multi_sample_predictions.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from datetime import datetime
from typing import List, Optional

# Ensure project root on sys.path for 'src.*' imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.wave_dataset import WaveDataset
from src.models.wave_source_resnet import create_wave_source_model
from src.common.normalization import load_training_stats
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.common.paths import get_experiments_dir, get_data_dir


def load_model_and_dataset():
    """Load the best T250 model and the T250 validation dataset (RAW)."""

    # --- Find Best Model ---
    t250_experiment_dir = get_experiments_dir() / "t250_cv_full"
    print(f"Searching for best T250 CV model in: {t250_experiment_dir}")
    model_info = find_best_cv_model(t250_experiment_dir)
    if not model_info:
        print(f"ERROR: No best model found in {t250_experiment_dir}. Cannot proceed.")
        return None, None, None, None, None

    fold_id, error, model_path = model_info
    print(f"✅ Found best model from fold {fold_id} (error: {error:.2f}px)")

    # --- Load Dataset ---
    dataset_path = get_data_dir() / "wave_dataset_T250_validation.h5"
    if not dataset_path.exists():
        print(f"ERROR: T250 validation dataset not found: {dataset_path}")
        return None, None, None, None, None

    print(f"Loading T250 validation dataset: {dataset_path}")
    try:
        # Load RAW dataset; we will normalize manually with TRAINING stats
        dataset = WaveDataset(str(dataset_path), normalize_wave_fields=False)
        print(f"  - Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return None, None, None, None, None

    # --- Load Model and Normalization Stats ---
    print(f"Loading T250 model: {model_path.name}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        model = create_wave_source_model(grid_size=128)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        # Load TRAINING normalization stats for T250 to ensure correct inference
        wave_mean, wave_std = load_training_stats("T250")
        print(
            f"  - Loaded T250 training normalization stats (mean={wave_mean:.4f}, std={wave_std:.4f})"
        )

        return model, dataset, device, wave_mean, wave_std

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None, None


def create_multi_sample_prediction_plot(
    model: torch.nn.Module,
    dataset: WaveDataset,
    device: torch.device,
    wave_mean: float,
    wave_std: float,
    sample_indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> List[float]:
    """Create a comprehensive multi-sample prediction visualization."""
    print("Creating multi-sample prediction visualization...")

    if sample_indices is None:
        # Default to first 5 samples if none are provided
        sample_indices = [0, 1, 2, 3, 4]

    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    all_errors: List[float] = []
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            raw_wave_data, target_coords = dataset[sample_idx]  # RAW wave field
            target_coords = target_coords.numpy()

            # Manually normalize with TRAINING stats before inference
            wf_raw = raw_wave_data.numpy()
            wf_norm = (wf_raw - wave_mean) / wave_std
            wave_input = (
                torch.from_numpy(wf_norm).float().unsqueeze(0).to(device)
            )  # Shape: [1,1,H,W]

            pred_coords = model(wave_input).cpu().numpy().squeeze()

            error = float(np.sqrt(np.sum((pred_coords - target_coords) ** 2)))
            all_errors.append(error)

            wave_field_to_plot = raw_wave_data.squeeze().numpy()

            # --- Plotting Panels ---
            # Panel 1: Raw Wave Field
            im1 = axes[i, 0].imshow(
                wave_field_to_plot,
                cmap="RdBu_r",
                origin="lower",
                extent=[0, 127, 0, 127],
            )
            axes[i, 0].set_title(f"Sample {sample_idx}\nWave Field (Raw)")
            axes[i, 0].set_xlabel("X Position")
            axes[i, 0].set_ylabel("Y Position")

            # Panel 2: Prediction Overlay
            im2 = axes[i, 1].imshow(
                wave_field_to_plot,
                cmap="RdBu_r",
                origin="lower",
                extent=[0, 127, 0, 127],
            )
            axes[i, 1].plot(
                target_coords[0],
                target_coords[1],
                "x",
                markersize=12,
                markeredgewidth=3,
                color="lime",
                label="True",
            )
            axes[i, 1].plot(
                pred_coords[0],
                pred_coords[1],
                "o",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="yellow",
                markeredgewidth=3,
                label="Predicted",
            )
            axes[i, 1].set_title(f"Prediction Comparison\nError: {error:.2f} px")
            axes[i, 1].set_xlabel("X Position")
            axes[i, 1].set_ylabel("Y Position")
            axes[i, 1].legend()

            # Panel 3: Coordinate Space and Stats
            axes[i, 2].set_xlim(-10, 140)
            axes[i, 2].set_ylim(-10, 140)
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].set_facecolor("lightgray")
            axes[i, 2].set_aspect("equal", adjustable="box")
            axes[i, 2].plot(
                target_coords[0],
                target_coords[1],
                "x",
                markersize=15,
                markeredgewidth=4,
                color="lime",
                label="True",
            )
            axes[i, 2].plot(
                pred_coords[0],
                pred_coords[1],
                "o",
                markersize=12,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=3,
                label="Predicted",
            )
            coord_text = (
                f"Sample {sample_idx}\n\n"
                f"True Position:\nx = {target_coords[0]:.1f}\ny = {target_coords[1]:.1f}\n\n"
                f"Predicted Position:\nx = {pred_coords[0]:.1f}\ny = {pred_coords[1]:.1f}\n\n"
                f"Error: {error:.2f} px\n\n"
                f"Individual Errors:\nΔx = {abs(pred_coords[0] - target_coords[0]):.2f}\n"
                f"Δy = {abs(pred_coords[1] - target_coords[1]):.2f}"
            )
            axes[i, 2].text(
                0.98,
                0.98,
                coord_text,
                transform=axes[i, 2].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                fontsize=9,
                fontfamily="monospace",
            )
            axes[i, 2].set_title("Coordinate Space\nΔx: ±, Δy: ±")
            axes[i, 2].set_xlabel("X Position")
            axes[i, 2].set_ylabel("Y Position")
            axes[i, 2].legend()

            if i == 0:
                plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

    mean_error = float(np.mean(all_errors))
    std_error = float(np.std(all_errors))
    fig.suptitle(
        f"T=250 Model Multi-Sample Predictions\nMean Error: {mean_error:.2f} ± {std_error:.2f} px | Samples: {num_samples}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Multi-sample prediction plot saved: {save_path}")
    plt.show()
    return all_errors


def main():
    print("T=250 Model Multi-Sample Prediction Visualization")
    print("=" * 60)

    model, dataset, device, wave_mean, wave_std = load_model_and_dataset()
    if model is None:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = (
        get_experiments_dir() / "t250_visualization" / f"multi_sample_{timestamp}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset has {len(dataset)} samples available.")
    print("Choose visualization option:")
    print("1. First 5 samples (default)")
    print("2. Specific sample indices")
    print("3. Random selection")

    choice = input("Enter choice (1-3, or press Enter for default): ").strip()

    sample_indices = None
    if choice == "2":
        indices_input = input(
            "Enter sample indices (comma-separated, e.g., 10,25,50,100,200): "
        )
        try:
            sample_indices = [int(x.strip()) for x in indices_input.split(",")]
            max_idx = len(dataset) - 1
            sample_indices = [idx for idx in sample_indices if 0 <= idx <= max_idx]
            print(f"Using samples: {sample_indices}")
        except Exception:
            print("Invalid input, using default...")
            sample_indices = None

    elif choice == "3":
        num_samples = input("How many random samples? (default 5): ").strip()
        try:
            num_samples = int(num_samples) if num_samples else 5
            sample_indices = np.random.choice(
                len(dataset), size=min(num_samples, len(dataset)), replace=False
            ).tolist()
            print(f"Using random samples: {sample_indices}")
        except Exception:
            print("Invalid input, using default...")
            sample_indices = None

    if sample_indices is None:
        sample_indices = [0, 1, 2, 3, 4]
        print(f"Using default samples: {sample_indices}")

    save_path = results_dir / f"multi_sample_predictions_{timestamp}.png"
    errors = create_multi_sample_prediction_plot(
        model, dataset, device, wave_mean, wave_std, sample_indices, save_path
    )

    print("\nVisualization completed.")
    print(f"  - Results saved to: {results_dir}")
    print(f"  - Sample errors: {[f'{e:.2f}' for e in errors]} px")


if __name__ == "__main__":
    main()
