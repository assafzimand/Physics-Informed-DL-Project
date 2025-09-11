#!/usr/bin/env python3
"""
Interactively Run Activation Maximization for a Chosen Layer.

This script provides a command-line interface to perform activation maximization
on a specific convolutional layer of the best-trained model.

Workflow:
1.  Loads the best model from the main cross-validation experiment directory.
2.  Lists all available `Conv2D` layers and prompts the user to select one.
3.  Loads a reference sample from the analysis dataset.
4.  Ranks the filters in the chosen layer by activation strength for that sample.
5.  Runs and saves a comprehensive activation maximization analysis for each of
    the top-K filters (or a specific filter if provided via arguments).

This is useful for targeted, iterative exploration of what different parts of
the network have learned.

Usage:
    # Run with defaults: prompts for layer, then optimizes top 5 filters
    python scripts/activation_maximization/simple_test.py

    # Specify optimization parameters
    python scripts/activation_maximization/simple_test.py --iterations 1000 --learning_rate 0.02

    # Run on a specific filter (e.g., filter 12) and skip the ranking step
    python scripts/activation_maximization/simple_test.py --filter_idx 12
"""

import sys
from pathlib import Path
import argparse
import torch
from typing import List

# Ensure project root is on sys.path so imports like 'src.*' resolve
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.activation_maximization.simple_activation_max import SimpleActivationMaximizer
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model
from src.data.wave_dataset import WaveDataset
from src.common.paths import get_data_dir, get_experiments_dir


def get_top_active_filters(
    model: torch.nn.Module,
    device: torch.device,
    layer: torch.nn.Module,
    sample_tensor: torch.Tensor,
    top_k: int = 5,
    activation_mode: str = "post_relu_mean",
) -> List[int]:
    """
    Find the top-K most active filters in a given layer for a specific sample.

    Args:
        model: The trained model.
        device: Torch device.
        layer: The target layer module.
        sample_tensor: Input sample tensor (1, 1, H, W), already normalized.
        top_k: Number of top filters to return.
        activation_mode: Ranking activation mode (pre- or post-ReLU).

    Returns:
        List of filter indices sorted by activation strength (descending).
    """
    print(f"Finding top {top_k} active filters...")

    activations = {}

    def hook_fn(module, input, output):
        activations["target"] = output.detach()

    hook = layer.register_forward_hook(hook_fn)

    try:
        model.eval()
        with torch.no_grad():
            _ = model(sample_tensor)

        layer_output = activations["target"]  # Shape: (1, C, H, W)
        if activation_mode == "mean_abs":
            filter_activations = layer_output.abs().mean(dim=(0, 2, 3))
        elif activation_mode == "mean":
            filter_activations = layer_output.mean(dim=(0, 2, 3))
        elif activation_mode == "l2":
            filter_activations = (layer_output.pow(2).sum(dim=(2, 3)).sqrt()).mean(
                dim=0
            )
        elif activation_mode == "post_relu_mean":
            filter_activations = torch.relu(layer_output).mean(dim=(0, 2, 3))
        else:
            raise ValueError(
                f"Unsupported activation_mode for ranking: {activation_mode}"
            )

        top_indices = (
            torch.argsort(filter_activations, descending=True)[:top_k].cpu().numpy()
        )
        top_values = filter_activations[top_indices].cpu().numpy()

        print(f"Top {top_k} filters:")
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            print(f"  {i+1}. Filter {int(idx)}: {float(val):.3f}")

        return [int(i) for i in top_indices.tolist()]

    finally:
        hook.remove()


def main():
    """Run interactive activation maximization analysis."""

    print("INTERACTIVE ACTIVATION MAXIMIZATION ANALYSIS")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Interactive AM test runner")
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of optimization steps per filter",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--tv_reg",
        type=float,
        default=1e-4,
        help="Total Variation regularization weight",
    )
    parser.add_argument(
        "--l2_reg", type=float, default=0.0, help="L2 regularization weight on input"
    )
    parser.add_argument(
        "--activation_mode",
        type=str,
        default="mean_abs",
        choices=["mean_abs", "mean", "l2", "post_relu_mean"],
        help="Activation measure for ranking and AM",
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0, help="Index of analysis sample to use"
    )
    parser.add_argument(
        "--filter_idx",
        type=int,
        default=-1,
        help="If >=0, optimize this specific filter and skip ranking",
    )
    args = parser.parse_args()

    # --- 1. Load Model ---
    cv_results_path = get_experiments_dir() / "cv_full"
    model_info = find_best_cv_model(cv_results_path)
    if model_info is None:
        print(
            f"ERROR: No best model found in {cv_results_path}. Ensure CV results exist."
        )
        return

    fold_id, error, model_path = model_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_wave_source_model(grid_size=128)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    print(f"Loaded best model from fold {fold_id} (error: {error:.2f}px)")

    # --- 2. Select a Layer ---
    print("\nAvailable Conv2D layers:")
    conv_layers = []
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((i, name, module))
            print(
                f"  {len(conv_layers)-1}: Layer {i} - {name} ({module.out_channels} filters)"
            )

    # Get user input for layer
    while True:
        try:
            layer_choice = int(input(f"Choose layer (0-{len(conv_layers)-1}): "))
            if 0 <= layer_choice < len(conv_layers):
                break
            else:
                print(f"Please choose a number between 0 and {len(conv_layers)-1}")
        except ValueError:
            print("Please enter a valid number")

    layer_idx_in_model, layer_name, target_layer = conv_layers[layer_choice]
    print(f"Selected: Layer {layer_idx_in_model} - {layer_name}")

    # Create a structured save directory for this layer's results
    layer_save_dir = (
        get_experiments_dir()
        / "activation_maximization"
        / "interactive"
        / f"layer_{layer_idx_in_model}_{layer_name.replace('.', '_')}"
    )
    layer_save_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Load Reference Sample ---
    analysis_dataset_path = get_data_dir() / "wave_dataset_analysis_20samples.h5"
    if not analysis_dataset_path.exists():
        print(f"ERROR: Analysis dataset not found at {analysis_dataset_path}")
        return

    raw_dataset = WaveDataset(analysis_dataset_path, normalize_wave_fields=False)

    sample_idx = args.sample_idx
    raw_wave_field, coordinates = raw_dataset[sample_idx]
    raw_wave_field = raw_wave_field.to(device)

    # Normalize the raw sample with the model's training stats to prepare for filter ranking
    norm_resolver = SimpleActivationMaximizer(model, device, model_path=str(model_path))
    wave_mean, wave_std = norm_resolver.wave_mean, norm_resolver.wave_std
    normalized_sample_tensor = (raw_wave_field - wave_mean) / wave_std

    print(
        f"\nUsing sample {sample_idx} from {analysis_dataset_path.name} as reference."
    )
    print(f"  - True coordinates: x={coordinates[0]:.1f}, y={coordinates[1]:.1f}")

    # --- 4. Rank or Select Filters ---
    if args.filter_idx >= 0:
        filters_to_optimize = [args.filter_idx]
        print(
            f"Using explicit filter index: {filters_to_optimize[0]} (skipping ranking)"
        )
    else:
        filters_to_optimize = get_top_active_filters(
            model,
            device,
            target_layer,
            normalized_sample_tensor,
            top_k=5,
            activation_mode=args.activation_mode,
        )

    # --- 5. Run Optimization ---
    maximizer = SimpleActivationMaximizer(model, device, model_path=str(model_path))
    maximizer.register_hook("target_layer", target_layer)

    try:
        print(f"\nOptimizing {len(filters_to_optimize)} filter(s)...")
        print(f"  - Saving all results to: {layer_save_dir}")

        for i, filter_idx in enumerate(filters_to_optimize):
            print(
                f"\nOptimizing filter {filter_idx} ({i+1}/{len(filters_to_optimize)})..."
            )
            # Use the raw, unnormalized wave field as the consistent starting point
            init_tensor = raw_wave_field.clone().detach()

            _ = maximizer.optimize_filter(
                layer_name="target_layer",
                filter_idx=filter_idx,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                use_real_data_init=False,  # Overridden by init_tensor
                init_tensor=init_tensor,
                save_dir=layer_save_dir,
                tv_reg=args.tv_reg,
                l2_reg=args.l2_reg,
                activation_mode=args.activation_mode,
            )

        print(f"\nAll optimizations complete. Check {layer_save_dir}/ for plots.")

    finally:
        maximizer.cleanup_hooks()


if __name__ == "__main__":
    main()
