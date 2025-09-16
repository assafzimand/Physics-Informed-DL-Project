#!/usr/bin/env python3
"""
Run Comprehensive Activation Maximization Analysis on All Conv2D Layers.

This script automates the process of analyzing a trained wave source localization
model. It performs the following steps:
1.  Loads the best-performing model from a cross-validation experiment directory.
2.  Identifies all `Conv2D` layers within the model.
3.  For each layer, it uses a sample from a reference dataset to find the `TOP_K`
    most strongly activating filters.
4.  It then runs activation maximization for each of these top filters to generate
    the input pattern that maximally excites it.
5.  All outputs, including detailed plots and statistics for each filter, are
    saved to a structured directory under `experiments/activation_maximization/comprehensive/`.

This is intended as a long-running analysis script to generate a complete
"atlas" of the features learned by the model's convolutional layers.

Usage:
    python scripts/activation_maximization/comprehensive_all_layers.py
"""

import sys
from pathlib import Path
import torch
from datetime import datetime, timedelta
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
    top_k: int = 10,
    activation_mode: str = "post_relu_mean",
) -> List[int]:
    """
    Find the top-K most active filters in a given layer for a specific sample.

    Args:
        model: The trained model.
        device: Torch device.
        layer: The target layer module.
        sample_tensor: Input sample tensor (1, 1, H, W), normalized.
        top_k: Number of top filters to return.
        activation_mode: Activation mode for ranking.

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

        layer_output = activations["target"]  # (1, C, H, W)
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
            print(f"  {i+1:2d}. Filter {int(idx):2d}: {float(val):.3f}")

        return [int(i) for i in top_indices.tolist()]

    except Exception as e:
        print(f"Error finding active filters: {e}")
        return []

    finally:
        hook.remove()


def estimate_total_time(
    num_layers: int, filters_per_layer: int, iterations_per_filter: int
) -> None:
    """Estimate total processing time based on rough per-iteration cost."""
    total_iterations = num_layers * filters_per_layer * iterations_per_filter
    estimated_seconds = total_iterations * 0.5  # empirical
    estimated_time = timedelta(seconds=estimated_seconds)

    print("ANALYSIS SCOPE:")
    print(f"   - Layers to process: {num_layers}")
    print(f"   - Filters per layer: {filters_per_layer}")
    print(f"   - Iterations per filter: {iterations_per_filter}")
    print(f"   - Total iterations: {total_iterations:,}")
    print(f"   - Estimated time: {estimated_time}")
    print(f"   - Expected completion: {datetime.now() + estimated_time}")


def main():
    """Run comprehensive activation maximization analysis on all layers."""

    print("COMPREHENSIVE ACTIVATION MAXIMIZATION - ALL LAYERS")
    print("=" * 80)

    start_time = datetime.now()

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
    print(f"Using device: {device}")

    # --- 2. Get Conv2D layers (exclude skip/projection convs) ---
    print("\nAvailable Conv2D layers (main path only):")
    conv_layers = []
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            name_has_skip = "skip_connection" in name
            is_projection_1x1 = getattr(module, "kernel_size", None) == (1, 1)
            if name_has_skip or is_projection_1x1:
                continue
            conv_layers.append((i, name, module))
            print(
                f"  {len(conv_layers)-1:2d}: Layer {i:2d} - {name} ({module.out_channels} filters)"
            )

    # --- 3. Analysis Parameters ---
    TOP_K = 10
    ITERATIONS = 500
    LEARNING_RATE = 0.01
    ACTIVATION_MODE = "mean_abs"  # Metric for ranking filters

    print("\nTIME ESTIMATION:")
    estimate_total_time(len(conv_layers), TOP_K, ITERATIONS)

    # --- 4. Load Reference Sample ---
    # The reference sample is used to rank filters and as a consistent initialization
    analysis_dataset_path = get_data_dir() / "wave_dataset_analysis_20samples.h5"
    if not analysis_dataset_path.exists():
        print(f"ERROR: Analysis dataset not found at {analysis_dataset_path}")
        return

    dataset = WaveDataset(analysis_dataset_path, normalize_wave_fields=False)
    sample_idx = 0  # Use the first sample for consistency
    wave_field, coordinates = dataset[sample_idx]
    sample_tensor_raw = wave_field.to(device)

    # This raw tensor will be the consistent starting point for all optimizations
    init_tensor = sample_tensor_raw.clone().detach()

    print(
        f"\nUsing sample {sample_idx} from {analysis_dataset_path.name} as reference."
    )
    print(f"  - True coordinates: x={coordinates[0]:.1f}, y={coordinates[1]:.1f}")
    print("  - All filter optimizations will start from this SAME initial pattern.")
    print(
        f"  - Init tensor stats (RAW): mean={init_tensor.mean():.6f}, std={init_tensor.std():.6f}"
    )

    # --- 5. Main Analysis Loop ---
    total_layers = len(conv_layers)
    successful_layers = 0
    failed_layers = 0
    total_filters_processed = 0

    print(f"\nSTARTING ANALYSIS OF {total_layers} LAYERS")
    print("=" * 80)

    for layer_num, (layer_idx_in_model, layer_name, target_layer) in enumerate(
        conv_layers
    ):
        layer_start_time = datetime.now()

        print(
            f"\nLAYER {layer_num + 1}/{total_layers}: Layer {layer_idx_in_model} - {layer_name}"
        )
        print(f"   ({target_layer.out_channels} filters)")

        # Define a structured save directory for this layer's results
        layer_save_dir = (
            get_experiments_dir()
            / "activation_maximization"
            / "comprehensive"
            / f"layer_{layer_idx_in_model}_{layer_name.replace('.', '_')}"
        )
        layer_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # First, determine the correct normalization stats from the model
            norm_resolver = SimpleActivationMaximizer(
                model, device, model_path=str(model_path)
            )
            wave_mean, wave_std = norm_resolver.wave_mean, norm_resolver.wave_std

            # Normalize the raw sample to create the input for ranking filters
            normalized_sample = (sample_tensor_raw - wave_mean) / wave_std

            # Rank filters in the current layer based on the normalized sample
            top_filters = get_top_active_filters(
                model,
                device,
                target_layer,
                normalized_sample,
                top_k=TOP_K,
                activation_mode=ACTIVATION_MODE,
            )

            if not top_filters:
                print(f"No active filters found for layer {layer_idx_in_model}")
                failed_layers += 1
                continue

            # Create a fresh maximizer for this layer to run optimizations
            maximizer = SimpleActivationMaximizer(
                model, device, model_path=str(model_path)
            )
            maximizer.register_hook("target_layer", target_layer)

            try:
                print(f"Optimizing top {len(top_filters)} filters...")
                print(f"  - Saving to: {layer_save_dir}")
                print(
                    "  - All filters will use the SAME initialization tensor (from raw sample)."
                )

                layer_successful_filters = 0

                for i, filter_idx in enumerate(top_filters):
                    filter_start_time = datetime.now()

                    print(
                        f"  Filter {filter_idx} ({i+1}/{len(top_filters)}) [Layer {layer_num+1}/{total_layers}]"
                    )

                    try:
                        results = maximizer.optimize_filter(
                            layer_name="target_layer",
                            filter_idx=filter_idx,
                            iterations=ITERATIONS,
                            learning_rate=LEARNING_RATE,
                            use_real_data_init=False,  # Overridden by init_tensor
                            init_tensor=init_tensor,
                            save_dir=layer_save_dir,
                            activation_mode=ACTIVATION_MODE,
                        )

                        final_activation = results["config"]["final_activation"]
                        filter_time = datetime.now() - filter_start_time
                        print(
                            f"     -> Final activation: {final_activation:.2f} (took {filter_time.total_seconds():.1f}s)"
                        )

                        layer_successful_filters += 1
                        total_filters_processed += 1

                    except Exception as e:
                        print(f"     -> Filter {filter_idx} FAILED: {str(e)[:100]}")
                        continue

                layer_time = datetime.now() - layer_start_time
                print(
                    f"\nLayer {layer_idx_in_model} complete: {layer_successful_filters}/{len(top_filters)} filters successful"
                )
                print(f"   Layer time: {layer_time}")

                if layer_successful_filters > 0:
                    successful_layers += 1
                else:
                    failed_layers += 1

            finally:
                maximizer.cleanup_hooks()

        except Exception as e:
            print(f"Layer {layer_idx_in_model} FAILED completely: {e}")
            failed_layers += 1
            continue

        # --- Progress Update ---
        elapsed_time = datetime.now() - start_time
        remaining_layers = total_layers - (layer_num + 1)
        avg_time_per_layer = elapsed_time / (layer_num + 1)
        estimated_remaining = avg_time_per_layer * remaining_layers

        print("\nPROGRESS UPDATE:")
        print(f"   - Layers completed: {layer_num + 1}/{total_layers}")
        print(f"   - Successful layers: {successful_layers}")
        print(f"   - Failed layers: {failed_layers}")
        print(f"   - Total filters processed: {total_filters_processed}")
        print(f"   - Elapsed time: {elapsed_time}")
        print(f"   - Estimated remaining: {estimated_remaining}")
        print(f"   - Expected completion: {datetime.now() + estimated_remaining}")

    total_time = datetime.now() - start_time

    print("\nANALYSIS COMPLETE")
    print("=" * 80)
    print("FINAL SUMMARY:")
    print(f"   - Total layers processed: {total_layers}")
    print(f"   - Successful layers: {successful_layers}")
    print(f"   - Failed layers: {failed_layers}")
    print(f"   - Total filters processed: {total_filters_processed}")
    print(f"   - Total time: {total_time}")
    print(
        f"   - Average time per layer: {total_time / total_layers if total_layers > 0 else 'N/A'}"
    )

    final_save_dir = get_experiments_dir() / "activation_maximization" / "comprehensive"
    print(f"   - Results saved in layer-specific folders under: {final_save_dir}")


if __name__ == "__main__":
    main()
