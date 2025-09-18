#!/usr/bin/env python3
"""
Simple Activation Maximization for Single-Channel Wave Models

This implementation bypasses Lucent's RGB assumptions and works directly
with single-channel wave field inputs, ensuring proper gradient flow.

Enhanced with comprehensive monitoring, plotting, and debugging capabilities.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from src.common.normalization import infer_dataset_tag, load_training_stats
from src.common.paths import get_data_dir, get_experiments_dir


class SimpleActivationMaximizer:
    """
    Performs activation maximization on single-channel wave models.

    This class handles the optimization loop for generating input patterns
    that maximize the activation of a specific filter in a convolutional layer.
    It includes functionality for various activation objectives, regularization,
    and detailed monitoring.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        model_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Initializes the activation maximizer.

        Args:
            model: The PyTorch model to be analyzed.
            device: The computing device ('cpu' or 'cuda').
            model_path: Optional path to the model checkpoint for inferring normalization.
            dataset_name: Optional explicit dataset tag ('T250' or 'T500') to force normalization.
        """
        self.model = model.eval()
        self.device = torch.device(device)
        self.hooks = {}
        self.activations = {}
        # Normalization stats resolved from training dataset (T250/T500)
        self.dataset_name = self._determine_dataset_name(model_path, dataset_name)
        self.wave_mean, self.wave_std = self._load_training_normalization_stats(
            self.dataset_name
        )

    def register_hook(self, layer_name: str, module: torch.nn.Module):
        """
        Registers a forward hook on a target layer to capture its activation.

        Args:
            layer_name: A unique name to identify the layer's activation.
            module: The PyTorch module (layer) to attach the hook to.
        """

        def hook_fn(module, input, output):
            self.activations[layer_name] = output

        handle = module.register_forward_hook(hook_fn)
        self.hooks[layer_name] = handle

    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()

    def _determine_dataset_name(
        self, model_path: Optional[str], dataset_name: Optional[str]
    ) -> str:
        """Infers dataset name ('T250'/'T500') for normalization purposes."""
        # Priority 1: explicit dataset_name
        if dataset_name is not None:
            return dataset_name.upper()
        # Priority 2: Delegate to central utility
        if model_path is not None:
            return infer_dataset_tag(model_path)
        # Fallback: default to T500 but warn
        print(
            "WARNING: Could not determine dataset from model; defaulting to T500 normalization"
        )
        return "T500"

    def _load_training_normalization_stats(
        self, dataset_name: str
    ) -> Tuple[float, float]:
        """Delegates loading of normalization stats to the central utility."""
        return load_training_stats(dataset_name)

    def load_real_wave_samples(
        self, dataset_path: Optional[str | Path] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Loads real wave field samples from an HDF5 file for initialization.

        Args:
            dataset_path: Path to HDF5 file. Defaults to `data/wave_dataset_analysis_20samples.h5`.

        Returns:
            A tuple of (wave_fields, coordinates) as numpy arrays, or (None, None) on failure.
        """
        if dataset_path is None:
            dataset_path = get_data_dir() / "wave_dataset_analysis_20samples.h5"

        try:
            with h5py.File(dataset_path, "r") as f:
                if "wave_fields" in f and ("coordinates" in f or "source_coords" in f):
                    wave_fields = f["wave_fields"][:]

                    coords_key = (
                        "source_coords" if "source_coords" in f else "coordinates"
                    )
                    coordinates = f[coords_key][:]

                    # Remove channel dimension if present (e.g., shape [N, 1, H, W])
                    if wave_fields.ndim == 4 and wave_fields.shape[1] == 1:
                        wave_fields = wave_fields[:, 0]

                    return wave_fields, coordinates
                else:
                    print(
                        f"WARN: 'wave_fields' or 'coordinates' not found in {dataset_path}"
                    )
                    return None, None
        except Exception as e:
            print(f"ERROR: Failed to load real wave samples from {dataset_path}: {e}")
            return None, None

    def optimize_filter(
        self,
        layer_name: str,
        filter_idx: int,
        iterations: int = 512,
        learning_rate: float = 0.01,
        image_size: int = 128,
        use_real_data_init: bool = True,
        save_intermediate: bool = True,
        save_every: int = 100,
        save_dir: Optional[str] = None,
        init_tensor: Optional[torch.Tensor] = None,
        tv_reg: float = 0.0,
        l2_reg: float = 0.0,
        suppression_weight: float = 1.0,
        activation_mode: str = "post_relu_mean",
        output_filename_base: Optional[str] = None,
        ground_truth_xy: Optional[Tuple[float, float]] = None,
        rank_within_layer: Optional[int] = None,
        total_filters_in_layer: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs the activation maximization optimization loop for a single filter.

        This method sets up the optimization problem, including the input tensor,
        optimizer, and loss function components (activation, suppression, regularization).
        It iteratively updates the input to maximize the target filter's activation
        while providing detailed logging and collecting data for visualization.

        Args:
            layer_name: The registered name of the target layer.
            filter_idx: The index of the target filter within the layer.
            iterations: The number of optimization steps to perform.
            learning_rate: The learning rate for the Adam optimizer.
            image_size: The spatial dimension of the input tensor (e.g., 128 for a 128x128 input).
            use_real_data_init: If True, initializes from a real wave sample; otherwise, random noise.
            save_intermediate: If True, saves intermediate patterns and stats during the run.
            save_every: The frequency (in iterations) at which to save intermediate patterns.
            save_dir: The directory where output plots will be saved. Defaults to `experiments/am/comprehensive`.
            init_tensor: An optional pre-defined tensor to use for initialization, overriding other methods.
            tv_reg: The weight for the Total Variation (TV) regularization term for smoothness.
            l2_reg: The weight for the L2 regularization term on the input tensor to control amplitude.
            suppression_weight: The weight for the term that minimizes the activation of non-target filters.
            activation_mode: The method for calculating a filter's activation score. One of
                             {'mean_abs', 'mean', 'l2', 'post_relu_mean'}.

        Returns:
            A dictionary containing the comprehensive results of the optimization, including the
            best pattern found, initial pattern, monitoring data, final stats, and the path to the summary plot.
        """
        print("\n- Comprehensive Activation Maximization -")
        print("=" * 70)
        print(f"Target: {layer_name} filter {filter_idx}")
        print(f"Config: {iterations} iterations, LR={learning_rate}")
        print(f"Real data init: {'Yes' if use_real_data_init else 'No'}")
        print(f"Normalization: ALWAYS ON ({self.dataset_name})")
        if tv_reg > 0:
            print(f"Smoothness (TV Regularization): lambda={tv_reg}")
        if l2_reg > 0:
            print(f"Amplitude (L2 on input): lambda={l2_reg}")
        print(f"Activation measure: {activation_mode}")

        # Load real data if requested
        wave_samples = None
        if use_real_data_init and init_tensor is None:
            wave_samples, _ = self.load_real_wave_samples()
            if wave_samples is not None:
                print(f"Loaded {len(wave_samples)} real wave samples")
            else:
                print("Failed to load real data, using random initialization")

        # Initialize monitoring
        monitoring_data = {
            "iteration": [],
            "loss": [],
            "activation": [],
            "target_loss": [],
            "suppression_loss": [],
            "tv_loss": [],
            "l2_loss": [],
            "input_mean": [],
            "input_std": [],
            "input_min": [],
            "input_max": [],
            "grad_magnitude": [],
            "grad_mean": [],
            "grad_std": [],
            "intermediate_patterns": [],
        }

        # Initialize input tensor
        if init_tensor is not None:
            # Use provided initialization tensor
            print("Using provided initialization tensor")
            input_tensor = init_tensor.clone().detach().to(self.device)
            input_tensor.requires_grad_(True)
            initial_pattern = input_tensor.clone().detach()
            print(
                f"Init stats: mean={input_tensor.mean():.6f}, std={input_tensor.std():.6f}"
            )
        elif use_real_data_init and wave_samples is not None:
            # Use random real sample
            sample_idx = random.randint(0, len(wave_samples) - 1)
            initial_sample = wave_samples[sample_idx]
            print(f"Using real sample {sample_idx} as initialization")
            print(
                f"Sample stats: mean={initial_sample.mean():.6f}, std={initial_sample.std():.6f}"
            )

            input_tensor = torch.from_numpy(initial_sample).float()
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            input_tensor.requires_grad_(True)
            initial_pattern = input_tensor.clone().detach()
        else:
            # Random initialization
            print("Using random initialization")
            input_tensor = torch.randn(
                1, 1, image_size, image_size, requires_grad=True, device=self.device
            )
            initial_pattern = input_tensor.clone().detach()

        # Use training normalization constants resolved at init
        wave_mean = self.wave_mean
        wave_std = self.wave_std

        # Optimizer
        optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)

        # Tracking variables
        loss_history = []
        best_loss = float("inf")
        best_input = None

        print("\nStarting optimization...")

        # Main optimization loop
        for i in range(iterations):
            optimizer.zero_grad()

            # Forward pass with training normalization (always applied)
            model_input = (input_tensor - wave_mean) / wave_std
            print_norm_status = "NORMALIZED" if i == 0 else ""

            if print_norm_status:
                print(f"Input type: {print_norm_status}")

            # Forward pass
            _ = self.model(model_input)

            # Get full layer activation (all filters)
            layer_activation = self.activations[
                layer_name
            ]  # [batch, filters, height, width]
            target_activation = layer_activation[
                :, filter_idx
            ]  # [batch, height, width]

            # Compute per-filter activation scores according to activation_mode
            batch_size, num_filters = layer_activation.shape[:2]
            if activation_mode == "mean":
                filter_scores = layer_activation.mean(dim=(2, 3))
            elif activation_mode == "l2":
                flat = layer_activation.view(batch_size, num_filters, -1)
                filter_scores = torch.norm(flat, dim=2)
            elif activation_mode == "post_relu_mean":
                # Apply ReLU to pre-activation feature maps, then take mean
                filter_scores = F.relu(layer_activation).mean(dim=(2, 3))
            else:  # 'mean_abs' (default)
                filter_scores = layer_activation.abs().mean(dim=(2, 3))

            target_score = filter_scores[:, filter_idx]
            other_filter_mask = torch.ones(
                num_filters, dtype=torch.bool, device=layer_activation.device
            )
            other_filter_mask[filter_idx] = False
            other_scores = filter_scores[:, other_filter_mask]

            # Loss: maximize target filter activation, minimize other filters with weight
            target_loss = -target_score.mean()  # Negative because we want to maximize
            suppression_loss = (
                other_scores.mean()
            )  # Positive because we want to minimize

            # Total Variation (on the input image, not normalized)
            tv_loss = (
                self.total_variation_loss(input_tensor)
                if tv_reg > 0
                else torch.tensor(0.0, device=self.device)
            )
            l2_loss = (
                (input_tensor**2).mean()
                if l2_reg > 0
                else torch.tensor(0.0, device=self.device)
            )

            total_loss = (
                target_loss
                + suppression_weight * suppression_loss
                + tv_reg * tv_loss
                + l2_reg * l2_loss
            )

            # Backward pass
            total_loss.backward()

            # Collect comprehensive monitoring data
            with torch.no_grad():
                grad_mag = (
                    input_tensor.grad.norm().item()
                    if input_tensor.grad is not None
                    else 0.0
                )
                grad_mean = (
                    input_tensor.grad.mean().item()
                    if input_tensor.grad is not None
                    else 0.0
                )
                grad_std = (
                    input_tensor.grad.std().item()
                    if input_tensor.grad is not None
                    else 0.0
                )

                monitoring_data["iteration"].append(i)
                monitoring_data["loss"].append(total_loss.item())
                monitoring_data["activation"].append(
                    -target_loss.item()
                )  # Target activation proxy (positive)
                monitoring_data["target_loss"].append(
                    target_loss.item()
                )  # Target loss term (negative)
                monitoring_data["suppression_loss"].append(
                    suppression_loss.item()
                )  # Suppression loss term (positive)
                monitoring_data["tv_loss"].append(tv_loss.item())
                monitoring_data["l2_loss"].append(l2_loss.item())
                monitoring_data["input_mean"].append(input_tensor.mean().item())
                monitoring_data["input_std"].append(input_tensor.std().item())
                monitoring_data["input_min"].append(input_tensor.min().item())
                monitoring_data["input_max"].append(input_tensor.max().item())
                monitoring_data["grad_magnitude"].append(grad_mag)
                monitoring_data["grad_mean"].append(grad_mean)
                monitoring_data["grad_std"].append(grad_std)

                # Save intermediate patterns
                if save_intermediate and i % save_every == 0:
                    pattern_copy = input_tensor.clone().detach()
                    monitoring_data["intermediate_patterns"].append(
                        {
                            "iteration": i,
                            "pattern": pattern_copy,
                            "activation": -target_loss.item(),
                            "target_loss": target_loss.item(),
                            "suppression_loss": suppression_loss.item(),
                            "tv_loss": tv_loss.item(),
                            "l2_loss": l2_loss.item(),
                        }
                    )

            # Update best result
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_input = input_tensor.clone().detach()

            # Optimize
            optimizer.step()

            # Progress reporting
            if i % 10 == 0 or i == iterations - 1:
                print(
                    f"   Step {i:4d}: Loss={total_loss.item():.4f}, "
                    f"Target={-target_loss.item():.2f}, "
                    f"Suppress={suppression_loss.item():.2f}, "
                    f"TV={tv_loss.item():.4f} (lambda={tv_reg}), "
                    f"L2={l2_loss.item():.4f} (lambda={l2_reg}), "
                    f"GradMag={grad_mag:.6f}, "
                    f"InputStd={input_tensor.std().item():.4f}"
                )

        print("\nOptimization complete!")
        final_activation = monitoring_data["activation"][-1]
        final_target_loss = monitoring_data["target_loss"][-1]
        final_suppression_loss = monitoring_data["suppression_loss"][-1]
        final_tv_loss = monitoring_data["tv_loss"][-1]
        final_l2_loss = monitoring_data["l2_loss"][-1]
        loss_reduction = monitoring_data["loss"][0] - monitoring_data["loss"][-1]
        avg_grad_mag = np.mean(monitoring_data["grad_magnitude"])
        grad_variation = np.std(monitoring_data["grad_magnitude"])

        print("Final Results:")
        print(f"   Final Activation: {final_activation:.2f}")
        print(f"   Final Target Loss: {final_target_loss:.4f}")
        print(f"   Final Suppression Loss: {final_suppression_loss:.4f}")
        if tv_reg > 0:
            print(f"   Final TV Loss: {final_tv_loss:.4f}")
        if l2_reg > 0:
            print(f"   Final L2 Loss: {final_l2_loss:.4f}")
        print(f"   Loss Reduction: {loss_reduction:.4f}")
        print(f"   Average Gradient Magnitude: {avg_grad_mag:.6f}")
        print(f"   Gradient Variation (std): {grad_variation:.6f}")
        if grad_variation > 1e-6:
            print("   Gradients are varying (std > 1e-6)")
        else:
            print("   Gradients are constant (std <= 1e-6)")

        # Create comprehensive results dictionary
        results = {
            "best_pattern": best_input,
            "initial_pattern": initial_pattern,
            "monitoring_data": monitoring_data,
            "config": {
                "layer_name": layer_name,
                "filter_idx": filter_idx,
                "iterations": iterations,
                "learning_rate": learning_rate,
                "use_real_data_init": use_real_data_init,
                "dataset_name": self.dataset_name,
                "final_activation": final_activation,
                "final_target_loss": final_target_loss,
                "final_suppression_loss": final_suppression_loss,
                "final_tv_loss": final_tv_loss,
                "final_l2_loss": final_l2_loss,
                "tv_reg": tv_reg,
                "l2_reg": l2_reg,
                "suppression_weight": suppression_weight,
                "loss_reduction": loss_reduction,
                "grad_variation": grad_variation,
                "file_basename": output_filename_base,
                "ground_truth_xy": ground_truth_xy,
                "rank_within_layer": rank_within_layer,
                "total_filters_in_layer": total_filters_in_layer,
            },
        }

        # ALWAYS create comprehensive plots and analysis
        plot_path = self.create_comprehensive_plots(results, save_dir)
        results["plot_path"] = plot_path

        return results

    def create_comprehensive_plots(
        self, results: Dict[str, Any], save_dir: Optional[str] = None
    ) -> Path:
        """
        Generates and saves a comprehensive plot summarizing the optimization run.

        The plot includes:
        - Initial vs. final optimized patterns.
        - A timeline showing the evolution of the pattern.
        - Curves for loss components, activation, gradients, and input statistics.
        - A text summary of the final results and configuration.

        Args:
            results: The dictionary returned by `optimize_filter`.
            save_dir: The directory where the output plot will be saved.

        Returns:
            The `pathlib.Path` object pointing to the saved plot.
        """
        if save_dir is None:
            save_dir = (
                get_experiments_dir() / "activation_maximization" / "comprehensive"
            )
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        monitoring_data = results["monitoring_data"]
        config = results["config"]

        # Create main figure with multiple panels - 3 rows for 500 iterations
        fig = plt.figure(figsize=(20, 15))

        # Use the same training normalization used during optimization
        wave_mean = self.wave_mean
        wave_std = self.wave_std

        # Panel 1: Initial vs Final patterns (in model input space)
        ax1 = plt.subplot(3, 4, 1)
        initial_raw = results["initial_pattern"][0, 0].cpu().numpy()
        # Show normalized version that model actually sees
        initial_normalized = (initial_raw - wave_mean) / wave_std
        im1 = ax1.imshow(initial_normalized, cmap="RdBu_r", interpolation="nearest")
        ax1.set_title("Initial Pattern\n(Model Input Space)", fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])
        # Overlay GT marker if available
        gt = config.get("ground_truth_xy")
        if gt is not None:
            try:
                gx, gy = float(gt[0]), float(gt[1])
                ax1.scatter(
                    [gx],
                    [gy],
                    color="black",
                    s=70,
                    linewidths=2.5,
                    marker="x",
                    label="GT",
                )
            except Exception:
                pass
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(3, 4, 2)
        final_raw = results["best_pattern"][0, 0].cpu().numpy()
        # Show normalized version that model actually sees
        final_normalized = (final_raw - wave_mean) / wave_std
        im2 = ax2.imshow(final_normalized, cmap="RdBu_r", interpolation="nearest")
        ax2.set_title("Final Optimized Pattern\n(Model Input Space)", fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])
        # Large rank label to the right of the final result panel
        if config.get("rank_within_layer") is not None and config.get("total_filters_in_layer"):
            try:
                r = int(config["rank_within_layer"]) + 1
                t = int(config["total_filters_in_layer"])
                ax2.text(
                    1.15,
                    0.5,
                    f"rank: {r}/{t}",
                    transform=ax2.transAxes,
                    fontsize=18,
                    fontweight="bold",
                    color="black",
                    va="center",
                    ha="left",
                    clip_on=False,
                )
            except Exception:
                pass
        # Overlay GT marker if available
        gt = config.get("ground_truth_xy")
        if gt is not None:
            try:
                gx, gy = float(gt[0]), float(gt[1])
                ax2.scatter(
                    [gx],
                    [gy],
                    color="black",
                    s=70,
                    linewidths=2.5,
                    marker="x",
                    label="GT",
                )
            except Exception:
                pass
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Panel 2: Evolution timeline - 6 steps for 500 iterations (every 100)
        evolution_data = monitoring_data["intermediate_patterns"]
        num_steps = min(6, len(evolution_data))

        for i in range(num_steps):
            ax = plt.subplot(3, 6, 7 + i)
            if i < len(evolution_data):
                step_data = evolution_data[i]
                pattern_raw = step_data["pattern"][0, 0].cpu().numpy()
                # Show normalized version that model actually sees
                pattern_normalized = (pattern_raw - wave_mean) / wave_std

                im = ax.imshow(
                    pattern_normalized, cmap="RdBu_r", interpolation="nearest"
                )
                ax.set_title(
                    f"Step {step_data['iteration']}\nAct: {step_data['activation']:.1f} T:{step_data['target_loss']:.1f} S:{step_data['suppression_loss']:.1f}\nTV: {step_data['tv_loss']:.2f}",
                    fontsize=8,
                    fontweight="bold",
                )
                ax.set_xticks([])
                ax.set_yticks([])

        # Panel 3: Loss and activation curves
        ax3 = plt.subplot(3, 4, 9)
        iterations = monitoring_data["iteration"]
        losses = monitoring_data["loss"]
        activations = monitoring_data["activation"]
        target_losses = monitoring_data["target_loss"]
        suppression_losses = monitoring_data["suppression_loss"]
        tv_losses = monitoring_data["tv_loss"]
        l2_losses = monitoring_data["l2_loss"]

        ax3.plot(iterations, losses, "b-", label="Total Loss", linewidth=2)
        ax3.plot(iterations, target_losses, "r-", label="Target Loss", linewidth=2)
        ax3.plot(
            iterations,
            suppression_losses,
            "orange",
            label="Suppression Loss",
            linewidth=2,
        )
        if config["tv_reg"] > 0 and any(tv > 0 for tv in tv_losses):
            ax3.plot(
                iterations,
                [v * config["tv_reg"] for v in tv_losses],
                "k--",
                label=f'TV (lambda={config["tv_reg"]})',
                linewidth=1.5,
            )
        if config.get("l2_reg", 0.0) > 0 and any(v > 0 for v in l2_losses):
            ax3.plot(
                iterations,
                [v * config["l2_reg"] for v in l2_losses],
                "c-.",
                label=f'L2 (lambda={config["l2_reg"]})',
                linewidth=1.5,
            )
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Loss Components")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax3.set_title("Optimization Progress", fontweight="bold")

        # Panel 4: Initial feature map for target filter (from the dataset sample)
        ax4 = plt.subplot(3, 4, 10)
        try:
            with torch.no_grad():
                init_in = (results["initial_pattern"] - wave_mean) / wave_std
                _ = self.model(init_in)
            fmap0 = None
            if config["layer_name"] in self.activations:
                layer_out0 = self.activations[config["layer_name"]]
                if layer_out0 is not None and layer_out0.ndim == 4:
                    fmap0 = layer_out0[0, int(config["filter_idx"])].detach().cpu().numpy()
            if fmap0 is not None:
                im_fm0 = ax4.imshow(fmap0, cmap="RdBu_r", interpolation="nearest")
                ax4.set_title("Initial Feature Map", fontweight="bold")
                ax4.set_xticks([])
                ax4.set_yticks([])
                # Overlay GT scaled to fmap size
                gt = config.get("ground_truth_xy")
                if gt is not None:
                    try:
                        gx, gy = float(gt[0]), float(gt[1])
                        H0, W0 = fmap0.shape
                        base = getattr(self.model, "grid_size", 128)
                        px0 = gx * (W0 / float(base))
                        py0 = gy * (H0 / float(base))
                        ax4.scatter([px0], [py0], color="black", s=70, linewidths=2.5, marker="x")
                    except Exception:
                        pass
                plt.colorbar(im_fm0, ax=ax4, fraction=0.046, pad=0.04)
            else:
                ax4.text(0.5, 0.5, "Feature map unavailable", ha="center", va="center")
                ax4.set_axis_off()
        except Exception:
            ax4.text(0.5, 0.5, "Feature map error", ha="center", va="center")
            ax4.set_axis_off()

        # Panel 5: Final feature map for target filter (from the optimized input)
        ax5 = plt.subplot(3, 4, 11)
        try:
            with torch.no_grad():
                final_in = (results["best_pattern"] - wave_mean) / wave_std
                _ = self.model(final_in)
            fmap = None
            if config["layer_name"] in self.activations:
                layer_out = self.activations[config["layer_name"]]
                if layer_out is not None and layer_out.ndim == 4:
                    fmap = layer_out[0, int(config["filter_idx"])].detach().cpu().numpy()
            if fmap is not None:
                im_fm = ax5.imshow(fmap, cmap="RdBu_r", interpolation="nearest")
                ax5.set_title("Final Feature Map", fontweight="bold")
                ax5.set_xticks([])
                ax5.set_yticks([])
                # Overlay GT scaled to fmap size
                gt = config.get("ground_truth_xy")
                if gt is not None:
                    try:
                        gx, gy = float(gt[0]), float(gt[1])
                        H, W = fmap.shape
                        base = getattr(self.model, "grid_size", 128)
                        px = gx * (W / float(base))
                        py = gy * (H / float(base))
                        ax5.scatter([px], [py], color="black", s=70, linewidths=2.5, marker="x")
                    except Exception:
                        pass
                plt.colorbar(im_fm, ax=ax5, fraction=0.046, pad=0.04)
            else:
                ax5.text(0.5, 0.5, "Feature map unavailable", ha="center", va="center")
                ax5.set_axis_off()
        except Exception:
            ax5.text(0.5, 0.5, "Feature map error", ha="center", va="center")
            ax5.set_axis_off()

        # Panel 5 previously showed input stats; now used for Final Feature Map

        # Panel 6: Summary statistics
        ax6 = plt.subplot(3, 4, 12)
        ax6.axis("off")

        summary_stats = {
            "Final Activation": f"{config['final_activation']:.2f}",
            "Final Target Loss": f"{config['final_target_loss']:.4f}",
            "Final Suppression Loss": f"{config['final_suppression_loss']:.4f}",
        }
        if config["tv_reg"] > 0:
            summary_stats["Final TV Loss"] = f"{config['final_tv_loss']:.4f}"
        if config.get("l2_reg", 0.0) > 0:
            summary_stats["Final L2 Loss"] = f"{config['final_l2_loss']:.4f}"
        summary_stats.update(
            {
                "Loss Reduction": f"{config['loss_reduction']:.4f}",
                "Gradient Variation": f"{config['grad_variation']:.6f}",
                "Real Data Init": "Yes" if config["use_real_data_init"] else "No",
                "Dataset": self.dataset_name,
                "Gradients Varying": "Yes" if config["grad_variation"] > 1e-6 else "No",
            }
        )

        stats_text = "\n".join([f"{k}: {v}" for k, v in summary_stats.items()])
        ax6.text(
            0.1,
            0.9,
            stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax6.set_title("Summary", fontweight="bold")

        # Main title
        norm_status = "WITH NORMALIZATION"
        init_status = "REAL DATA" if config["use_real_data_init"] else "RANDOM"
        regs = []
        if config["tv_reg"] > 0:
            regs.append(f"TV lambda={config['tv_reg']}")
        if config.get("l2_reg", 0.0) > 0:
            regs.append(f"L2 lambda={config['l2_reg']}")
        reg_status = " | ".join(regs) if regs else "NO REG"

        # Compose rank text if available
        rank_text = ""
        if config.get("rank_within_layer") is not None and config.get("total_filters_in_layer"):
            rank_text = f" | rank: {int(config['rank_within_layer'])+1}/{int(config['total_filters_in_layer'])}"

        fig.suptitle(
            f"Comprehensive Activation Maximization\n"
            f'{config["layer_name"]} Filter {config["filter_idx"]}{rank_text} | '
            f"{norm_status} | {init_status} INIT | {reg_status}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save plot
        # Prefer caller-provided concise basename when available
        if config.get("file_basename"):
            # Append rank to filename when provided
            if config.get("rank_within_layer") is not None and config.get("total_filters_in_layer"):
                filename = (
                    f"{config['file_basename']}_rank{int(config['rank_within_layer'])+1}"
                    f"from{int(config['total_filters_in_layer'])}.png"
                )
            else:
                filename = f"{config['file_basename']}.png"
        else:
            filename = (
                f"comprehensive_layer_{config['layer_name']}_filter_"
                f"{config['filter_idx']:02d}"
            )
            if config["use_real_data_init"]:
                filename += "_REAL_INIT"
            filename += ".png"

        save_path = save_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Comprehensive plot saved: {save_path}")
        return save_path

    def total_variation_loss(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the Total Variation (TV) loss for a 4D tensor.
        Promotes spatial smoothness in the generated pattern.
        """
        tv_h = torch.mean(torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]))
        return tv_h + tv_w


def get_layer_by_path(model: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    """
    Navigates to a model layer using a dot-notation path string.

    This utility allows accessing nested modules and sequential blocks by name or index.
    For example: 'stage1.0.conv1' or 'coordinate_predictor.2'.

    Args:
        model: The parent PyTorch model.
        layer_path: The dot-separated string path to the target layer.

    Returns:
        The PyTorch module (layer) found at the specified path.

    Raises:
        AttributeError: If a named part of the path does not exist.
        IndexError: If an indexed part of the path is out of bounds.
    """
    current_module = model
    for part in layer_path.split("."):
        if part.isdigit():
            current_module = current_module[int(part)]
        else:
            current_module = getattr(current_module, part)
    return current_module
