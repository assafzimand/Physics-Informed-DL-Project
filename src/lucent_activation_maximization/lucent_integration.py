#!/usr/bin/env python3
"""
Lucent Integration Helper for Wave Pattern Activation Maximization.

Provides high-level functions to setup and run Lucent-based activation 
maximization for wave source localization model analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Any
from pathlib import Path

# Lucent imports
try:
    import lucent
    from lucent.optvis import render
    from lucent.optvis.objectives import channel
    from lucent.optvis.param import image, cppn
    from lucent.optvis.param.spatial import pixel_image
    LUCENT_AVAILABLE = True
except ImportError:
    LUCENT_AVAILABLE = False
    print("⚠️  Lucent not available. Install with: pip install torch-lucent")


def verify_lucent():
    """Verify Lucent is available and working."""
    if not LUCENT_AVAILABLE:
        raise ImportError("Lucent is required for activation maximization")
    print(f"✅ Lucent available: version {lucent.__version__}")


def setup_wave_optimization(model_wrapper, layer_name: str, filter_idx: int, 
                           config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Setup Lucent optimization for wave pattern generation.
    
    Args:
        model_wrapper: LucentModelWrapper instance
        layer_name: Target layer name (e.g., 'stage2_conv2')
        filter_idx: Filter index to optimize
        config: Optimization configuration
        
    Returns:
        Tuple of (objective, param_function)
    """
    verify_lucent()
    
    # Default configuration
    default_config = {
        "image_size": 128,
        "channels": 1,
        "param_type": "pixel",  # or "cppn" for compositional patterns
        "decorrelate": True,
        "fft": True
    }
    
    if config:
        default_config.update(config)
    
    # Setup optimization objective
    # Lucent expects layer access as model attribute
    objective = channel(layer_name, filter_idx)
    
    # Setup parameter function for wave field optimization
    if default_config["param_type"] == "pixel":
        def param_func():
            # Create parameter tensor with 3 channels for Lucent compatibility
            import torch
            # Shape: [batch_size=1, channels=3, height, width] for RGB format
            shape = (1, 3, default_config["image_size"], default_config["image_size"])
            tensor = torch.randn(*shape, requires_grad=True)
            return [tensor], lambda: tensor  # Return list for optimizer
        param_f = param_func
    elif default_config["param_type"] == "cppn":
        def param_func():
            return cppn(
                default_config["image_size"],
                default_config["image_size"]
            )
        param_f = param_func
    else:
        raise ValueError(f"Unknown param_type: {default_config['param_type']}")
    
    return objective, param_f


def run_activation_maximization(
    model_wrapper, 
    objective, 
    param_f,
    iterations=256,
    learning_rate=0.05,
    show_progress=True
):
    """
    Run activation maximization optimization with loss tracking.
    
    Returns:
        tuple: (optimized_pattern, loss_history)
    """
    import lucent.optvis.render as render
    
    print(f"🎯 Running activation maximization...")
    print(f"   Iterations: {iterations}")
    print(f"   Learning rate: {learning_rate}")
    
    # Simple loss tracking approach
    loss_history = []
    
    # Create wrapper objective that tracks losses
    class LossTrackingObjective:
        def __init__(self, original_objective):
            self.original_objective = original_objective
            self.losses = []
            
        def __call__(self, model):
            loss = self.original_objective(model)
            self.losses.append(float(loss.item()))
            return loss
    
    tracking_objective = LossTrackingObjective(objective)
    
    try:
        # Run Lucent optimization
        result = render.render_vis(
            model_wrapper,
            tracking_objective,
            param_f=param_f,
            transforms=[],  # Explicitly disable transforms
            show_inline=False,
            show_image=show_progress
        )
        
        # Get tracked losses
        loss_history = tracking_objective.losses
        
        # Extract the optimized pattern
        if isinstance(result, list):
            # If result is a list, take the first element (usually the image)
            optimized_pattern = result[0]
        else:
            optimized_pattern = result
            
        # Convert to numpy if it's a tensor
        if hasattr(optimized_pattern, 'detach'):
            optimized_pattern = optimized_pattern.detach().cpu().numpy()
        elif hasattr(optimized_pattern, 'numpy'):
            optimized_pattern = optimized_pattern.numpy()
            
        print(f"✅ Optimization complete! Generated pattern: {optimized_pattern.shape}")
        print(f"📊 Tracked {len(loss_history)} loss values")
        
        return optimized_pattern, loss_history
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        raise


def visualize_wave_pattern(wave_pattern: torch.Tensor, 
                          layer_name: str, 
                          filter_idx: int,
                          save_path: Optional[Path] = None,
                          show_plot: bool = True) -> plt.Figure:
    """
    Visualize generated wave pattern with proper formatting.
    
    Args:
        wave_pattern: Generated wave pattern tensor
        layer_name: Layer name for title
        filter_idx: Filter index for title
        save_path: Optional path to save visualization
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib figure
    """
    # Convert to numpy for plotting
    if isinstance(wave_pattern, torch.Tensor):
        pattern_np = wave_pattern.detach().cpu().numpy()
    else:
        pattern_np = wave_pattern
    
    # Handle different tensor shapes and convert RGB to grayscale like the model does
    if pattern_np.ndim == 4:
        # [B, C, H, W] format - remove batch dimension
        pattern_np = pattern_np[0]  # Now [C, H, W]
    
    if pattern_np.ndim == 3:
        # [C, H, W] format
        if pattern_np.shape[0] == 3:
            # RGB format - convert to grayscale using same method as model (R+G+B)
            pattern_np = pattern_np[0] + pattern_np[1] + pattern_np[2]
            
            # Apply the SAME normalization that the model sees during optimization
            wave_mean = 0.000460  # Training dataset mean
            wave_std = 0.020842   # Training dataset std
            pattern_np = (pattern_np - wave_mean) / wave_std
        else:
            # Already grayscale
            pattern_np = pattern_np[0]
    elif pattern_np.ndim == 2:
        # Already 2D grayscale
        pass
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot wave pattern
    im = ax.imshow(pattern_np, cmap='RdBu_r', interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Wave Amplitude', rotation=270, labelpad=20)
    
    # Formatting
    ax.set_title(f'{layer_name.upper()}: Filter {filter_idx}\nOptimized Wave Pattern', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    pattern_min, pattern_max = pattern_np.min(), pattern_np.max()
    pattern_std = pattern_np.std()
    
    stats_text = f'Range: [{pattern_min:.3f}, {pattern_max:.3f}]\nStd: {pattern_std:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


def analyze_wave_pattern(wave_pattern: torch.Tensor) -> Dict[str, float]:
    """
    Analyze generated wave pattern for physical properties.
    
    Args:
        wave_pattern: Generated wave pattern tensor
        
    Returns:
        analysis: Dictionary of wave pattern properties
    """
    if isinstance(wave_pattern, torch.Tensor):
        pattern = wave_pattern.detach().cpu().numpy()
    else:
        pattern = wave_pattern
    
    # Handle tensor dimensions
    if pattern.ndim == 3:
        pattern = pattern[0]
    elif pattern.ndim == 4:
        pattern = pattern[0, 0]
    
    # Compute wave properties
    analysis = {
        "amplitude_range": float(pattern.max() - pattern.min()),
        "mean_amplitude": float(pattern.mean()),
        "std_amplitude": float(pattern.std()),
        "energy": float(np.sum(pattern ** 2)),
        "sparsity": float(np.count_nonzero(np.abs(pattern) < 0.01) / pattern.size),
    }
    
    # Compute spatial gradients (rough frequency analysis)
    grad_x = np.gradient(pattern, axis=1)
    grad_y = np.gradient(pattern, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    analysis.update({
        "spatial_frequency": float(gradient_magnitude.mean()),
        "gradient_std": float(gradient_magnitude.std()),
    })
    
    return analysis


def compare_with_real_wave(generated_pattern: torch.Tensor, 
                          real_wave: torch.Tensor) -> Dict[str, float]:
    """
    Compare generated pattern with real wave field.
    
    Args:
        generated_pattern: AI-generated wave pattern
        real_wave: Real wave field from dataset
        
    Returns:
        comparison: Similarity metrics
    """
    # Convert to numpy
    gen = generated_pattern.detach().cpu().numpy()
    real = real_wave.detach().cpu().numpy()
    
    # Ensure same shape
    if gen.ndim == 3:
        gen = gen[0]
    if real.ndim == 3:
        real = real[0]
    
    # Normalize both patterns
    gen_norm = (gen - gen.mean()) / (gen.std() + 1e-8)
    real_norm = (real - real.mean()) / (real.std() + 1e-8)
    
    # Compute similarity metrics
    correlation = np.corrcoef(gen_norm.flatten(), real_norm.flatten())[0, 1]
    mse = np.mean((gen_norm - real_norm) ** 2)
    
    comparison = {
        "correlation": float(correlation),
        "mse": float(mse),
        "pattern_similarity": float(max(0, correlation))  # Clamp to positive
    }
    
    return comparison 