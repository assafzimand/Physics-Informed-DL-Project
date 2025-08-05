#!/usr/bin/env python3
"""
Interactive Activation Maximization Analysis

Asks user for layer, finds top 5 active filters, and optimizes each one.
Uses the same real sample as initialization for all filter optimizations.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.activation_maximization.simple_activation_max import (
    SimpleActivationMaximizer
)
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model
from src.data.wave_dataset import WaveDataset


def get_top_active_filters(model, device, layer, sample_tensor, top_k=5):
    """
    Find the top K most active filters in a given layer for a specific sample.
    
    Args:
        model: The trained model
        device: Torch device
        layer: The target layer module
        sample_tensor: Input sample tensor (1, 1, H, W)
        top_k: Number of top filters to return
        
    Returns:
        List of filter indices sorted by activation strength
    """
    print(f"🔍 Finding top {top_k} active filters...")
    
    # Hook to capture activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations['target'] = output.detach()
    
    # Register hook
    hook = layer.register_forward_hook(hook_fn)
    
    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(sample_tensor)
        
        # Get activations and compute filter-wise mean activation
        layer_output = activations['target']  # Shape: (1, C, H, W)
        
        # Compute mean activation per filter across spatial dimensions
        filter_activations = layer_output.mean(dim=(0, 2, 3))  # Shape: (C,)
        
        # Get top K filter indices
        top_indices = torch.argsort(filter_activations, descending=True)[:top_k].cpu().numpy()
        top_values = filter_activations[top_indices].cpu().numpy()
        
        print(f"📊 Top {top_k} filters:")
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            print(f"  {i+1}. Filter {idx}: {val:.3f}")
        
        return top_indices.tolist()
    
    finally:
        hook.remove()


def main():
    """Run interactive activation maximization analysis"""
    
    print("🧪 INTERACTIVE ACTIVATION MAXIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Load model
    cv_results_path = (Path(__file__).parent.parent.parent / 
                      "experiments" / "cv_full")
    model_info = find_best_cv_model(cv_results_path)
    fold_id, error, model_path = model_info
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_wave_source_model(grid_size=128)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    
    print(f"✅ Loaded best model: Fold {fold_id}")
    
    # Print available layers
    print("\n🔧 Available Conv2D layers:")
    conv_layers = []
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((i, name, module))
            print(f"  {len(conv_layers)-1}: Layer {i} - {name} "
                  f"({module.out_channels} filters)")
    
    # Get user input for layer
    while True:
        try:
            layer_choice = int(input(f"\n🎯 Choose layer "
                                   f"(0-{len(conv_layers)-1}): "))
            if 0 <= layer_choice < len(conv_layers):
                break
            else:
                print(f"❌ Please choose a number between 0 and "
                      f"{len(conv_layers)-1}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    layer_idx, layer_name, target_layer = conv_layers[layer_choice]
    print(f"✅ Selected: Layer {layer_idx} - {layer_name}")
    
    # Create layer-specific save directory
    layer_save_dir = f"experiments/activation_maximization/comprehensive/layer_{layer_idx}"
    
    # Load a sample from the dataset
    dataset_path = "data/wave_dataset_analysis_20samples.h5"
    dataset = WaveDataset(dataset_path, normalize_wave_fields=True)
    
    # Use a fixed sample (sample 0) for consistency
    sample_idx = 0
    wave_field, coordinates = dataset[sample_idx]
    sample_tensor = wave_field.to(device)  # Already has correct dimensions [1, 1, H, W]
    
    print(f"🌊 Using sample {sample_idx} as reference and initialization")
    print(f"📊 Sample coordinates: x={coordinates[0]:.1f}, "
          f"y={coordinates[1]:.1f}")
    
    # Find top 5 active filters
    top_filters = get_top_active_filters(model, device, target_layer, 
                                       sample_tensor, top_k=5)
    
    # Setup maximizer
    maximizer = SimpleActivationMaximizer(model, device)
    maximizer.register_hook("target_layer", target_layer)
    
    try:
        # Optimize each of the top 5 filters
        print(f"\n🚀 Optimizing top 5 filters...")
        print(f"📁 Saving all results to: {layer_save_dir}")
        
        for i, filter_idx in enumerate(top_filters):
            print(f"\n🎯 Optimizing filter {filter_idx} ({i+1}/5)...")
            
            # Create a copy of the sample for initialization
            init_sample = sample_tensor.clone().detach()
            
            # Run optimization with the layer-specific save directory
            results = maximizer.optimize_filter(
                layer_name="target_layer",
                filter_idx=filter_idx,
                iterations=500,
                learning_rate=0.01,
                skip_normalization=False,
                use_real_data_init=True,
                save_dir=layer_save_dir
            )
            
            print(f"✅ Filter {filter_idx} complete! Final activation: "
                  f"{results['config']['final_activation']:.2f}")
        
        print(f"\n🎉 All optimizations complete! Check {layer_save_dir}/ for plots.")
        
    finally:
        maximizer.cleanup_hooks()


if __name__ == "__main__":
    main() 