#!/usr/bin/env python3
"""
Comprehensive Layer Analysis - Activation Maximization

This script automatically runs activation maximization on the first 9 filters
for every convolutional layer in the network and saves all results.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.activation_maximization.simple_activation_max import SimpleActivationMaximizer
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model


def map_conv_layers(model):
    """Map all convolutional layers in the model with their paths and filter counts"""
    conv_layers = []
    
    # Stage 0 - Input processor (Layer 0)
    if hasattr(model, 'wave_input_processor') and len(model.wave_input_processor) > 0:
        if isinstance(model.wave_input_processor[0], torch.nn.Conv2d):
            conv_layers.append({
                'layer_id': 0,
                'path': 'wave_input_processor.0',
                'name': 'Stage 0 Conv (Input)',
                'filters': model.wave_input_processor[0].out_channels,
                'module': model.wave_input_processor[0]
            })
    
    layer_counter = 1
    
    # Stage 1-4: Feature extraction stages
    stage_info = [
        (1, 'wave_feature_stage1', 'Basic Wave Features'),
        (2, 'wave_pattern_stage2', 'Complex Wave Patterns'), 
        (3, 'interference_stage3', 'Interference Patterns'),
        (4, 'source_localization_stage4', 'Source Localization')
    ]
    
    for stage_num, stage_name, stage_desc in stage_info:
        if hasattr(model, stage_name):
            stage = getattr(model, stage_name)
            
            # Each stage has 2 blocks, each block has 2 conv layers
            for block_idx in range(len(stage)):
                block = stage[block_idx]
                
                # Conv1 in this block
                if hasattr(block, 'wave_feature_conv1') and isinstance(block.wave_feature_conv1, torch.nn.Conv2d):
                    conv_layers.append({
                        'layer_id': layer_counter,
                        'path': f'{stage_name}.{block_idx}.wave_feature_conv1',
                        'name': f'{stage_desc} Block {block_idx} Conv1',
                        'filters': block.wave_feature_conv1.out_channels,
                        'module': block.wave_feature_conv1
                    })
                    layer_counter += 1
                
                # Conv2 in this block
                if hasattr(block, 'wave_feature_conv2') and isinstance(block.wave_feature_conv2, torch.nn.Conv2d):
                    conv_layers.append({
                        'layer_id': layer_counter,
                        'path': f'{stage_name}.{block_idx}.wave_feature_conv2',
                        'name': f'{stage_desc} Block {block_idx} Conv2',
                        'filters': block.wave_feature_conv2.out_channels,
                        'module': block.wave_feature_conv2
                    })
                    layer_counter += 1
    
    return conv_layers


def get_layer_by_path(model, layer_path: str):
    """Navigate to layer using dot notation path"""
    parts = layer_path.split('.')
    current = model
    
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    
    return current


def optimize_multiple_filters(model, layer_info, filter_indices, device='cpu'):
    """Optimize multiple filters for a given layer"""
    print(f"\n🎯 Optimizing Layer {layer_info['layer_id']}: {layer_info['name']}")
    print(f"   Path: {layer_info['path']}")
    print(f"   Filters: {filter_indices}")
    
    # Create maximizer
    maximizer = SimpleActivationMaximizer(model, device)
    
    # Get target layer and register hook
    target_layer = layer_info['module']
    layer_name = f"layer_{layer_info['layer_id']}"
    maximizer.register_hook(layer_name, target_layer)
    
    results = []
    iterations = 512  # Faster for comprehensive analysis
    
    try:
        for i, filter_idx in enumerate(filter_indices):
            print(f"📊 Filter {i+1}/{len(filter_indices)}: Index {filter_idx}")
            
            # Run optimization
            optimized_pattern, loss_history = maximizer.optimize_filter(
                layer_name, 
                filter_idx, 
                iterations=iterations
            )
            
            # Store result
            final_activation = loss_history[-1] if loss_history else 0
            results.append({
                'filter_idx': filter_idx,
                'pattern': optimized_pattern,
                'loss_history': loss_history,
                'final_activation': -final_activation  # Convert back to positive
            })
            
            print(f"   ✅ Final activation: {-final_activation:.2f}")
    
    finally:
        maximizer.cleanup_hooks()
    
    return results


def create_layer_grid_visualization(results, layer_info, save_path):
    """Create a 3x3 grid visualization for a layer's filters"""
    num_filters = len(results)
    if num_filters == 0:
        print("❌ No results to visualize!")
        return
    
    # Create 3x3 grid
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
    # Remove extra subplots if we have fewer than 9 filters
    for i in range(num_filters, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].remove()
    
    for i, result in enumerate(results):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Extract pattern data
        pattern = result['pattern'][0, 0].numpy()  # Remove batch and channel dims
        filter_idx = result['filter_idx']
        final_activation = result['final_activation']
        
        # Display pattern
        im = ax.imshow(pattern, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f'Filter {filter_idx}\nActivation: {final_activation:.1f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Main title
    fig.suptitle(f'Layer {layer_info["layer_id"]}: {layer_info["name"]}\n'
                f'First 9 Filters - Activation Maximization (No Regularization)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {save_path}")


def main():
    """Main function for comprehensive layer analysis"""
    print("🌟 Comprehensive Layer Analysis - Activation Maximization")
    print("=" * 80)
    
    # Load best CV model
    print("🔍 Loading Best CV Model...")
    cv_results_path = Path(__file__).parent.parent.parent / "experiments" / "cv_full"
    model_info = find_best_cv_model(cv_results_path)
    if model_info is None:
        print("❌ No CV model found!")
        return
        
    fold_id, error, model_path = model_info
    print(f"✅ Found best model: Fold {fold_id} (error: {error:.4f}px)")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_wave_source_model(grid_size=128)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Map all convolutional layers
    conv_layers = map_conv_layers(model)
    print(f"\n📋 Found {len(conv_layers)} convolutional layers")
    
    # Create output directory
    output_dir = Path("experiments/activation_maximization/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each layer
    total_layers = len(conv_layers)
    for i, layer_info in enumerate(conv_layers):
        print(f"\n🔄 Processing Layer {i+1}/{total_layers}: {layer_info['name']}")
        
        # Always do first 9 filters (or all filters if less than 9)
        num_filters_to_analyze = min(9, layer_info['filters'])
        filter_indices = list(range(num_filters_to_analyze))
        
        # Run optimization
        results = optimize_multiple_filters(model, layer_info, filter_indices, device)
        
        # Create visualization
        save_path = output_dir / f"layer_{layer_info['layer_id']:02d}_first_9_filters.png"
        create_layer_grid_visualization(results, layer_info, save_path)
        
        print(f"✅ Layer {layer_info['layer_id']} complete: {len(results)} filters optimized")
    
    print(f"\n🎉 Comprehensive Analysis COMPLETED!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"📊 Total layers analyzed: {total_layers}")
    print(f"🔍 Filters per layer: 9 (first 9 filters)")
    print(f"⚡ Configuration: 512 iterations, no regularization")


if __name__ == "__main__":
    main() 