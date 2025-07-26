#!/usr/bin/env python3
"""
Explore Multiple Filters in a Single Layer

This script allows the user to select a layer and specify how many filters
to optimize, then visualizes activation maximization results for the first N filters.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from src.activation_maximization.simple_activation_max import run_simple_activation_maximization
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model


def map_conv_layers(model):
    """Map all convolutional layers in the model with their paths and filter counts"""
    conv_layers = []
    
    # Stage 0 - Input processor (Layer 0)
    if hasattr(model, 'wave_input_processor') and len(model.wave_input_processor) > 0:
        if hasattr(model.wave_input_processor[0], 'out_channels'):
            conv_layers.append({
                'layer_id': 0,
                'path': 'wave_input_processor.0',
                'name': 'Stage 0 Conv (Input)',
                'filters': model.wave_input_processor[0].out_channels
            })
    
    layer_counter = 1
    
    # Stage 1: Basic wave features (32 channels)
    stage_info = [
        (1, 'wave_feature_stage1', 'Basic Wave Features'),
        (2, 'wave_pattern_stage2', 'Complex Wave Patterns'), 
        (3, 'interference_stage3', 'Interference Patterns'),
        (4, 'source_localization_stage4', 'Source Localization')
    ]
    
    for stage_num, stage_name, stage_desc in stage_info:
        # Debug: Print what we're looking for
        print(f"🔍 Looking for stage: {stage_name}")
        
        if hasattr(model, stage_name):
            stage = getattr(model, stage_name)
            print(f"✅ Found {stage_name} with {len(stage)} blocks")
            
            # Each stage has 2 blocks, each block has 2 conv layers
            for block_idx in range(len(stage)):
                block = stage[block_idx]
                print(f"   Block {block_idx}: {type(block)}")
                
                # Conv1 in this block
                if hasattr(block, 'wave_feature_conv1'):
                    conv_layers.append({
                        'layer_id': layer_counter,
                        'path': f'{stage_name}.{block_idx}.wave_feature_conv1',
                        'name': f'{stage_desc} Block {block_idx} Conv1',
                        'filters': block.wave_feature_conv1.out_channels
                    })
                    print(f"      Layer {layer_counter}: Conv1 ({block.wave_feature_conv1.out_channels} filters)")
                    layer_counter += 1
                
                # Conv2 in this block
                if hasattr(block, 'wave_feature_conv2'):
                    conv_layers.append({
                        'layer_id': layer_counter,
                        'path': f'{stage_name}.{block_idx}.wave_feature_conv2',
                        'name': f'{stage_desc} Block {block_idx} Conv2',
                        'filters': block.wave_feature_conv2.out_channels
                    })
                    print(f"      Layer {layer_counter}: Conv2 ({block.wave_feature_conv2.out_channels} filters)")
                    layer_counter += 1
        else:
            print(f"❌ Stage {stage_name} not found!")
    
    print(f"\n📊 Total layers mapped: {len(conv_layers)}")
    return conv_layers


def select_random_filters(num_filters, num_to_select=10):
    """Randomly select filter indices"""
    available_filters = list(range(num_filters))
    selected = random.sample(available_filters, min(num_to_select, num_filters))
    return sorted(selected)


def optimize_multiple_filters(model, layer_info, filter_indices, device='cpu'):
    """Optimize multiple filters and return their patterns"""
    print(f"\n🎯 Optimizing {len(filter_indices)} filters from {layer_info['name']}")
    print(f"   Layer path: {layer_info['path']}")
    print(f"   Filter indices: {filter_indices}")
    
    results = []
    iterations = 512
    
    for i, filter_idx in enumerate(filter_indices):
        print(f"\n📊 Filter {i+1}/{len(filter_indices)}: Index {filter_idx}")
        
        try:
            pattern, loss_history = run_simple_activation_maximization(
                model, layer_info['path'], filter_idx, 
                iterations=iterations, device=device
            )
            
            final_activation = -loss_history[-1]  # Convert back from negative loss
            
            results.append({
                'filter_idx': filter_idx,
                'pattern': pattern,
                'final_activation': final_activation,
                'loss_reduction': loss_history[0] - loss_history[-1]
            })
            
            print(f"   ✅ Final activation: {final_activation:.2f}")
            
        except Exception as e:
            print(f"   ❌ Failed to optimize filter {filter_idx}: {str(e)}")
            continue
    
    return results


def create_filter_grid_visualization(results, layer_info, save_path):
    """Create a grid visualization of all optimized filters"""
    num_filters = len(results)
    if num_filters == 0:
        print("❌ No results to visualize!")
        return
    
    # Create grid layout for filters (dynamic based on number of filters)
    if num_filters <= 5:
        rows, cols = 1, num_filters
    elif num_filters <= 10:
        rows, cols = 2, 5
    elif num_filters <= 16:
        rows, cols = 2, 8
    elif num_filters <= 24:
        rows, cols = 3, 8
    elif num_filters <= 32:
        rows, cols = 4, 8
    else:
        # For larger numbers, use 8 columns
        rows = int(np.ceil(num_filters / 8))
        cols = 8
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, 12))
    
    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        # Single subplot - wrap in array for consistent indexing
        axes = np.array([[axes]])
    elif rows == 1:
        # Single row - make it 2D array
        axes = axes.reshape(1, -1)
    elif cols == 1:
        # Single column - make it 2D array  
        axes = axes.reshape(-1, 1)
    
    # Remove extra subplots
    for i in range(num_filters, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].remove()
    
    for i, result in enumerate(results):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Get pattern data
        pattern = result['pattern'][0, 0].cpu().numpy()  # Remove batch and channel dims
        
        # Plot pattern
        im = ax.imshow(pattern, cmap='RdBu_r', interpolation='bilinear')
        
        # Title with filter info
        ax.set_title(f'Filter {result["filter_idx"]}\n'
                    f'Activation: {result["final_activation"]:.1f}', 
                    fontsize=12, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Main title
    fig.suptitle(f'Activation Maximization: {layer_info["name"]}\n'
                f'{layer_info["path"]} - First {num_filters} Filters', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved grid visualization: {save_path}")


def main():
    """Main function for exploring layer filters"""
    print("🔍 Layer Filter Explorer")
    print("=" * 60)
    
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
    print(f"\n📋 Found {len(conv_layers)} convolutional layers (indexed 0-{len(conv_layers)-1}):")
    
    for layer in conv_layers:
        print(f"   Layer {layer['layer_id']:2d}: {layer['name']} ({layer['filters']} filters)")
    
    # Ask user for layer selection
    max_layer_id = max(layer['layer_id'] for layer in conv_layers)
    print(f"\n🎯 Layer Selection:")
    print(f"   Available layer indices: 0 to {max_layer_id}")
    
    while True:
        try:
            selected_layer_id = int(input(f"Enter layer number (0-{max_layer_id}): "))
            if 0 <= selected_layer_id <= max_layer_id:
                break
            else:
                print(f"❌ Please enter a number between 0 and {max_layer_id}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Find selected layer info
    selected_layer = next(layer for layer in conv_layers if layer['layer_id'] == selected_layer_id)
    print(f"\n✅ Selected: {selected_layer['name']}")
    print(f"   Path: {selected_layer['path']}")
    print(f"   Total filters: {selected_layer['filters']}")
    
    # Ask user for number of filters
    max_filters = selected_layer['filters']
    print(f"\n🔢 Filter Selection:")
    print(f"   Available filters: 1 to {max_filters}")
    
    while True:
        try:
            num_filters_to_test = int(input(f"Enter number of filters to optimize (1-{max_filters}): "))
            if 1 <= num_filters_to_test <= max_filters:
                break
            else:
                print(f"❌ Please enter a number between 1 and {max_filters}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Select first N filters (in order)
    filter_indices = list(range(num_filters_to_test))
    
    print(f"\n🔢 Selected first {len(filter_indices)} filters: {filter_indices}")
    
    # Optimize all selected filters
    results = optimize_multiple_filters(model, selected_layer, filter_indices, device)
    
    if not results:
        print("❌ No successful optimizations!")
        return
    
    # Create visualization
    output_dir = Path("experiments/activation_maximization")
    output_dir.mkdir(exist_ok=True)
    
    save_path = output_dir / f"layer_{selected_layer_id:02d}_multiple_filters.png"
    create_filter_grid_visualization(results, selected_layer, save_path)
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Layer: {selected_layer['name']}")
    print(f"   Successful optimizations: {len(results)}/{len(filter_indices)}")
    print(f"   Activation range: {min(r['final_activation'] for r in results):.1f} - {max(r['final_activation'] for r in results):.1f}")
    print(f"   Results saved to: {save_path}")
    
    print(f"\n🎉 Layer Filter Exploration COMPLETED!")


if __name__ == "__main__":
    main() 