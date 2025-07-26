#!/usr/bin/env python3
"""
Visualize Learned Filter Weights

This script loads the best CV model and visualizes the actual learned
convolutional filter weights from any layer.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np
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


def extract_filter_weights(conv_module, filter_indices):
    """Extract filter weights from a convolutional layer"""
    weights = conv_module.weight.data.cpu()  # Shape: [out_channels, in_channels, H, W]
    
    selected_filters = []
    for filter_idx in filter_indices:
        if filter_idx < weights.shape[0]:
            filter_weight = weights[filter_idx]  # Shape: [in_channels, H, W]
            selected_filters.append(filter_weight)
    
    return selected_filters


def create_filter_weight_visualization(filter_weights, filter_indices, layer_info, save_path):
    """Create a grid visualization of learned filter weights"""
    num_filters = len(filter_weights)
    if num_filters == 0:
        print("❌ No filters to visualize!")
        return
    
    # Get filter dimensions
    sample_filter = filter_weights[0]
    in_channels, filter_h, filter_w = sample_filter.shape
    
    print(f"📊 Filter info: {in_channels} input channels, {filter_h}x{filter_w} kernel size")
    
    # Create grid layout
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
        rows = int(np.ceil(num_filters / 8))
        cols = 8
    
    # Handle multiple input channels by showing each separately or combining
    if in_channels == 1:
        # Single input channel - direct visualization
        fig, axes = plt.subplots(rows, cols, figsize=(24, 12))
        
        # Handle different subplot configurations
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Remove extra subplots
        for i in range(num_filters, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].remove()
        
        # Calculate global min/max across all filters for reference
        all_weights = torch.stack([fw[0] for fw in filter_weights])  # Stack all filters
        global_min = all_weights.min().item()
        global_max = all_weights.max().item()
        
        print(f"📊 Raw weight range: [{global_min:.4f}, {global_max:.4f}] - Using raw values (no normalization)")
        
        for i, (filter_weight, filter_idx) in enumerate(zip(filter_weights, filter_indices)):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            # Show single channel filter with RAW weights (no normalization!)
            weight_2d = filter_weight[0].numpy()  # Remove channel dim
            
            im = ax.imshow(weight_2d, cmap='RdBu_r', interpolation='nearest')
            
            # Show actual weight range for this filter
            f_min, f_max = weight_2d.min(), weight_2d.max()
            ax.set_title(f'Filter {filter_idx}\nRange: [{f_min:.4f}, {f_max:.4f}]', 
                        fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    else:
        # Multiple input channels - show average or first channel
        fig, axes = plt.subplots(rows, cols, figsize=(24, 12))
        
        # Handle different subplot configurations  
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Remove extra subplots
        for i in range(num_filters, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].remove()
        
        # Calculate global min/max across all averaged filters for reference
        all_avg_weights = torch.stack([fw.mean(dim=0) for fw in filter_weights])
        global_min = all_avg_weights.min().item()
        global_max = all_avg_weights.max().item()
        
        print(f"📊 Raw averaged weight range: [{global_min:.4f}, {global_max:.4f}] - Using raw values (no normalization)")
        
        for i, (filter_weight, filter_idx) in enumerate(zip(filter_weights, filter_indices)):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            # Average across input channels for visualization with RAW weights (no normalization!)
            weight_2d = filter_weight.mean(dim=0).numpy()
            
            im = ax.imshow(weight_2d, cmap='RdBu_r', interpolation='nearest')
            
            # Show actual weight range for this averaged filter
            f_min, f_max = weight_2d.min(), weight_2d.max()
            ax.set_title(f'Filter {filter_idx} (avg {in_channels}ch)\nRange: [{f_min:.4f}, {f_max:.4f}]', 
                        fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Main title
    fig.suptitle(f'Learned Filter Weights: {layer_info["name"]}\n'
                f'{layer_info["path"]} - First {num_filters} Filters', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved filter weights visualization: {save_path}")


def main():
    """Main function for visualizing learned filter weights"""
    print("🔍 Learned Filter Weight Visualizer")
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
            num_filters_to_show = int(input(f"Enter number of filters to visualize (1-{max_filters}): "))
            if 1 <= num_filters_to_show <= max_filters:
                break
            else:
                print(f"❌ Please enter a number between 1 and {max_filters}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Select first N filters (in order)
    filter_indices = list(range(num_filters_to_show))
    print(f"\n🔢 Selected first {len(filter_indices)} filters: {filter_indices}")
    
    # Extract filter weights
    print(f"\n📊 Extracting filter weights from {selected_layer['name']}...")
    filter_weights = extract_filter_weights(selected_layer['module'], filter_indices)
    
    if not filter_weights:
        print("❌ No filter weights extracted!")
        return
    
    # Create visualization
    output_dir = Path("experiments/filter_visualization")
    output_dir.mkdir(exist_ok=True)
    
    save_path = output_dir / f"layer_{selected_layer_id:02d}_learned_filters.png"
    create_filter_weight_visualization(filter_weights, filter_indices, selected_layer, save_path)
    
    # Print summary
    sample_filter = filter_weights[0]
    in_channels, filter_h, filter_w = sample_filter.shape
    print(f"\n📊 Summary:")
    print(f"   Layer: {selected_layer['name']}")
    print(f"   Filters visualized: {len(filter_weights)}")
    print(f"   Filter shape: {in_channels} channels, {filter_h}x{filter_w} kernel")
    print(f"   Results saved to: {save_path}")
    
    print(f"\n🎉 Filter Weight Visualization COMPLETED!")


if __name__ == "__main__":
    main() 