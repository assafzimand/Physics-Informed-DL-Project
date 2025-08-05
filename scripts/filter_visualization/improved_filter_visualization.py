#!/usr/bin/env python3
"""
Improved Filter Visualization

Enhanced visualization techniques to better reveal learned patterns in 
convolutional filters, especially for layer 0 edge detectors.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model


def create_enhanced_filter_visualization(filter_weights, filter_indices, layer_info, save_path):
    """Create enhanced visualization with multiple approaches"""
    num_filters = len(filter_weights)
    if num_filters == 0:
        print("❌ No filters to visualize!")
        return

    # Get filter dimensions
    sample_filter = filter_weights[0]
    in_channels, filter_h, filter_w = sample_filter.shape
    
    print(f"📊 Enhanced visualization for {num_filters} filters ({filter_h}x{filter_w})")
    
    # Show first 9 filters with multiple visualization approaches
    num_show = min(9, num_filters)
    
    # Create figure with 3 different visualization approaches
    fig = plt.figure(figsize=(20, 16))
    
    # Approach 1: Raw weights with enhanced contrast
    fig.suptitle(f'Layer {layer_info["layer_id"]}: {layer_info["name"]} - Enhanced Filter Visualization', 
                fontsize=16, fontweight='bold')
    
    for i in range(num_show):
        filter_weight = filter_weights[i][0].numpy()  # First input channel
        filter_idx = filter_indices[i]
        
        # Calculate subplot positions (3 rows of approaches, 9 columns for filters)
        
        # Row 1: Raw weights with enhanced contrast
        ax1 = plt.subplot(3, 9, i + 1)
        
        # Enhance contrast by centering around zero and scaling
        centered = filter_weight - filter_weight.mean()
        std_val = centered.std()
        if std_val > 0:
            enhanced = centered / std_val  # Normalize by std
        else:
            enhanced = centered
            
        im1 = ax1.imshow(enhanced, cmap='RdBu_r', interpolation='none', vmin=-3, vmax=3)
        ax1.set_title(f'F{filter_idx}\nEnhanced', fontsize=8, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Row 2: Absolute values to see structure
        ax2 = plt.subplot(3, 9, i + 10)
        abs_weights = np.abs(filter_weight)
        im2 = ax2.imshow(abs_weights, cmap='viridis', interpolation='none')
        ax2.set_title(f'F{filter_idx}\nAbs Values', fontsize=8, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Row 3: Edge detection visualization
        ax3 = plt.subplot(3, 9, i + 19)
        
        # Apply edge detection to the filter itself to highlight structures
        if filter_h > 2 and filter_w > 2:
            # Simple edge detection using gradients
            grad_x = np.diff(filter_weight, axis=1)
            grad_y = np.diff(filter_weight, axis=0)
            
            # Pad to maintain size
            grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
            grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
            
            edge_mag = np.sqrt(grad_x**2 + grad_y**2)
            im3 = ax3.imshow(edge_mag, cmap='hot', interpolation='none')
        else:
            im3 = ax3.imshow(filter_weight, cmap='RdBu_r', interpolation='none')
            
        ax3.set_title(f'F{filter_idx}\nEdge Mag', fontsize=8, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # Add row labels
    fig.text(0.02, 0.75, 'Enhanced\nContrast', fontsize=12, fontweight='bold', ha='center', va='center')
    fig.text(0.02, 0.5, 'Absolute\nValues', fontsize=12, fontweight='bold', ha='center', va='center')
    fig.text(0.02, 0.25, 'Edge\nMagnitude', fontsize=12, fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Enhanced visualization saved: {save_path}")


def create_large_scale_visualization(filter_weights, filter_indices, layer_info, save_path):
    """Create large-scale visualization by upsampling filters"""
    num_filters = min(9, len(filter_weights))
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_filters):
        ax = axes[i]
        filter_weight = filter_weights[i][0].numpy()  # First input channel
        filter_idx = filter_indices[i]
        
        # Upsample filter using interpolation to make patterns more visible
        from scipy.ndimage import zoom
        upsampled = zoom(filter_weight, 4, order=3)  # 4x upsampling with cubic interpolation
        
        # Enhance contrast
        centered = upsampled - upsampled.mean()
        std_val = centered.std()
        if std_val > 0:
            enhanced = centered / std_val
        else:
            enhanced = centered
        
        im = ax.imshow(enhanced, cmap='RdBu_r', interpolation='bilinear', vmin=-2, vmax=2)
        
        # Calculate filter statistics for title
        f_min, f_max = filter_weight.min(), filter_weight.max()
        f_std = filter_weight.std()
        
        ax.set_title(f'Filter {filter_idx}\nRange: [{f_min:.3f}, {f_max:.3f}]\nStd: {f_std:.3f}', 
                    fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Remove unused subplots
    for i in range(num_filters, 9):
        axes[i].remove()
    
    plt.suptitle(f'Layer {layer_info["layer_id"]}: {layer_info["name"]}\n'
                f'Large Scale Visualization (4x Upsampled)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Large-scale visualization saved: {save_path}")


def analyze_filter_patterns(filter_weights, filter_indices):
    """Analyze filters to categorize their patterns"""
    print(f"\n🔍 FILTER PATTERN ANALYSIS:")
    print("=" * 60)
    
    pattern_types = []
    
    for i, (filter_weight, filter_idx) in enumerate(zip(filter_weights, filter_indices)):
        weight_2d = filter_weight[0].numpy()  # First input channel
        
        # Calculate directional gradients
        grad_x = np.gradient(weight_2d, axis=1)
        grad_y = np.gradient(weight_2d, axis=0)
        
        # Analyze gradient patterns
        mean_grad_x = np.abs(grad_x).mean()
        mean_grad_y = np.abs(grad_y).mean()
        
        # Categorize pattern type
        if mean_grad_x > mean_grad_y * 1.5:
            pattern_type = "Vertical Edge"
        elif mean_grad_y > mean_grad_x * 1.5:
            pattern_type = "Horizontal Edge"
        elif abs(mean_grad_x - mean_grad_y) < 0.01:
            pattern_type = "Diagonal/Complex"
        else:
            pattern_type = "Mixed"
        
        # Calculate center vs edge activation
        center_val = weight_2d[weight_2d.shape[0]//2, weight_2d.shape[1]//2]
        edge_vals = np.concatenate([weight_2d[0, :], weight_2d[-1, :], 
                                   weight_2d[:, 0], weight_2d[:, -1]])
        center_vs_edge = center_val - edge_vals.mean()
        
        pattern_info = {
            'filter_idx': filter_idx,
            'type': pattern_type,
            'grad_x': mean_grad_x,
            'grad_y': mean_grad_y,
            'center_vs_edge': center_vs_edge,
            'std': weight_2d.std()
        }
        pattern_types.append(pattern_info)
        
        print(f"   Filter {filter_idx:2d}: {pattern_type:15s} | "
              f"GradX: {mean_grad_x:.3f}, GradY: {mean_grad_y:.3f} | "
              f"C-E: {center_vs_edge:.3f} | Std: {weight_2d.std():.3f}")
    
    return pattern_types


def main():
    """Main function with improved filter visualization"""
    print("🎨 IMPROVED FILTER VISUALIZATION")
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
    
    # Get layer 0 filters
    conv_layer = model.wave_input_processor[0]
    layer_info = {
        'layer_id': 0,
        'name': 'Stage 0 Conv (Input)',
        'path': 'wave_input_processor.0'
    }
    
    # Extract first 9 filters
    weights = conv_layer.weight.data.cpu()
    filter_indices = list(range(9))
    filter_weights = [weights[i] for i in filter_indices]
    
    print(f"\n📊 Analyzing {len(filter_weights)} filters from Layer 0...")
    
    # Create output directory
    output_dir = Path("experiments/filter_visualization/improved")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate improved visualizations
    enhanced_path = output_dir / "layer_0_enhanced_filters.png"
    create_enhanced_filter_visualization(filter_weights, filter_indices, layer_info, enhanced_path)
    
    large_scale_path = output_dir / "layer_0_large_scale_filters.png"
    create_large_scale_visualization(filter_weights, filter_indices, layer_info, large_scale_path)
    
    # Analyze patterns
    pattern_analysis = analyze_filter_patterns(filter_weights, filter_indices)
    
    print(f"\n🎉 Improved visualization complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Summary of findings
    pattern_counts = {}
    for p in pattern_analysis:
        pattern_counts[p['type']] = pattern_counts.get(p['type'], 0) + 1
    
    print(f"\n📊 Pattern Summary:")
    for pattern_type, count in pattern_counts.items():
        print(f"   {pattern_type}: {count} filters")


if __name__ == "__main__":
    main() 