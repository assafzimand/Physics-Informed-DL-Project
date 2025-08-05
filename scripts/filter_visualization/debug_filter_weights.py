#!/usr/bin/env python3
"""
Debug Filter Weights - Diagnose why layer 0 filters look noisy

This script examines the raw filter weights to understand what's happening.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model


def inspect_model_architecture(model):
    """Inspect the model architecture to understand the layers"""
    print("\n🔍 MODEL ARCHITECTURE INSPECTION:")
    print("=" * 60)
    
    # Check input processor
    if hasattr(model, 'wave_input_processor'):
        print(f"📋 Wave Input Processor: {model.wave_input_processor}")
        for i, layer in enumerate(model.wave_input_processor):
            print(f"   [{i}] {layer}")
            if isinstance(layer, torch.nn.Conv2d):
                w = layer.weight.data
                print(f"       Weight shape: {w.shape}")
                print(f"       Weight range: [{w.min():.6f}, {w.max():.6f}]")
                print(f"       Weight mean: {w.mean():.6f}, std: {w.std():.6f}")
    
    print(f"\n📋 Model state dict keys:")
    for key in list(model.state_dict().keys())[:10]:  # First 10 keys
        print(f"   {key}")
    print("   ...")


def analyze_layer0_filters(model):
    """Deep analysis of layer 0 filters"""
    print("\n🔬 LAYER 0 FILTER ANALYSIS:")
    print("=" * 60)
    
    # Get layer 0 (input processor)
    if hasattr(model, 'wave_input_processor') and len(model.wave_input_processor) > 0:
        conv_layer = model.wave_input_processor[0]
        if isinstance(conv_layer, torch.nn.Conv2d):
            weights = conv_layer.weight.data.cpu()  # Shape: [out_channels, in_channels, H, W]
            
            print(f"📊 Layer 0 Conv2d Details:")
            print(f"   Input channels: {conv_layer.in_channels}")
            print(f"   Output channels: {conv_layer.out_channels}")
            print(f"   Kernel size: {conv_layer.kernel_size}")
            print(f"   Stride: {conv_layer.stride}")
            print(f"   Padding: {conv_layer.padding}")
            
            print(f"\n📊 Weight Tensor Analysis:")
            print(f"   Shape: {weights.shape}")
            print(f"   Data type: {weights.dtype}")
            print(f"   Device: {weights.device}")
            print(f"   Requires grad: {weights.requires_grad}")
            
            print(f"\n📊 Statistical Analysis:")
            print(f"   Min value: {weights.min():.6f}")
            print(f"   Max value: {weights.max():.6f}")
            print(f"   Mean: {weights.mean():.6f}")
            print(f"   Std: {weights.std():.6f}")
            print(f"   Median: {weights.median():.6f}")
            
            # Check for suspicious patterns
            print(f"\n🔍 Pattern Detection:")
            
            # Check if weights are all zeros or very close to initialization
            near_zero = (weights.abs() < 1e-6).float().mean()
            print(f"   Proportion near zero (<1e-6): {near_zero:.2%}")
            
            # Check if weights look like they haven't been trained (uniform distribution)
            weight_variance = weights.var()
            print(f"   Variance: {weight_variance:.6f}")
            
            # Analyze individual filters
            print(f"\n📊 Individual Filter Analysis (first 5 filters):")
            for i in range(min(5, weights.shape[0])):
                filter_weight = weights[i, 0]  # First input channel
                print(f"   Filter {i}: min={filter_weight.min():.4f}, max={filter_weight.max():.4f}, "
                      f"mean={filter_weight.mean():.4f}, std={filter_weight.std():.4f}")
                
                # Check for edge-like patterns
                # Calculate gradients to see if there are edge-like structures
                grad_x = torch.diff(filter_weight, dim=1)
                grad_y = torch.diff(filter_weight, dim=0)
                edge_strength = grad_x.abs().mean() + grad_y.abs().mean()
                print(f"             Edge strength: {edge_strength:.4f}")
            
            return weights
        else:
            print("❌ First layer is not Conv2d!")
            return None
    else:
        print("❌ No wave_input_processor found!")
        return None


def visualize_raw_filters(weights, num_filters=9):
    """Visualize raw filter weights with enhanced analysis"""
    print(f"\n🎨 ENHANCED FILTER VISUALIZATION:")
    print("=" * 60)
    
    if weights is None:
        return
    
    num_filters = min(num_filters, weights.shape[0])
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_filters):
        ax = axes[i]
        filter_weight = weights[i, 0].numpy()  # First input channel
        
        # Try different visualization approaches
        
        # Approach 1: Raw weights
        im = ax.imshow(filter_weight, cmap='RdBu_r', interpolation='nearest')
        
        # Enhanced title with more statistics
        f_min, f_max = filter_weight.min(), filter_weight.max()
        f_mean, f_std = filter_weight.mean(), filter_weight.std()
        
        ax.set_title(f'Filter {i}\nRange: [{f_min:.4f}, {f_max:.4f}]\n'
                    f'Mean: {f_mean:.4f}, Std: {f_std:.4f}', 
                    fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Remove unused subplots
    for i in range(num_filters, 9):
        axes[i].remove()
    
    plt.suptitle('Layer 0 Filter Weights - Raw Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save to debug folder
    debug_dir = Path("experiments/filter_visualization/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_path = debug_dir / "layer_0_debug_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Debug visualization saved: {save_path}")


def compare_with_random_filters():
    """Compare with random initialized filters to see the difference"""
    print(f"\n🎲 RANDOM FILTER COMPARISON:")
    print("=" * 60)
    
    # Create random filters with same dimensions as our model
    random_weights = torch.randn(32, 1, 7, 7) * 0.1  # Similar to Xavier initialization
    
    print(f"📊 Random Filter Statistics:")
    print(f"   Min: {random_weights.min():.6f}")
    print(f"   Max: {random_weights.max():.6f}")
    print(f"   Mean: {random_weights.mean():.6f}")
    print(f"   Std: {random_weights.std():.6f}")
    
    # Visualize a few random filters
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i in range(3):
        ax = axes[i]
        filter_weight = random_weights[i, 0].numpy()
        
        im = ax.imshow(filter_weight, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f'Random Filter {i}', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Random Initialized Filters (for comparison)', fontsize=14)
    plt.tight_layout()
    
    debug_dir = Path("experiments/filter_visualization/debug")
    save_path = debug_dir / "random_filters_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Random comparison saved: {save_path}")


def main():
    """Main debugging function"""
    print("🐛 FILTER WEIGHT DEBUGGING")
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
    print(f"📁 Model path: {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_wave_source_model(grid_size=128)
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"\n📊 Checkpoint Info:")
    print(f"   Keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded from 'model_state_dict'")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded directly from checkpoint")
    
    model = model.to(device).eval()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run diagnostic analyses
    inspect_model_architecture(model)
    weights = analyze_layer0_filters(model)
    visualize_raw_filters(weights)
    compare_with_random_filters()
    
    print(f"\n🎉 Debug analysis complete!")
    print(f"📁 Results saved in: experiments/filter_visualization/debug/")


if __name__ == "__main__":
    main() 