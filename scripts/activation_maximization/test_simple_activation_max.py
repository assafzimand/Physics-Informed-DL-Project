#!/usr/bin/env python3
"""
Test script for simple single-channel activation maximization.
This bypasses Lucent entirely and works directly with wave fields.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.activation_maximization.simple_activation_max import run_simple_activation_maximization
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model


def test_simple_activation_maximization():
    """Test our custom single-channel activation maximization"""
    
    print("🚀 Simple Single-Channel Activation Maximization Test")
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
    print(f"   📄 Path: {model_path}")
    
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
    
    # Test configuration
    layer_path = "wave_input_processor.0"  # Very first conv layer that sees raw input
    filter_idx = 5
    iterations = 1024
    
    print(f"\n🎯 Test Configuration:")
    print(f"   • Layer: {layer_path}")
    print(f"   • Filter: {filter_idx}")
    print(f"   • Iterations: {iterations}")
    print(f"   • Device: {device}")
    
    # Run optimization
    print(f"\n🚀 Running Simple Activation Maximization...")
    try:
        optimized_pattern, loss_history = run_simple_activation_maximization(
            model, layer_path, filter_idx, iterations=iterations, device=device
        )
        
        print(f"✅ Optimization completed successfully!")
        print(f"   Pattern shape: {optimized_pattern.shape}")
        print(f"   Loss history: {len(loss_history)} values")
        
        # Plot results
        print(f"\n📊 Creating visualizations...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot optimized pattern
        pattern = optimized_pattern[0, 0].cpu().numpy()  # Remove batch and channel dims
        im1 = ax1.imshow(pattern, cmap='RdBu_r', interpolation='bilinear')
        ax1.set_title(f'Optimized Wave Pattern\n{layer_path} Filter {filter_idx}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        plt.colorbar(im1, ax=ax1, label='Wave Amplitude')
        
        # Add pattern statistics
        stats_text = f'Range: [{pattern.min():.3f}, {pattern.max():.3f}]\n'
        stats_text += f'Mean: {pattern.mean():.3f}\n'
        stats_text += f'Std: {pattern.std():.3f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot loss curve
        ax2.plot(loss_history, 'b-', linewidth=2)
        ax2.set_title('Optimization Loss Curve', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        # Add loss statistics
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        loss_reduction = initial_loss - final_loss
        
        loss_text = f'Initial: {initial_loss:.4f}\n'
        loss_text += f'Final: {final_loss:.4f}\n'
        loss_text += f'Reduction: {loss_reduction:.4f}'
        ax2.text(0.98, 0.98, loss_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save results
        output_dir = Path("experiments/activation_maximization")
        output_dir.mkdir(exist_ok=True)
        
        save_path = output_dir / f"simple_activation_max_{layer_path.replace('.', '_')}_filter_{filter_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved visualization: {save_path}")
        print(f"\n📊 Pattern Analysis:")
        print(f"   • Amplitude range: {pattern.max() - pattern.min():.4f}")
        print(f"   • Mean amplitude: {pattern.mean():.4f}")
        print(f"   • Std amplitude: {pattern.std():.4f}")
        print(f"   • Loss reduction: {loss_reduction:.4f}")
        
        print(f"\n🎉 Simple Activation Maximization Test COMPLETED!")
        print(f"📁 Results saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_activation_maximization() 