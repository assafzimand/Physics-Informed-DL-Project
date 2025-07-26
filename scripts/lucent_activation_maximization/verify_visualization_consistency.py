#!/usr/bin/env python3
"""
Verification script to demonstrate the difference between:
1. OLD: Visualizing only R channel (inconsistent with model)
2. NEW: Visualizing R+G+B sum (consistent with model)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_visualization_difference():
    """Show the difference between R-only vs R+G+B visualization"""
    
    print("🔍 Demonstrating Visualization Consistency")
    print("=" * 60)
    
    # Create a sample RGB pattern (simulating Lucent output)
    np.random.seed(42)  # For reproducible results
    
    # Simulate an RGB pattern with different content in each channel
    R = np.random.randn(128, 128) * 10 + 5   # Red channel
    G = np.random.randn(128, 128) * 8 - 2    # Green channel  
    B = np.random.randn(128, 128) * 6 + 3    # Blue channel
    
    rgb_pattern = np.stack([R, G, B], axis=0)  # Shape: [3, 128, 128]
    
    print(f"📊 Sample RGB Pattern Statistics:")
    print(f"   R channel: mean={R.mean():.3f}, std={R.std():.3f}")
    print(f"   G channel: mean={G.mean():.3f}, std={G.std():.3f}")
    print(f"   B channel: mean={B.mean():.3f}, std={B.std():.3f}")
    
    # Old method: R channel only
    old_visualization = R  # pattern_np[0]
    
    # New method: R+G+B sum (what model sees)
    new_visualization = R + G + B  # pattern_np[0] + pattern_np[1] + pattern_np[2]
    
    print(f"\n🎯 Visualization Comparison:")
    print(f"   OLD (R only):  mean={old_visualization.mean():.3f}, std={old_visualization.std():.3f}")
    print(f"   NEW (R+G+B):   mean={new_visualization.mean():.3f}, std={new_visualization.std():.3f}")
    print(f"   Difference:    mean={(new_visualization - old_visualization).mean():.3f}")
    
    # Show what the model wrapper actually computes
    rgb_tensor = torch.tensor(rgb_pattern).unsqueeze(0)  # Add batch dim: [1, 3, 128, 128]
    
    # Simulate wrapper conversion
    model_input = rgb_tensor[:, 0:1] + rgb_tensor[:, 1:2] + rgb_tensor[:, 2:3]
    model_sees = model_input[0, 0].numpy()  # Convert back to numpy
    
    print(f"   Model sees:    mean={model_sees.mean():.3f}, std={model_sees.std():.3f}")
    print(f"   Match? {np.allclose(model_sees, new_visualization)}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Individual channels
    im1 = axes[0, 0].imshow(R, cmap='RdBu_r')
    axes[0, 0].set_title('R Channel')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(G, cmap='RdBu_r')
    axes[0, 1].set_title('G Channel')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(B, cmap='RdBu_r')
    axes[0, 2].set_title('B Channel')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Bottom row: Visualizations
    im4 = axes[1, 0].imshow(old_visualization, cmap='RdBu_r')
    axes[1, 0].set_title('OLD: R Only\n(What we used to show)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(new_visualization, cmap='RdBu_r')
    axes[1, 1].set_title('NEW: R+G+B Sum\n(What model sees)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Difference
    difference = new_visualization - old_visualization
    im6 = axes[1, 2].imshow(difference, cmap='RdBu_r')
    axes[1, 2].set_title('Difference\n(NEW - OLD)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save comparison
    output_path = Path("experiments/activation_maximization/visualization_consistency_check.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Saved comparison plot: {output_path}")
    print(f"✅ Verification complete!")
    print(f"\n🎯 Key Insight:")
    print(f"   The NEW method shows the actual wave field that maximally")
    print(f"   activates the target filter, ensuring complete consistency")
    print(f"   between optimization and visualization!")

if __name__ == "__main__":
    demonstrate_visualization_difference() 