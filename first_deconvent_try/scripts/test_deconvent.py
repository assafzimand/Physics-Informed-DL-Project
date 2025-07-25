#!/usr/bin/env python3
"""
Test script for WaveDeconvolutionalNetwork implementation.
Verifies that our deconvent approach can visualize filter patterns.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')

# Import after path setup
from models.wave_source_resnet import WaveSourceMiniResNet
from deconvent.wave_deconvent_blocks import WaveDeconvolutionalNetwork


def load_test_sample():
    """Load a test wave field sample for visualization."""
    # Use our consolidated 20-sample dataset
    dataset_path = "data/wave_dataset_analysis_20samples.h5"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Please ensure the 20-sample analysis dataset exists")
        return None
        
    print(f"📂 Loading from: {dataset_path}")
    
    try:
        with h5py.File(dataset_path, 'r') as f:
            # Load first sample
            wave_field = f['wave_fields'][0]  # Shape: [1, 128, 128]
            source_x = f['coordinates'][0, 0]
            source_y = f['coordinates'][0, 1]
            
        # Convert to tensor and add batch dimension (channel already exists)
        wave_field = torch.from_numpy(wave_field).float()
        wave_field = wave_field.unsqueeze(0)  # [1, 1, 128, 128]
        
        print("✅ Loaded sample 0")
        print(f"   Wave field shape: {wave_field.shape}")
        print(f"   Source location: ({source_x:.1f}, {source_y:.1f})")
        
        return wave_field
        
    except Exception as e:
        print(f"❌ Failed to load sample: {e}")
        return None


def test_basic_deconvent():
    """Test basic deconvolutional network functionality."""
    print("🧪 Testing Basic Deconvent Functionality")
    print("="*50)
    
    # Load trained model
    model_path = ("experiments/cv_full/data/models/"
                  "cv_full_5fold_75epochs_fold_2_best.pth")
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
        
    print(f"📂 Loading model: {model_path}")
    
    # Create and load model
    model = WaveSourceMiniResNet(grid_size=128)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Create deconvolutional network
    print("🔄 Creating deconvolutional network...")
    try:
        deconvent = WaveDeconvolutionalNetwork(model)
        print("✅ Deconvolutional network created successfully")
    except Exception as e:
        print(f"❌ Failed to create deconvent: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load test sample
    wave_field = load_test_sample()
    if wave_field is None:
        print("❌ Failed to load test sample")
        return False
    
    # Test forward pass to different stages
    print("\n🧪 Testing forward passes to different stages:")
    
    for stage in range(5):
        try:
            activations = deconvent.forward_to_stage(wave_field, stage)
            print(f"   Stage {stage}: {activations.shape}")
        except Exception as e:
            print(f"   ❌ Stage {stage} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("✅ All forward passes successful")
    
    # Test filter visualization
    print("\n🎨 Testing filter visualization:")
    
    try:
        # Test visualization of a filter from Stage 2
        target_stage = 2
        filter_idx = 0
        
        print(f"   Visualizing Stage {target_stage}, Filter {filter_idx}...")
        
        reconstructed = deconvent.visualize_filter(
            wave_field, target_stage, filter_idx
        )
        
        print(f"   ✅ Reconstruction shape: {reconstructed.shape}")
        print(f"   ✅ Reconstruction range: [{reconstructed.min():.3f}, "
              f"{reconstructed.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ Filter visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 All tests passed!")
    return True


def visualize_sample_filters():
    """Create a sample visualization of filter patterns."""
    print("\n🎨 Creating Sample Filter Visualizations")
    print("="*50)
    
    # Load model and create deconvent
    model_path = ("experiments/cv_full/data/models/"
                  "cv_full_5fold_75epochs_fold_2_best.pth")
    
    model = WaveSourceMiniResNet(grid_size=128)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    deconvent = WaveDeconvolutionalNetwork(model)
    
    # Load test sample
    wave_field = load_test_sample()
    if wave_field is None:
        return False
    
    # Create visualization for a few filters from Stage 2
    stage = 2
    filter_indices = [0, 5, 10]  # Test a few different filters
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Plot original wave field
    wave_np = wave_field[0, 0].numpy()
    axes[0, 0].imshow(wave_np, cmap='RdBu_r', aspect='equal')
    axes[0, 0].set_title('Original Wave Field')
    axes[0, 0].axis('off')
    
    # Plot filter visualizations
    for i, filter_idx in enumerate(filter_indices):
        try:
            print(f"   Generating visualization for Filter {filter_idx}...")
            
            reconstructed = deconvent.visualize_filter(
                wave_field, stage, filter_idx
            )
            
            # Plot in second row
            recon_np = reconstructed[0, 0].detach().numpy()
            axes[1, i].imshow(recon_np, cmap='RdBu_r', aspect='equal')
            axes[1, i].set_title(f'Filter {filter_idx}\nPattern')
            axes[1, i].axis('off')
            
        except Exception as e:
            print(f"   ❌ Failed for Filter {filter_idx}: {e}")
            axes[1, i].text(0.5, 0.5, f'Error\nFilter {filter_idx}', 
                           ha='center', va='center', 
                           transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
    
    # Remove unused subplots
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("experiments/deconvent")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "test_deconvent_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sample visualization saved: {output_file}")
    return True


def main():
    """Main test function."""
    print("🚀 Testing WaveDeconvolutionalNetwork Implementation")
    print("="*60)
    
    # Test 1: Basic functionality
    if not test_basic_deconvent():
        print("❌ Basic tests failed")
        return False
    
    # Test 2: Create sample visualizations
    if not visualize_sample_filters():
        print("❌ Visualization test failed")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("💡 Deconvolutional network is ready for filter analysis!")
    
    return True


if __name__ == "__main__":
    main() 