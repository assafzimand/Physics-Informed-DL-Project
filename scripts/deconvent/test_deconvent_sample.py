#!/usr/bin/env python3
"""
Simple test to visualize deconvent results after our fixes.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append('src/deconvent')

from models.wave_source_resnet import WaveSourceMiniResNet
from data.wave_dataset import WaveDataset
from deconvent.wave_deconvent_blocks import WaveDeconvolutionalNetwork
from deconvent.reverse_operations import ReverseReLU

def test_single_sample():
    """Test deconvent on a single sample to see our fixes."""
    
    print("🧪 Testing Deconvent Fixes on Single Sample...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Quick verification our fixes are active
    print("\n🔧 Verifying Fixes:")
    reverse_relu = ReverseReLU()
    test_input = torch.tensor([-1.0, 2.0])
    output = reverse_relu(test_input)
    if torch.equal(test_input, output):
        print("✅ ReLU fix active: passes negatives through")
    else:
        print("❌ ReLU fix NOT active: still clipping")
    
    # Load model
    model_path = Path("results/models/best_cv_model.pth") 
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Available model files:")
        models_dir = Path("results/models")
        if models_dir.exists():
            for f in models_dir.iterdir():
                print(f"  - {f}")
        return
    
    print(f"📁 Loading model: {model_path}")
    model = WaveSourceMiniResNet()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load dataset
    dataset_path = Path("data/wave_dataset_t250_test.pkl")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Available dataset files:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.iterdir():
                if f.suffix == '.pkl':
                    print(f"  - {f}")
        return
    
    print(f"📁 Loading dataset: {dataset_path}")
    dataset = WaveDataset(str(dataset_path))
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Create deconvent network
    print("\n🔄 Creating deconvent network...")
    deconvent = WaveDeconvolutionalNetwork(model)
    deconvent.to(device)
    deconvent.eval()
    print("✅ Deconvent created")
    
    # Test on one sample
    sample_idx = 100  # Pick a middle sample
    wave_field, source_pos = dataset[sample_idx]
    wave_field_batch = wave_field.unsqueeze(0).to(device)
    
    print(f"\n🌊 Testing sample {sample_idx}")
    print(f"Source position: ({source_pos[0]:.1f}, {source_pos[1]:.1f})")
    print(f"Wave field shape: {wave_field.shape}")
    
    # Generate filter visualizations for Stage 2
    stage = 2
    num_filters = 9
    
    print(f"\n🔍 Generating {num_filters} filter visualizations for Stage {stage}...")
    
    try:
        # Get stage activations
        activations = deconvent.forward_to_stage(wave_field_batch, stage)
        print(f"Stage {stage} activations shape: {activations.shape}")
        
        # Visualize filters
        filter_patterns = deconvent.visualize_filters(
            wave_field_batch, stage=stage, num_filters=num_filters
        )
        
        print(f"Generated {len(filter_patterns)} filter patterns")
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        identical_count = 0
        sparse_count = 0
        
        for i in range(num_filters):
            pattern = filter_patterns[i].cpu().numpy()
            
            # Analysis
            pattern_min, pattern_max = pattern.min(), pattern.max()
            pattern_std = pattern.std()
            pattern_range = pattern_max - pattern_min
            
            # Check for identical patterns (main bug we fixed)
            if i > 0:
                first_pattern = filter_patterns[0].cpu().numpy()
                if np.allclose(pattern, first_pattern, atol=1e-6):
                    identical_count += 1
                    print(f"⚠️  Filter {i} identical to Filter 0")
            
            # Check for sparsity (dots issue)
            non_zero_ratio = np.count_nonzero(np.abs(pattern) > 1e-6) / pattern.size
            if non_zero_ratio < 0.1:
                sparse_count += 1
                print(f"⚠️  Filter {i} sparse: {non_zero_ratio:.3f} non-zero ratio")
            
            # Plot
            axes[i].imshow(pattern, cmap='RdBu_r', 
                          vmin=-pattern_range/2, vmax=pattern_range/2)
            axes[i].set_title(f'Filter {i}\nRange: [{pattern_min:.4f}, {pattern_max:.4f}]\n'
                             f'Std: {pattern_std:.4f}\nNZ: {non_zero_ratio:.3f}', 
                             fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Stage {stage} Filters - Sample {sample_idx} (AFTER FIXES)\n'
                    f'Source: ({source_pos[0]:.1f}, {source_pos[1]:.1f})', 
                    fontsize=16, y=0.98)
        
        # Save
        output_path = Path("results/deconvent_fixes_test")
        output_path.mkdir(exist_ok=True)
        save_path = output_path / f"stage{stage}_sample{sample_idx}_FIXED.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"✅ Saved visualization: {save_path}")
        print(f"🔍 Identical filters: {identical_count}/{num_filters-1}")
        print(f"🔍 Sparse filters: {sparse_count}/{num_filters}")
        
        if identical_count == 0:
            print("🎉 SUCCESS: No identical filters! Fix #1 WORKED!")
        else:
            print("⚠️  Still have identical filters - may need more fixes")
            
        if sparse_count < num_filters//2:
            print("🎉 SUCCESS: Reduced sparsity! Fixes are working!")
        else:
            print("⚠️  Still quite sparse - may need ReLU or MaxPool improvements")
        
    except Exception as e:
        print(f"❌ Error during deconvent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_sample() 