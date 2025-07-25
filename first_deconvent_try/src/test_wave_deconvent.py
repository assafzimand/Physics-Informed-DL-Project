#!/usr/bin/env python3
"""
Test script for Wave Deconvent visualization.
Tests our fixes for identical filter outputs and sparsity issues.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.wave_source_resnet import WaveSourceMiniResNet
from src.deconvent.wave_deconvent import WaveDeconvNet
from src.data.wave_dataset import WaveDataset


def test_deconvent_fixes():
    """Test our deconvent fixes on wave field samples."""
    
    print("🧪 Testing Wave Deconvent Fixes...")
    
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = Path("results/models/best_cv_model.pth")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = WaveSourceMiniResNet()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Create deconvent
    deconvent = WaveDeconvNet(model)
    deconvent.to(device)
    deconvent.eval()
    
    # Load test samples
    dataset_path = Path("data/wave_dataset_t250_test.pkl")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    dataset = WaveDataset(str(dataset_path))
    
    # Test on 20 samples with different wave patterns
    print("\n🔍 Testing on 20 diverse wave samples...")
    
    # Select diverse samples (different positions to get varied patterns)
    test_indices = [0, 10, 25, 50, 75, 100, 150, 200, 250, 300,
                    400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
    
    sample_idx = 0
    stage_to_test = 2  # Test Stage 2 (good middle layer)
    num_filters_to_show = 9  # 3x3 grid
    
    for i, data_idx in enumerate(test_indices[:5]):  # Test first 5 for now
        if data_idx >= len(dataset):
            continue
            
        wave_field, source_pos = dataset[data_idx]
        wave_field = wave_field.unsqueeze(0).to(device)  # Add batch dim
        
        print(f"Sample {i+1}: Source at ({source_pos[0]:.1f}, {source_pos[1]:.1f})")
        
        # Generate filter visualizations
        try:
            filter_patterns = deconvent.visualize_filters(
                wave_field, 
                stage=stage_to_test, 
                num_filters=num_filters_to_show
            )
            
            # Create visualization
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes = axes.flatten()
            
            for f_idx in range(num_filters_to_show):
                pattern = filter_patterns[f_idx].cpu().numpy()
                
                # Check for issues
                pattern_min, pattern_max = pattern.min(), pattern.max()
                pattern_std = pattern.std()
                
                axes[f_idx].imshow(pattern, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
                axes[f_idx].set_title(f'Filter {f_idx}\nRange:[{pattern_min:.3f}, {pattern_max:.3f}]\nStd:{pattern_std:.3f}')
                axes[f_idx].axis('off')
                
                # Check for identical patterns (our main bug)
                if f_idx > 0:
                    prev_pattern = filter_patterns[0].cpu().numpy()
                    if np.allclose(pattern, prev_pattern, atol=1e-6):
                        print(f"⚠️  Filter {f_idx} identical to Filter 0!")
                
                # Check for sparsity (dots issue)
                non_zero_ratio = np.count_nonzero(np.abs(pattern) > 1e-6) / pattern.size
                if non_zero_ratio < 0.1:
                    print(f"⚠️  Filter {f_idx} very sparse: {non_zero_ratio:.3f} non-zero ratio")
            
            plt.tight_layout()
            plt.suptitle(f'Stage {stage_to_test} Filters - Sample {i+1} (Pos: {source_pos[0]:.1f}, {source_pos[1]:.1f})', 
                        fontsize=16, y=0.98)
            
            # Save results
            output_path = Path("results/deconvent_test_fixes")
            output_path.mkdir(exist_ok=True)
            plt.savefig(output_path / f"sample_{i+1}_stage{stage_to_test}_filters_fixed.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved: sample_{i+1}_stage{stage_to_test}_filters_fixed.png")
            
        except Exception as e:
            print(f"❌ Error processing sample {i+1}: {e}")
            continue
    
    print(f"\n🎯 Test complete! Check results/deconvent_test_fixes/ for visualizations.")
    print("\n📊 Expected improvements:")
    print("✅ Filters should look DIFFERENT (not identical)")
    print("✅ Less sparsity/dots (more continuous patterns)")
    print("✅ Better wave-like structures")

if __name__ == "__main__":
    test_deconvent_fixes() 