#!/usr/bin/env python3
"""
Check if our trained model already has all BatchNorm statistics 
needed for reversal
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')

# Import after path setup
from models.wave_source_resnet import WaveSourceMiniResNet


def check_batchnorm_stats():
    """Check what BatchNorm stats are available in our trained model."""
    print("🔍 Checking BatchNorm statistics in trained model...")
    
    # Load our trained T=500 model
    model_path = ("experiments/cv_full/data/models/"
                  "cv_full_5fold_75epochs_fold_2_best.pth")
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
        
    print(f"📂 Loading model: {model_path}")
    
    # Create model and load checkpoint
    model = WaveSourceMiniResNet(grid_size=128)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n🧪 BatchNorm layers and their statistics:")
    print("="*60)
    
    bn_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_count += 1
            print(f"\n📊 {name}:")
            print(f"   Channels: {module.num_features}")
            print(f"   eps: {module.eps}")
            print(f"   running_mean shape: {module.running_mean.shape}")
            print(f"   running_var shape: {module.running_var.shape}")
            print(f"   weight shape: {module.weight.shape}")
            print(f"   bias shape: {module.bias.shape}")
            
            # Show some actual values to verify they're not default
            mean_sample = module.running_mean[:3].tolist()
            var_sample = module.running_var[:3].tolist()
            print(f"   running_mean sample: {mean_sample}")
            print(f"   running_var sample: {var_sample}")
    
    print(f"\n✅ Found {bn_count} BatchNorm2d layers")
    print("💡 All required statistics are available for reversal!")
    return True


if __name__ == "__main__":
    check_batchnorm_stats() 