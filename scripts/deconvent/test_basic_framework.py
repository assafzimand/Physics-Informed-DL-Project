#!/usr/bin/env python3
"""
Test script for basic deconvent framework
Phase 1: Verify that we can load the model and extract filter activations
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from deconvent.wave_deconvent import WaveDeconventAnalyzer


def test_basic_framework():
    """Test that our basic deconvent framework can load and run."""
    print("🧪 Testing Wave Deconvent Framework - Phase 1")
    
    # Check if model exists
    model_path = "experiments/cv_full/data/models/cv_full_5fold_75epochs_fold_2_best.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("💡 Please ensure T=500 CV model is available")
        return False
    
    try:
        print("📥 Loading WaveSourceMiniResNet model...")
        analyzer = WaveDeconventAnalyzer(model_path, device='cpu')  # Use CPU for testing
        
        print("✅ Model loaded successfully!")
        print(f"🎯 Device: {analyzer.device}")
        
        # Test that hooks are registered
        print("🔗 Testing hook registration...")
        dummy_input = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            output = analyzer.wave_model(dummy_input)
        
        print(f"📊 Model output shape: {output.shape}")
        print(f"🎯 Captured activations: {list(analyzer.activations.keys())}")
        
        for stage, activation in analyzer.activations.items():
            print(f"   {stage}: {activation.shape}")
        
        # Test filter extraction
        print("🧬 Testing filter extraction...")
        filter_indices = [0, 15, 31]  # Test 3 filters from stage 2
        filter_features = analyzer.extract_filter_features(
            dummy_input, 'stage_2', filter_indices)
        
        print(f"✅ Extracted features for filters: {list(filter_features.keys())}")
        for filter_idx, features in filter_features.items():
            print(f"   Filter {filter_idx}: {features.shape}")
        
        print("\n🎉 Phase 1 Framework Test: PASSED!")
        print("📋 Next steps:")
        print("   1. Integrate with existing feature analysis data")
        print("   2. Implement up-convolutional reconstruction networks")
        print("   3. Train feature inverters for each stage")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_basic_framework()
    if success:
        print("\n✅ Basic framework is ready for Phase 2 development!")
    else:
        print("\n❌ Framework needs debugging before proceeding.")


if __name__ == "__main__":
    main() 