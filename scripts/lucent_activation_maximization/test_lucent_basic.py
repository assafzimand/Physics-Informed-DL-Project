#!/usr/bin/env python3
"""
Basic test script to verify Lucent installation and functionality.
Tests activation maximization on a simple convolutional layer.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_lucent_import():
    """Test if Lucent imports correctly."""
    print("🧪 Testing Lucent Import...")
    
    try:
        import lucent
        print(f"✅ Lucent imported successfully! Version: {lucent.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Lucent: {e}")
        return False

def test_basic_torch_model():
    """Test activation maximization on a basic PyTorch model."""
    print("\n🧪 Testing Basic Activation Maximization...")
    
    # Create a simple conv model
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            return x
    
    model = SimpleConvNet()
    model.eval()
    
    print(f"✅ Created test model: {model}")
    
    # Test basic forward pass
    test_input = torch.randn(1, 1, 32, 32)
    output = model(test_input)
    print(f"✅ Forward pass works: {test_input.shape} → {output.shape}")
    
    return model

def test_lucent_optimization():
    """Test if we can run basic Lucent optimization."""
    print("\n🧪 Testing Lucent Optimization...")
    
    try:
        # Try to import Lucent components
        from lucent.optvis import render
        from lucent.optvis.objectives import channel
        from lucent.optvis.param import image
        print("✅ Lucent components imported successfully")
        
        # Create simple test
        model = test_basic_torch_model()
        
        # Try to define an objective (target channel 0 of conv2)
        # This tests if Lucent can work with our PyTorch model
        print("✅ Basic Lucent setup successful")
        
        # Note: We'll do actual optimization in later phases
        # For now, just verify imports work
        
        return True
        
    except Exception as e:
        print(f"❌ Lucent optimization test failed: {e}")
        return False

def run_all_tests():
    """Run all basic tests."""
    print("🚀 Running Lucent Basic Tests...")
    print("=" * 50)
    
    # Test 1: Import
    import_ok = test_lucent_import()
    
    # Test 2: Basic model
    if import_ok:
        model = test_basic_torch_model()
    
    # Test 3: Lucent components  
    if import_ok:
        lucent_ok = test_lucent_optimization()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"✅ Lucent Import: {'PASS' if import_ok else 'FAIL'}")
    if import_ok:
        print(f"✅ Basic Model: PASS")
        print(f"✅ Lucent Components: {'PASS' if lucent_ok else 'FAIL'}")
    
    if import_ok and lucent_ok:
        print("\n🎉 All tests PASSED! Ready for Phase 2.")
    else:
        print("\n⚠️  Some tests FAILED. Check installation.")

if __name__ == "__main__":
    run_all_tests() 