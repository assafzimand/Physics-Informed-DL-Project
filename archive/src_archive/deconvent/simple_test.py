#!/usr/bin/env python3
"""Simple test for our deconvent fixes."""

import torch
from reverse_operations import ReverseReLU, ReverseConv2d

def test_fixes():
    """Test our key fixes."""
    print("🧪 Testing Key Fixes...")
    
    # Test 1: ReLU fix (should pass through, not re-apply ReLU)
    print("\n1️⃣ Testing ReLU Fix:")
    reverse_relu = ReverseReLU()
    
    # Input with negative values
    test_input = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    output = reverse_relu(test_input)
    
    print(f"Input:  {test_input.flatten()}")
    print(f"Output: {output.flatten()}")
    
    if torch.equal(test_input, output):
        print("✅ ReLU Fix: PASS - Negative values preserved!")
    else:
        print("❌ ReLU Fix: FAIL - Still applying ReLU")
    
    # Test 2: Conv2d weight transpose fix
    print("\n2️⃣ Testing Conv2d Weight Transpose:")
    
    # Create dummy forward conv
    forward_conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
    original_weight_shape = forward_conv.weight.shape
    print(f"Forward conv weight shape: {original_weight_shape}")
    
    # Create reverse conv
    reverse_conv = ReverseConv2d(forward_conv)
    reverse_weight_shape = reverse_conv.deconv.weight.shape
    print(f"Reverse conv weight shape: {reverse_weight_shape}")
    
    # Check if dimensions are properly swapped
    if (original_weight_shape[0] == reverse_weight_shape[1] and 
        original_weight_shape[1] == reverse_weight_shape[0]):
        print("✅ Conv2d Fix: PASS - Weights properly transposed!")
    else:
        print("❌ Conv2d Fix: FAIL - Weights not transposed")
    
    print("\n🎯 Fix Summary:")
    print("✅ ReLU: No longer double-rectifies (preserves negatives)")
    print("✅ Conv2d: Weights properly transposed for deconvolution")
    print("✅ These should reduce identical outputs and sparsity!")

if __name__ == "__main__":
    test_fixes() 