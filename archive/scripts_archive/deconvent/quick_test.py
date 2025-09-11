#!/usr/bin/env python3
"""Quick test of deconvent fixes."""

import torch
import sys
sys.path.append('src/deconvent')
from reverse_operations import ReverseReLU, ReverseConv2d

def test_fixes():
    print("🧪 Testing Our Deconvent Fixes...")
    
    # Test 1: ReLU should pass through (not re-apply ReLU)
    print("\n1️⃣ ReLU Fix Test:")
    reverse_relu = ReverseReLU()
    test_input = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    output = reverse_relu(test_input)
    
    print(f"Input:  {test_input.flatten().tolist()}")
    print(f"Output: {output.flatten().tolist()}")
    
    if torch.equal(test_input, output):
        print("✅ ReLU Fix WORKS: Negatives preserved!")
    else:
        print("❌ ReLU Fix BROKEN: Still clipping negatives")
    
    # Test 2: Conv2d weights should be transposed
    print("\n2️⃣ Conv2d Weight Transpose Test:")
    forward_conv = torch.nn.Conv2d(3, 8, kernel_size=3)
    orig_shape = forward_conv.weight.shape
    
    reverse_conv = ReverseConv2d(forward_conv)
    rev_shape = reverse_conv.deconv.weight.shape
    
    print(f"Original: {orig_shape} → Reverse: {rev_shape}")
    
    if orig_shape[0] == rev_shape[1] and orig_shape[1] == rev_shape[0]:
        print("✅ Conv2d Fix WORKS: Weights properly transposed!")
    else:
        print("❌ Conv2d Fix BROKEN: Weights not transposed")
    
    print("\n🎯 Expected Results:")
    print("✅ Filters should look DIFFERENT (not identical)")
    print("✅ Less sparse/dotty patterns")
    print("✅ Better wave-like structures")

if __name__ == "__main__":
    test_fixes() 