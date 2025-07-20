"""
Test script for WaveSourceMiniResNet model

Tests model creation, forward pass, and basic functionality.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.wave_source_resnet import create_wave_source_model


def test_model():
    """Test the model with dummy input."""
    print("Testing WaveSourceMiniResNet model...")
    
    model = create_wave_source_model()
    
    # Create dummy input (batch_size=2, channels=1, height=128, width=128)
    dummy_input = torch.randn(2, 1, 128, 128)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print("✓ Model created successfully!")
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    out_min, out_max = output.min().item(), output.max().item()
    print(f"✓ Output range: [{out_min:.2f}, {out_max:.2f}]")
    print(f"✓ Total parameters: {model.get_num_parameters():,}")
    
    # Check available activations
    activations = list(model.activations.keys())
    print(f"✓ Available activations: {activations}")
    
    # Verify output constraints
    expected_shape = (2, 2)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, got {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 127), \
        f"Output coordinates should be in range [0, 127], " \
        f"got [{out_min:.2f}, {out_max:.2f}]"
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_model() 