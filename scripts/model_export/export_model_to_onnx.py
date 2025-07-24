#!/usr/bin/env python3
"""
Export WaveSourceMiniResNet to ONNX for Visualization
Converts our PyTorch model to ONNX format for interactive visualization with Netron.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from models.wave_source_resnet import WaveSourceMiniResNet


def export_model_to_onnx(grid_size=128, output_file="WaveSourceMiniResNet.onnx"):
    """Export WaveSourceMiniResNet to ONNX format."""
    print("🚀 Exporting WaveSourceMiniResNet to ONNX format...")
    
    # Create model
    print(f"🏗️  Creating model (grid_size={grid_size})")
    model = WaveSourceMiniResNet(grid_size=grid_size)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input that matches expected input shape
    # Input: [batch_size, channels, height, width] = [1, 1, 128, 128]
    print(f"📊 Creating dummy input: [1, 1, {grid_size}, {grid_size}]")
    dummy_input = torch.randn(1, 1, grid_size, grid_size)
    
    # Test forward pass to make sure model works
    print("🧪 Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Model output shape: {output.shape}")
    
    # Export to ONNX
    print(f"💾 Exporting to {output_file}...")
    torch.onnx.export(
        model,                          # model being run
        dummy_input,                    # model input (or a tuple for multiple inputs)
        output_file,                    # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=11,              # the ONNX version to export the model to
        do_constant_folding=True,      # whether to execute constant folding for optimization
        input_names=['wave_field'],     # the model's input names
        output_names=['coordinates'],   # the model's output names
        dynamic_axes={
            'wave_field': {0: 'batch_size'},     # variable length axes
            'coordinates': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Successfully exported to {output_file}")
    print(f"🌐 Next steps:")
    print(f"   1. Go to https://netron.app")
    print(f"   2. Drag {output_file} into the browser window")
    print(f"   3. Explore your model architecture interactively!")
    
    return output_file


def main():
    """Main function to export model."""
    print("🎨 Creating ONNX visualization of WaveSourceMiniResNet...")
    
    # Export with default parameters
    output_file = export_model_to_onnx(grid_size=128, output_file="WaveSourceMiniResNet.onnx")
    
    # Check if file was created
    if Path(output_file).exists():
        file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
        print(f"\n🎉 Export complete!")
        print(f"📁 File: {output_file} ({file_size:.1f} MB)")
        print(f"🎯 Ready for visualization at https://netron.app")
    else:
        print(f"❌ Export failed - file not found: {output_file}")


if __name__ == "__main__":
    main() 