#!/usr/bin/env python3
"""
Export the WaveSourceMiniResNet Model to ONNX Format.

This script creates an instance of the `WaveSourceMiniResNet` model, initializes
it with random weights (no training is required for architecture visualization),
and exports it to the ONNX (Open Neural Network Exchange) format.

The resulting `.onnx` file can be opened with visualization tools like Netron
(https://netron.app) to interactively explore the model's architecture, including
layer types, shapes, and connections. This is invaluable for debugging,
documentation, and sharing model designs.

The script defines a dynamic batch axis, allowing the visualized model to show
that it can handle variable batch sizes.

Usage:
    python scripts/utils/model_export/export_model_to_onnx.py
"""

import sys
import torch
from pathlib import Path

# Ensure project root is on sys.path so src.* imports resolve correctly
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.wave_source_resnet import create_wave_source_model
from src.common.paths import get_project_root


def export_model_to_onnx(grid_size=128, output_dir: Path = None):
    """
    Creates and exports the WaveSourceMiniResNet to an ONNX file.

    Args:
        grid_size (int): The spatial dimension of the input grid (e.g., 128 for 128x128).
        output_dir (Path): The directory to save the ONNX file in. Defaults to project root.
    """
    if output_dir is None:
        output_dir = get_project_root()
    output_file = output_dir / "WaveSourceMiniResNet.onnx"

    print("🚀 Exporting WaveSourceMiniResNet to ONNX format...")

    # Create a model instance (no need to load trained weights for architecture view)
    print(f"🏗️  Creating model instance (grid_size={grid_size})")
    model = create_wave_source_model(grid_size=grid_size)
    model.eval()

    # Create a dummy input tensor with the expected shape
    # Input shape: [batch_size, channels, height, width] = [1, 1, 128, 128]
    print(f"📊 Creating dummy input tensor: [1, 1, {grid_size}, {grid_size}]")
    dummy_input = torch.randn(1, 1, grid_size, grid_size)

    # --- Export to ONNX ---
    print(f"💾 Exporting model to: {output_file}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_file),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["wave_field"],
        output_names=["coordinates"],
        dynamic_axes={
            "wave_field": {0: "batch_size"},
            "coordinates": {0: "batch_size"},
        },
    )

    print(f"✅ Successfully exported to {output_file}")
    print("\nNext Steps:")
    print("  1. Go to https://netron.app")
    print(f"  2. Drag and drop the '{output_file.name}' file into the browser.")
    print("  3. Explore your model architecture interactively!")


def main():
    """Main function to run the ONNX export."""
    export_model_to_onnx()


if __name__ == "__main__":
    main()
