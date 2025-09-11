#!/usr/bin/env python3
"""
Visualize WaveSourceMiniResNet with Torchviz.

This script generates a detailed computational graph of the `WaveSourceMiniResNet`
architecture using the `torchviz` library.

The output is a set of files (PNG, PDF, SVG) that visually represent every
operation and tensor flow within the model. This is extremely useful for
debugging, understanding the model's structure, and for documentation.

The script will automatically attempt to install `torchviz` if it's not found.
However, for rendering image formats like PNG and PDF, a system-level installation
of Graphviz is required.

Usage:
    python scripts/utils/model_export/visualize_with_torchviz.py

Requirements:
    - `pip install torchviz`
    - For image rendering: Graphviz (e.g., `choco install graphviz` on Windows or
      `brew install graphviz` on macOS).
"""

import sys
import torch
from pathlib import Path

# Ensure project root is on sys.path so src.* imports resolve correctly
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.wave_source_resnet import create_wave_source_model
from src.common.paths import get_project_root

# Check if torchviz is available and install it if not
try:
    from torchviz import make_dot
except ImportError:
    print("⚠️ Torchviz not found. Attempting to install...")
    import subprocess

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchviz"])
        from torchviz import make_dot

        print("✅ Torchviz installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install torchviz: {e}")
        print("Please install it manually: pip install torchviz")
        sys.exit(1)


def create_torchviz_diagram(grid_size=128, output_dir: Path = None):
    """
    Creates a Torchviz visualization of the WaveSourceMiniResNet.

    Args:
        grid_size (int): The spatial dimension of the input grid.
        output_dir (Path): The directory to save the visualization files.
                           Defaults to `docs/images`.
    """
    if output_dir is None:
        output_dir = get_project_root() / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎨 Creating Torchviz visualization of WaveSourceMiniResNet...")

    # Create a model instance
    print(f"🏗️  Creating model instance (grid_size={grid_size})")
    model = create_wave_source_model(grid_size=grid_size)
    model.eval()

    # Create a sample input tensor
    print(f"📊 Creating sample input tensor: [1, 1, {grid_size}, {grid_size}]")
    sample_input = torch.randn(1, 1, grid_size, grid_size, requires_grad=True)

    # Run a forward pass to build the graph
    print("🔄 Running forward pass...")
    output = model(sample_input)

    # Generate the computational graph
    print("🎯 Generating computational graph...")
    dot = make_dot(
        output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True
    )

    # --- Customize Graph Appearance ---
    dot.graph_attr.update(
        {
            "rankdir": "TB",  # Top to Bottom layout
            "dpi": "300",
            "bgcolor": "transparent",
        }
    )
    dot.node_attr.update(
        {
            "style": "filled",
            "shape": "box",
            "fillcolor": "#AED6F1",  # Light blue
            "fontname": "Arial",
            "fontsize": "10",
        }
    )
    dot.edge_attr.update({"fontname": "Arial", "fontsize": "8"})

    # --- Save Visualization in Multiple Formats ---
    output_filename_base = output_dir / "WaveSourceMiniResNet_Architecture"
    print(f"💾 Saving visualization to {output_filename_base}.* ...")

    formats = ["png", "pdf", "svg"]
    for fmt in formats:
        try:
            dot.render(str(output_filename_base), format=fmt, cleanup=True)
            print(f"   ✅ Saved: {output_filename_base.name}.{fmt}")
        except Exception as e:
            print(f"   ❌ Failed to save {fmt}: {e}")
            print("      Ensure Graphviz is installed and in your system's PATH.")

    print("\n🎉 Torchviz visualization complete!")
    print(f"   -> Files saved in: {output_dir}")


def main():
    """Main function to create the visualization."""
    create_torchviz_diagram()


if __name__ == "__main__":
    main()
