#!/usr/bin/env python3
"""
Visualize WaveSourceMiniResNet with Torchviz
Creates a computational graph visualization of the model architecture.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from models.wave_source_resnet import WaveSourceMiniResNet

# Check if torchviz is available
try:
    from torchviz import make_dot
except ImportError:
    print("❌ Torchviz not installed! Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchviz"])
    from torchviz import make_dot


def create_torchviz_diagram(grid_size=128, output_dir="scripts/model_export"):
    """Create a Torchviz visualization of WaveSourceMiniResNet."""
    print("🎨 Creating Torchviz visualization of WaveSourceMiniResNet...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print(f"🏗️  Creating model (grid_size={grid_size})")
    model = WaveSourceMiniResNet(grid_size=grid_size)
    model.eval()
    
    # Create sample input
    print(f"📊 Creating sample input: [1, 1, {grid_size}, {grid_size}]")
    sample_input = torch.randn(1, 1, grid_size, grid_size, requires_grad=True)
    
    # Forward pass
    print("🔄 Running forward pass...")
    output = model(sample_input)
    
    # Create the visualization
    print("🎯 Generating computational graph...")
    dot = make_dot(output, 
                   params=dict(model.named_parameters()),
                   show_attrs=True,
                   show_saved=True)
    
    # Customize the graph appearance
    dot.graph_attr.update({
        'rankdir': 'TB',  # Top to Bottom layout
        'dpi': '300',     # High resolution
        'size': '12,16',  # Size in inches
        'bgcolor': 'white'
    })
    
    # Node styling
    dot.node_attr.update({
        'style': 'filled',
        'fillcolor': 'lightblue',
        'fontname': 'Arial',
        'fontsize': '10'
    })
    
    # Edge styling  
    dot.edge_attr.update({
        'fontname': 'Arial',
        'fontsize': '8'
    })
    
    # Save the visualization
    output_file = output_dir / "WaveSourceMiniResNet_torchviz"
    print("💾 Saving visualization...")
    
    # Save as multiple formats
    formats = ['png', 'pdf', 'svg']
    saved_files = []
    
    for fmt in formats:
        try:
            file_path = dot.render(str(output_file), format=fmt, cleanup=True)
            saved_files.append(file_path)
            print(f"   ✅ Saved: {Path(file_path).name}")
        except Exception as e:
            print(f"   ❌ Failed to save {fmt}: {e}")
    
    print("\n🎉 Torchviz visualization complete!")
    print(f"📁 Files saved in: {output_dir}")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n📊 Model Summary:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Input Shape: [1, 1, {grid_size}, {grid_size}]")
    print(f"   Output Shape: {output.shape}")
    
    return saved_files


def create_simplified_diagram(grid_size=128, output_dir="scripts/model_export"):
    """Create a simplified layer-by-layer visualization."""
    print("\n🎨 Creating simplified layer diagram...")
    
    output_dir = Path(output_dir)
    model = WaveSourceMiniResNet(grid_size=grid_size)
    
    # Create sample input and trace through model
    sample_input = torch.randn(1, 1, grid_size, grid_size)
    
    # Hook to capture layer outputs
    layer_info = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if hasattr(output, 'shape'):
                layer_info.append({
                    'name': name,
                    'input_shape': input[0].shape if input else None,
                    'output_shape': output.shape,
                    'module_type': type(module).__name__
                })
        return hook
    
    # Register hooks on main components using CORRECT attribute names
    hooks = []
    hooks.append(model.wave_input_processor.register_forward_hook(hook_fn('Wave_Input_Processor')))
    hooks.append(model.wave_feature_stage1.register_forward_hook(hook_fn('Stage1_Wave_Features')))
    hooks.append(model.wave_pattern_stage2.register_forward_hook(hook_fn('Stage2_Wave_Patterns')))
    hooks.append(model.interference_stage3.register_forward_hook(hook_fn('Stage3_Interference')))
    hooks.append(model.source_localization_stage4.register_forward_hook(hook_fn('Stage4_Source_Localization')))
    hooks.append(model.global_wave_pool.register_forward_hook(hook_fn('Global_Wave_Pool')))
    
    # Forward pass to collect info
    with torch.no_grad():
        output = model(sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create simple text diagram
    diagram_file = output_dir / "WaveSourceMiniResNet_simplified.txt"
    with open(diagram_file, 'w') as f:
        f.write("WaveSourceMiniResNet Architecture Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input: {sample_input.shape}\n")
        f.write("↓\n")
        
        for info in layer_info:
            f.write(f"{info['name']} ({info['module_type']})\n")
            f.write(f"  Output: {info['output_shape']}\n")
            f.write("↓\n")
        
        f.write(f"Final Output: {output.shape}\n")
        f.write(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    print(f"✅ Simplified diagram saved: {diagram_file}")
    
    return str(diagram_file)


def main():
    """Main function to create visualizations."""
    print("🚀 Starting WaveSourceMiniResNet visualization...")
    
    # Create output directory
    output_dir = "scripts/model_export"
    
    try:
        # Create Torchviz diagram
        torchviz_files = create_torchviz_diagram(grid_size=128, output_dir=output_dir)
        
        # Create simplified diagram
        simple_file = create_simplified_diagram(grid_size=128, output_dir=output_dir)
        
        print("\n🎉 All visualizations complete!")
        print(f"📁 Check the files in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        print("💡 For PNG/PDF/SVG rendering, install system Graphviz:")
        print("   Windows: choco install graphviz")
        print("   Or download from: https://graphviz.org/download/")


if __name__ == "__main__":
    main() 