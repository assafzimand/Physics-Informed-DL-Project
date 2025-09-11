#!/usr/bin/env python3
"""
Inspect CV Model and Find Best Fold.

Automatically detects the best performing fold from CV results and loads the model
for detailed layer analysis.
"""

import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.wave_source_resnet import WaveSourceMiniResNet
from src.activation_maximization.layer_hooks import (
    DetailedLayerHookManager, 
    find_best_cv_model
)


def load_best_cv_model() -> tuple:
    """Load the best performing model from CV results."""
    
    print("🔍 Finding Best CV Model...")
    
    # Path to CV results
    cv_path = Path("experiments/cv_full")
    
    if not cv_path.exists():
        print(f"❌ CV results path not found: {cv_path}")
        return None, None, None
    
    try:
        # Automatically find best model
        best_fold, best_error, model_path = find_best_cv_model(cv_path)
        
        print("✅ Best model found:")
        print(f"   📁 Fold: {best_fold}")
        print(f"   📊 Error: {best_error:.4f} px")
        print(f"   📄 Path: {model_path}")
        
        # Load model
        print("\n📥 Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = WaveSourceMiniResNet()
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded state dict from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded direct state dict")
        
        model.to(device)
        model.eval()
        
        return model, best_fold, best_error
        
    except Exception as e:
        print(f"❌ Error loading best model: {e}")
        return None, None, None


def inspect_model_structure(model: torch.nn.Module):
    """Inspect and print detailed model structure."""
    
    print("\n🏗️  Model Structure Inspection:")
    print("=" * 60)
    
    # Print overall structure
    print(f"📋 Model: {model.__class__.__name__}")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n🔍 Layer Structure:")
    print("-" * 40)
    
    # Inspect each stage
    for stage_name in ['stage0', 'stage1', 'stage2', 'stage3', 'stage4']:
        if hasattr(model, stage_name):
            stage = getattr(model, stage_name)
            print(f"\n📦 {stage_name.upper()}:")
            
            if hasattr(stage, '__len__'):  # Sequential container
                for i, block in enumerate(stage):
                    print(f"   Block {i+1}: {block.__class__.__name__}")
                    if hasattr(block, 'conv1'):
                        conv1 = block.conv1
                        print(f"      Conv1: {conv1.in_channels}→{conv1.out_channels}, {conv1.kernel_size}")
                    if hasattr(block, 'conv2'):
                        conv2 = block.conv2
                        print(f"      Conv2: {conv2.in_channels}→{conv2.out_channels}, {conv2.kernel_size}")
            else:  # Single layer (stage0)
                print(f"   Type: {stage.__class__.__name__}")
                if hasattr(stage, 'in_channels'):
                    print(f"   Channels: {stage.in_channels}→{stage.out_channels}")


def test_layer_hooks(model: torch.nn.Module):
    """Test layer hook functionality."""
    
    print("\n🎯 Testing Layer Hooks:")
    print("=" * 60)
    
    # Create hook manager
    hook_manager = DetailedLayerHookManager(model)
    
    # Test target layers
    target_layers = list(hook_manager.target_layers.values())
    print(f"🎯 Target layers: {target_layers}")
    
    # Register hooks
    print("\n📌 Registering hooks...")
    hook_results = hook_manager.register_hooks(target_layers)
    
    for layer_name, success in hook_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {layer_name}")
    
    # Create test input
    print("\n🧪 Testing with dummy input...")
    device = next(model.parameters()).device
    test_input = torch.randn(1, 1, 128, 128).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    # Check activations
    print("\n📊 Activation Shapes:")
    for layer_name, layer_num in hook_manager.target_layers.items():
        activations = hook_manager.get_activations(layer_num)
        if activations is not None:
            print(f"   ✅ Layer {layer_num} ({layer_name}): {activations.shape}")
        else:
            print(f"   ❌ Layer {layer_num} ({layer_name}): No activations captured")
    
    # Cleanup
    hook_manager.remove_all_hooks()
    print("\n🧹 Hooks cleaned up")
    
    return True


def main():
    """Main inspection routine."""
    
    print("🚀 CV Model Inspection & Hook Testing")
    print("=" * 60)
    
    # Load best model
    model, best_fold, best_error = load_best_cv_model()
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Inspect structure
    inspect_model_structure(model)
    
    # Test hooks
    success = test_layer_hooks(model)
    
    if success:
        print("\n🎉 All tests PASSED!")
        print("📋 Summary:")
        print(f"   • Best fold: {best_fold} (error: {best_error:.4f}px)")
        print("   • Model loaded and hooks working")
        print("   • Ready for Phase 3: Model Wrapper")
    else:
        print("\n⚠️  Some tests FAILED. Check implementation.")


if __name__ == "__main__":
    main() 