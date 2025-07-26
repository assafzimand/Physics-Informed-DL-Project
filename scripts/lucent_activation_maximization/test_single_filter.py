#!/usr/bin/env python3
"""
Single Filter Activation Maximization Test.

Tests our complete activation maximization pipeline on a single filter
to verify everything works before running larger experiments.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.wave_source_resnet import WaveSourceMiniResNet
from src.activation_maximization.layer_hooks import (
    DetailedLayerHookManager, 
    find_best_cv_model
)
from src.activation_maximization.lucent_wrapper import create_lucent_wrapper
from src.activation_maximization.lucent_integration import (
    setup_wave_optimization,
    run_activation_maximization,
    visualize_wave_pattern,
    analyze_wave_pattern,
    verify_lucent
)
from load_analysis_data import load_analysis_dataset, get_diverse_samples


def load_best_model():
    """Load the best CV model for testing."""
    
    print("🔍 Loading Best CV Model...")
    
    cv_path = Path("experiments/cv_full")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        best_fold, best_error, model_path = find_best_cv_model(cv_path)
        print(f"✅ Found best model: Fold {best_fold} (error: {best_error:.4f}px)")
        print(f"   📄 Path: {model_path}")
        
        # Load model
        model = WaveSourceMiniResNet()
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded: {model.get_num_parameters():,} parameters")
        return model, device
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def test_single_filter_optimization():
    """Test activation maximization on a single filter."""
    
    print("🧪 Testing Single Filter Activation Maximization")
    print("=" * 60)
    
    # Verify dependencies
    verify_lucent()
    
    # Load model
    model, device = load_best_model()
    if model is None:
        return False
    
    # Create Lucent wrapper
    print("\n🔧 Creating Lucent Wrapper...")
    wrapper = create_lucent_wrapper(model)
    wrapper.to(device)
    
    # Test configuration
    target_layer = "stage1_conv2"  # Earlier layer for simpler, clearer features
    target_filter = 10
    iterations = 512  # Increased for better convergence and cleaner patterns
    learning_rate = 0.05
    
    print(f"\n🎯 Test Configuration:")
    print(f"   • Layer: {target_layer}")
    print(f"   • Filter: {target_filter}")
    print(f"   • Iterations: {iterations}")
    print(f"   • Learning rate: {learning_rate}")
    
    # Check layer info
    layer_info = wrapper.get_layer_info(target_layer)
    print(f"\n📋 Layer Information:")
    for key, value in layer_info.items():
        print(f"   • {key}: {value}")
    
    try:
        # Setup optimization
        print(f"\n⚙️  Setting up optimization...")
        objective, param_f = setup_wave_optimization(
            wrapper, 
            target_layer, 
            target_filter
        )
        
        # Run optimization
        optimized_pattern, loss_history = run_activation_maximization(
            wrapper,  # model_wrapper first
            objective,
            param_f,
            iterations=iterations,
            learning_rate=learning_rate,
            show_progress=True
        )
        
        print(f"✅ Optimization completed!")
        print(f"   Generated pattern shape: {optimized_pattern.shape}")
        print(f"   Loss history: {len(loss_history)} values")
        
        # Plot loss curve
        print(f"\n📈 Plotting loss curve...")
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2)
        plt.title(f"Activation Maximization Loss Curve\n{target_layer} Filter {target_filter}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (Negative Activation)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save loss plot
        output_dir = Path("experiments/activation_maximization")
        output_dir.mkdir(exist_ok=True)
        loss_plot_path = output_dir / f"loss_curve_{target_layer}_filter_{target_filter}.png"
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Loss curve saved: {loss_plot_path}")
        print(f"   Initial loss: {loss_history[0]:.4f}")
        print(f"   Final loss: {loss_history[-1]:.4f}")
        print(f"   Loss reduction: {loss_history[0] - loss_history[-1]:.4f}")
        
        # Visualize and analyze pattern
        print(f"\n📊 Pattern Analysis:")
        analysis = analyze_wave_pattern(optimized_pattern)
        for key, value in analysis.items():
            print(f"   • {key}: {value:.4f}")
        
        # Visualize result
        output_dir = Path("experiments/activation_maximization")
        output_dir.mkdir(exist_ok=True)
        
        save_path = output_dir / f"single_filter_test_{target_layer}_filter_{target_filter}.png"
        
        fig = visualize_wave_pattern(
            optimized_pattern,
            target_layer,
            target_filter,
            save_path=save_path,
            show_plot=False  # Don't show in script
        )
        
        # Test comparison with real data
        print(f"\n🔍 Loading real wave data for comparison...")
        try:
            wave_fields, source_positions = load_analysis_dataset()
            diverse_fields, diverse_pos, indices = get_diverse_samples(
                wave_fields, source_positions, num_samples=3
            )
            
            print(f"✅ Loaded {len(diverse_fields)} diverse samples for comparison")
            
            # Compare with first sample
            real_sample = diverse_fields[0]
            
            from src.activation_maximization.lucent_integration import compare_with_real_wave
            comparison = compare_with_real_wave(optimized_pattern, real_sample)
            
            print(f"\n🔬 Comparison with Real Wave Field:")
            for key, value in comparison.items():
                print(f"   • {key}: {value:.4f}")
            
        except Exception as e:
            print(f"⚠️  Could not load comparison data: {e}")
        
        print(f"\n🎉 Single Filter Test COMPLETED!")
        print(f"📁 Results saved to: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper_functionality():
    """Test the wrapper functionality separately."""
    
    print("\n🔧 Testing Wrapper Functionality...")
    
    model, device = load_best_model()
    if model is None:
        return False
    
    wrapper = create_lucent_wrapper(model)
    wrapper.to(device)
    
    # Test layer access
    print(f"\n📋 Available Layers:")
    layers_info = wrapper.list_available_layers()
    for name, info in layers_info.items():
        print(f"   • {name}: {info}")
    
    # Test forward pass with hook capture
    print(f"\n🧪 Testing Forward Pass with Hooks...")
    test_input = torch.randn(1, 1, 128, 128).to(device)
    
    with torch.no_grad():
        output = wrapper(test_input)
    
    print(f"   • Model output shape: {output.shape}")
    
    # Check captured activations
    for layer_name in wrapper.layer_mapping.keys():
        activations = wrapper.get_layer_activations(layer_name)
        if activations is not None:
            print(f"   • {layer_name} activations: {activations.shape}")
        else:
            print(f"   • {layer_name}: No activations captured")
    
    wrapper.cleanup_hooks()
    return True


def main():
    """Main test routine."""
    
    print("🚀 Single Filter Activation Maximization Test")
    print("=" * 60)
    
    # Test wrapper first
    wrapper_success = test_wrapper_functionality()
    
    if wrapper_success:
        # Test full optimization
        optimization_success = test_single_filter_optimization()
        
        if optimization_success:
            print(f"\n✅ ALL TESTS PASSED!")
            print(f"🎯 Ready for Phase 4: Multi-Filter Grid Generation")
        else:
            print(f"\n⚠️  Optimization test failed")
    else:
        print(f"\n❌ Wrapper test failed")


if __name__ == "__main__":
    main() 