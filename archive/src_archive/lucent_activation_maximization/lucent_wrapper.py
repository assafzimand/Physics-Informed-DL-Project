#!/usr/bin/env python3
"""
Lucent-Compatible Model Wrapper for WaveSourceMiniResNet.

Wraps our wave source localization model to make it compatible with Lucent's
activation maximization framework, exposing target convolutional layers
with clean naming for filter visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from .layer_hooks import DetailedLayerHookManager


class LucentModelWrapper(nn.Module):
    """
    Wraps WaveSourceMiniResNet for Lucent activation maximization compatibility.
    
    Features:
    - Exposes target layers with clean names
    - Manages forward hooks for activation capture
    - Provides layer access for Lucent optimization
    - Maintains original model functionality
    """
    
    def __init__(self, model: nn.Module, hook_manager: DetailedLayerHookManager):
        """
        Initialize Lucent wrapper.
        
        Args:
            model: Trained WaveSourceMiniResNet model
            hook_manager: Configured layer hook manager
        """
        super(LucentModelWrapper, self).__init__()
        
        self.model = model.eval()  # Ensure model is in eval mode
        self.hook_manager = hook_manager
        
        # Map clean layer names to our internal layer numbers
        self.layer_mapping = {
            "stage1_conv2": 14,  # Early wave features (32 channels, 32x32)
            "stage2_conv2": 26,  # Complex patterns (64 channels, 16x16)  
            "stage3_conv2": 38,  # Interference patterns (128 channels, 8x8)
            "stage4_conv2": 50,  # Source localization (256 channels, 4x4)
        }
        
        # Reverse mapping for convenience
        self.layer_numbers = {v: k for k, v in self.layer_mapping.items()}
        
        # Store target layer modules for direct access
        self.target_layers = {}
        self._setup_layer_access()
        
    def _setup_layer_access(self):
        """Setup direct access to target layers for Lucent."""
        for layer_name, layer_num in self.layer_mapping.items():
            module = self.hook_manager.get_module_by_layer_number(layer_num)
            if module is not None:
                self.target_layers[layer_name] = module
                # Register the module as an attribute for Lucent access
                setattr(self, layer_name, module)
            else:
                print(f"⚠️  Warning: Could not access layer {layer_name} (#{layer_num})")
    
    def forward(self, x):
        """Forward pass through the model with channel conversion and normalization"""
        # Convert 3-channel RGB to 1-channel grayscale for wave model
        if x.shape[1] == 3:  # If RGB input
            # Simple addition: R + G + B (equal gradient flow to all channels)
            x = x[:, 0:1] + x[:, 1:2] + x[:, 2:3]
        
        # Apply training normalization (CRITICAL!)
        # These are the exact statistics from training
        wave_mean = 0.000460
        wave_std = 0.020842
        x = (x - wave_mean) / wave_std
        
        return self.model(x)
    
    def get_layer_module(self, layer_name: str) -> Optional[nn.Module]:
        """
        Get PyTorch module for a specific layer.
        
        Args:
            layer_name: Clean layer name (e.g., 'stage2_conv2')
            
        Returns:
            module: PyTorch module or None if not found
        """
        return self.target_layers.get(layer_name)
    
    def get_layer_activations(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get captured activations for a specific layer.
        
        Args:
            layer_name: Clean layer name (e.g., 'stage2_conv2')
            
        Returns:
            activations: Captured activation tensor or None
        """
        layer_num = self.layer_mapping.get(layer_name)
        if layer_num is not None:
            return self.hook_manager.get_activations(layer_num)
        return None
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a layer.
        
        Args:
            layer_name: Clean layer name (e.g., 'stage2_conv2')
            
        Returns:
            info: Dictionary with layer details
        """
        layer_num = self.layer_mapping.get(layer_name)
        module = self.get_layer_module(layer_name)
        
        if layer_num is None or module is None:
            return {"error": f"Layer {layer_name} not found"}
        
        info = {
            "layer_name": layer_name,
            "layer_number": layer_num,
            "module_type": module.__class__.__name__,
            "available": True
        }
        
        # Add convolution-specific info if applicable
        if isinstance(module, nn.Conv2d):
            info.update({
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding
            })
        
        return info
    
    def list_available_layers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available target layers."""
        return {name: self.get_layer_info(name) for name in self.layer_mapping.keys()}
    
    def register_target_hooks(self) -> Dict[str, bool]:
        """Register hooks on all target layers."""
        target_layer_nums = list(self.layer_mapping.values())
        return self.hook_manager.register_hooks(target_layer_nums)
    
    def cleanup_hooks(self):
        """Remove all registered hooks."""
        self.hook_manager.remove_all_hooks()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_hooks()


def create_lucent_wrapper(model: nn.Module) -> LucentModelWrapper:
    """
    Factory function to create a Lucent-compatible wrapper.
    
    Args:
        model: Trained WaveSourceMiniResNet model
        
    Returns:
        wrapper: Configured LucentModelWrapper
    """
    # Create hook manager
    hook_manager = DetailedLayerHookManager(model)
    
    # Create wrapper
    wrapper = LucentModelWrapper(model, hook_manager)
    
    # Register hooks on target layers
    hook_results = wrapper.register_target_hooks()
    
    # Verify setup
    success_count = sum(hook_results.values())
    total_count = len(hook_results)
    
    print(f"✅ Lucent wrapper created: {success_count}/{total_count} hooks registered")
    
    if success_count < total_count:
        print("⚠️  Some hooks failed to register:")
        for layer_path, success in hook_results.items():
            if not success:
                print(f"   ❌ {layer_path}")
    
    return wrapper 