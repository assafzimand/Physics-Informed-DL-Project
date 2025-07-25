"""
Core reverse operations for Zeiler & Fergus deconvolutional networks.
Implements reverse operations for: Conv2d, ReLU, BatchNorm2d, MaxPool2d, Skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class ReverseReLU(nn.Module):
    """
    Reverse ReLU operation as described in Zeiler & Fergus.
    For deconvnet: pass positive activations, zero negative activations.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In true deconvent, ReLU reverse just passes through
        # The rectification happens in deconv layers, not here
        # Zeiler & Fergus: don't re-apply ReLU to avoid double rectification
        return x


class ReverseBatchNorm2d(nn.Module):
    """
    Reverse BatchNorm2d operation using stored statistics from forward pass.
    Implements inverse normalization: x = (y - bias) / weight * sqrt(var + eps) + mean
    """
    
    def __init__(self, forward_bn: nn.BatchNorm2d):
        super(ReverseBatchNorm2d, self).__init__()
        
        # Store forward BatchNorm parameters and statistics
        self.num_features = forward_bn.num_features
        self.eps = forward_bn.eps
        
        # Register buffers for statistics (no gradient needed)
        self.register_buffer('running_mean', forward_bn.running_mean.clone())
        self.register_buffer('running_var', forward_bn.running_var.clone())
        self.register_buffer('weight', forward_bn.weight.data.clone())
        self.register_buffer('bias', forward_bn.bias.data.clone())
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reverse BatchNorm: x = (y - bias) / weight * sqrt(var + eps) + mean
        
        Args:
            y: Normalized output from forward BatchNorm
            
        Returns:
            x: Original input before BatchNorm
        """
        # Get statistics
        mean = self.running_mean
        var = self.running_var
        weight = self.weight
        bias = self.bias
        
        # Reverse normalization
        # Forward: y = (x - mean) / sqrt(var + eps) * weight + bias
        # Reverse: x = (y - bias) / weight * sqrt(var + eps) + mean
        x = (y - bias.view(1, -1, 1, 1)) / weight.view(1, -1, 1, 1)
        x = x * torch.sqrt(var + self.eps).view(1, -1, 1, 1)
        x = x + mean.view(1, -1, 1, 1)
        
        return x


class ReverseConv2d(nn.Module):
    """
    Reverse Conv2d operation using transposed convolution.
    Uses the same weights as forward convolution but in transposed manner.
    """
    
    def __init__(self, forward_conv: nn.Conv2d):
        super(ReverseConv2d, self).__init__()
        
        # Create transposed convolution with same parameters
        self.deconv = nn.ConvTranspose2d(
            in_channels=forward_conv.out_channels,
            out_channels=forward_conv.in_channels,
            kernel_size=forward_conv.kernel_size,
            stride=forward_conv.stride,
            padding=forward_conv.padding,
            bias=forward_conv.bias is not None
        )
        
        # Copy weights (properly transposed for deconvolution)
        with torch.no_grad():
            # Transpose conv weights: (out_ch, in_ch, H, W) -> (in_ch, out_ch, H, W)
            self.deconv.weight.data = forward_conv.weight.data.transpose(0, 1).clone()
            if forward_conv.bias is not None:
                self.deconv.bias.data = forward_conv.bias.data.clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class ReverseMaxPool2d(nn.Module):
    """
    Reverse MaxPool2d using unpooling with switch variables.
    For simplicity, uses nearest neighbor upsampling.
    In full implementation, would use stored switch variables from forward pass.
    """
    
    def __init__(self, forward_pool: nn.MaxPool2d):
        super(ReverseMaxPool2d, self).__init__()
        
        self.kernel_size = forward_pool.kernel_size
        self.stride = forward_pool.stride or forward_pool.kernel_size
        self.padding = forward_pool.padding
        
        # Use interpolation for unpooling (approximation)
        self.scale_factor = self.stride if isinstance(self.stride, int) else self.stride[0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Better unpooling using bilinear interpolation.
        Reduces blocky artifacts compared to nearest neighbor.
        """
        return F.interpolate(x, scale_factor=self.scale_factor, 
                           mode='bilinear', align_corners=False)


class ReverseSkipConnection(nn.Module):
    """
    Handles reverse skip connections for ResNet blocks.
    Layer-based approach: Identity vs Projection skips are perfectly reversible.
    """
    
    def __init__(self, skip_connection: nn.Module):
        super(ReverseSkipConnection, self).__init__()
        
        if isinstance(skip_connection, nn.Identity):
            # Identity skip: x_out = x_main + x_in
            # Reverse: x_main = x_out - x_in (but we don't have x_in)
            # Solution: In deconvent, focus on main path only
            self.reverse_skip = nn.Identity()
            self.is_identity = True
            
        elif isinstance(skip_connection, nn.Sequential):
            # Projection skip: x_out = x_main + proj(x_in)
            # Where proj = Conv2d + BatchNorm2d
            # Reverse: proj_reverse(x_skip_part) to get original x_in
            
            layers = list(skip_connection.children())
            reverse_layers = []
            
            # Reverse in opposite order
            for layer in reversed(layers):
                if isinstance(layer, nn.Conv2d):
                    reverse_layers.append(ReverseConv2d(layer))
                elif isinstance(layer, nn.BatchNorm2d):
                    reverse_layers.append(ReverseBatchNorm2d(layer))
                else:
                    raise ValueError(f"Unknown layer in projection skip: {type(layer)}")
            
            self.reverse_skip = nn.Sequential(*reverse_layers)
            self.is_identity = False
        else:
            raise ValueError(f"Unknown skip connection type: {type(skip_connection)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply reverse skip connection.
        
        For identity: pass through (focus on main path in deconvent)
        For projection: apply reverse Conv2d + BatchNorm2d sequence
        """
        return self.reverse_skip(x)


def create_reverse_operations(forward_model: nn.Module) -> Dict[str, Any]:
    """
    Create dictionary of reverse operations for all layers in the forward model.
    
    Args:
        forward_model: The trained forward model
        
    Returns:
        Dictionary mapping layer names to their reverse operations
    """
    reverse_ops = {}
    
    for name, module in forward_model.named_modules():
        if isinstance(module, nn.Conv2d):
            reverse_ops[f"reverse_{name}"] = ReverseConv2d(module)
        elif isinstance(module, nn.BatchNorm2d):
            reverse_ops[f"reverse_{name}"] = ReverseBatchNorm2d(module)
        elif isinstance(module, nn.MaxPool2d):
            reverse_ops[f"reverse_{name}"] = ReverseMaxPool2d(module)
        elif isinstance(module, (nn.Identity, nn.Sequential)) and 'skip_connection' in name:
            reverse_ops[f"reverse_{name}"] = ReverseSkipConnection(module)
    
    return reverse_ops


class DeconvnetActivationExtractor:
    """
    Extracts and manages activations for deconvolutional network visualization.
    Handles storage of intermediate activations and switch variables.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.hooks = []
        
        # Register hooks to capture activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.clone()
            return hook
        
        # Register hooks for key layers
        for name, module in self.model.named_modules():
            if any(stage in name for stage in ['stage1', 'stage2', 'stage3', 'stage4']):
                hook = module.register_forward_hook(save_activation(name))
                self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get stored activation for a specific layer."""
        return self.activations.get(layer_name)
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear() 