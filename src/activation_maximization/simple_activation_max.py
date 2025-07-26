#!/usr/bin/env python3
"""
Simple Activation Maximization for Single-Channel Wave Models

This implementation bypasses Lucent's RGB assumptions and works directly
with single-channel wave field inputs, ensuring proper gradient flow.
"""

import torch
from typing import Tuple, List


class SimpleActivationMaximizer:
    """
    Direct activation maximization for single-channel wave models.
    No RGB conversion, no external dependencies - just pure PyTorch.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the activation maximizer.
        
        Args:
            model: The wave source model
            device: Computing device
        """
        self.model = model.eval()
        self.device = device
        self.hooks = {}
        self.activations = {}
        
    def register_hook(self, layer_name: str, module):
        """Register forward hook on target layer"""
        def hook_fn(module, input, output):
            self.activations[layer_name] = output  # Keep gradients for activation maximization
            
        handle = module.register_forward_hook(hook_fn)
        self.hooks[layer_name] = handle
        
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        
    def optimize_filter(self, 
                       layer_name: str,
                       filter_idx: int,
                       iterations: int = 1024,
                       learning_rate: float = 0.01,
                       image_size: int = 128,
                       l2_weight: float = 1e-4,
                       total_variation_weight: float = 1e-2) -> Tuple[torch.Tensor, List[float]]:
        """
        Optimize input to maximize a specific filter's activation.
        
        Args:
            layer_name: Name of target layer
            filter_idx: Index of target filter
            iterations: Number of optimization steps
            learning_rate: Optimization learning rate
            image_size: Size of input image
            l2_weight: L2 regularization weight
            total_variation_weight: Total variation regularization weight
            
        Returns:
            Tuple of (optimized_input, loss_history)
        """
        print(f"🎯 Optimizing {layer_name} filter {filter_idx}")
        print(f"   Iterations: {iterations}, LR: {learning_rate}")
        
        # Initialize input tensor - single channel wave field
        # Shape: [1, 1, height, width] - matches our model's expectation!
        input_tensor = torch.randn(1, 1, image_size, image_size, 
                                 requires_grad=True, device=self.device)
        
        # Apply training normalization
        wave_mean = 0.000460
        wave_std = 0.020842
        
        # Optimizer for the input tensor
        optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)
        
        loss_history = []
        best_loss = float('inf')
        best_input = None
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Normalize input like during training
            normalized_input = (input_tensor - wave_mean) / wave_std
            
            # Forward pass
            _ = self.model(normalized_input)
            
            # Get target activation
            target_activation = self.activations[layer_name][0, filter_idx]
            
            # Primary loss: maximize filter activation (negative because we minimize)
            activation_loss = -target_activation.mean()
            
            # No regularization - focus purely on activation maximization
            # l2_loss = l2_weight * (input_tensor ** 2).mean()
            # grad_x = torch.gradient(input_tensor, dim=3)[0]  # Horizontal gradient ∂I/∂x
            # grad_y = torch.gradient(input_tensor, dim=2)[0]  # Vertical gradient ∂I/∂y
            # tv_loss = total_variation_weight * (grad_x**2 + grad_y**2).mean()
            
            # Total loss - just activation loss (no regularization)
            total_loss = activation_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track loss
            loss_history.append(total_loss.item())
            
            # Keep track of best result
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_input = input_tensor.detach().clone()
            
            if i % 50 == 0:
                activation_val = target_activation.mean().item()
                print(f"   Step {i:3d}: Loss={total_loss.item():.4f}, "
                      f"Activation={activation_val:.2f}, Best={best_loss:.4f}")
        
        print(f"✅ Optimization complete!")
        print(f"   Final activation: {target_activation.mean().item():.2f}")
        print(f"   Best loss achieved: {best_loss:.4f}")
        print(f"   Loss reduction: {loss_history[0] - best_loss:.4f}")
        
        # Return the best input found during optimization
        return best_input, loss_history
    
    def total_variation_loss(self, tensor):
        """Compute total variation loss for smoothness"""
        tv_h = torch.mean((tensor[:, :, 1:, :] - tensor[:, :, :-1, :]) ** 2)
        tv_w = torch.mean((tensor[:, :, :, 1:] - tensor[:, :, :, :-1]) ** 2)
        return tv_h + tv_w


def get_layer_by_path(model, layer_path: str):
    """Navigate to layer using dot notation path"""
    parts = layer_path.split('.')
    current = model
    
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    
    return current


def run_simple_activation_maximization(model, layer_path: str, filter_idx: int, 
                                     iterations: int = 512, device='cpu'):
    """
    Run activation maximization on single-channel wave model.
    
    Args:
        model: Wave source model
        layer_path: Dot notation path to target layer (e.g., 'wave_feature_stage2.1.wave_feature_conv2')
        filter_idx: Index of target filter
        iterations: Number of optimization iterations
        device: Computing device
        
    Returns:
        Tuple of (optimized_pattern, loss_history)
    """
    # Create maximizer
    maximizer = SimpleActivationMaximizer(model, device)
    
    try:
        # Get target layer
        target_layer = get_layer_by_path(model, layer_path)
        
        # Register hook
        layer_name = layer_path.split('.')[-1]  # Use last part as name
        maximizer.register_hook(layer_name, target_layer)
        
        # Run optimization
        result, losses = maximizer.optimize_filter(
            layer_name, filter_idx, iterations=iterations
        )
        
        return result, losses
        
    finally:
        # Always cleanup
        maximizer.cleanup_hooks() 