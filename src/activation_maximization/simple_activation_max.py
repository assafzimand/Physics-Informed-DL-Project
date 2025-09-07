#!/usr/bin/env python3
"""
Simple Activation Maximization for Single-Channel Wave Models

This implementation bypasses Lucent's RGB assumptions and works directly
with single-channel wave field inputs, ensuring proper gradient flow.

Enhanced with comprehensive monitoring, plotting, and debugging capabilities.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any


class SimpleActivationMaximizer:
    """
    Direct activation maximization for single-channel wave models.
    No RGB conversion, no external dependencies - just pure PyTorch.
    """
    
    def __init__(self, model, device='cpu', model_path: Optional[str] = None, dataset_name: Optional[str] = None):
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
        # Normalization stats resolved from training dataset (T250/T500)
        self.dataset_name = self._determine_dataset_name(model_path, dataset_name)
        self.wave_mean, self.wave_std = self._load_training_normalization_stats(self.dataset_name)
        
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
    
    def _determine_dataset_name(self, model_path: Optional[str], dataset_name: Optional[str]) -> str:
        """Infer dataset name (T250/T500) from explicit arg or model checkpoint path/config."""
        # Priority 1: explicit dataset_name
        if dataset_name is not None:
            return dataset_name.upper()
        # Priority 2: try reading checkpoint config
        if model_path is not None:
            try:
                import torch as _torch
                ckpt = _torch.load(model_path, map_location='cpu')
                if isinstance(ckpt, dict):
                    cfg = ckpt.get('config') or {}
                    ds_name = (cfg.get('dataset_name') or '').upper()
                    ds_path = (cfg.get('dataset_path') or '').upper()
                    if ds_name in { 'T250', 'T500' }:
                        return ds_name
                    if 'T250' in ds_path:
                        return 'T250'
                    if 'T500' in ds_path:
                        return 'T500'
            except Exception:
                pass
            # Priority 3: heuristics from model_path string
            path_lower = str(model_path).lower()
            if 't250' in path_lower:
                return 'T250'
            if 't500' in path_lower:
                return 'T500'
        # Fallback: default to T500 but warn
        print("⚠️ Could not determine dataset from model; defaulting to T500 normalization")
        return 'T500'

    def _load_training_normalization_stats(self, dataset_name: str) -> Tuple[float, float]:
        """Load wave_mean/std from the corresponding TRAINING HDF5 file attributes.

        Uses:
          - data/wave_dataset_T250.h5 for T250
          - data/wave_dataset_T500.h5 for T500
        Falls back to historical constants if files are unavailable.
        """
        import h5py as _h5
        from pathlib import Path as _Path
        ds = dataset_name.upper()
        if ds == 'T250':
            h5_path = _Path('data/wave_dataset_T250.h5')
        else:
            h5_path = _Path('data/wave_dataset_T500.h5')
        try:
            if h5_path.exists():
                with _h5.File(str(h5_path), 'r') as f:
                    mean = float(f.attrs.get('wave_mean'))
                    std = float(f.attrs.get('wave_std'))
                    print(f"📊 Using training normalization ({ds}): mean={mean:.6f}, std={std:.6f}")
                    return mean, std
        except Exception:
            pass
        # Historical defaults (kept for safety)
        default_mean = 0.000460
        default_std = 0.020842
        print(f"⚠️ Using fallback normalization ({ds}): mean={default_mean:.6f}, std={default_std:.6f}")
        return default_mean, default_std
        
    def load_real_wave_samples(self, dataset_path="data/wave_dataset_analysis_20samples.h5"):
        """Load real wave samples for initialization"""
        try:
            with h5py.File(dataset_path, 'r') as f:
                if 'wave_fields' in f and 'coordinates' in f:
                    wave_fields = f['wave_fields'][:]
                    coordinates = f['coordinates'][:]
                    
                    # Remove channel dimension if present
                    if len(wave_fields.shape) == 4 and wave_fields.shape[1] == 1:
                        wave_fields = wave_fields[:, 0]
                    
                    return wave_fields, coordinates
                else:
                    return None, None
        except:
            return None, None

    def optimize_filter(self, 
                       layer_name: str,
                       filter_idx: int,
                       iterations: int = 512,
                       learning_rate: float = 0.01,
                       image_size: int = 128,
                       use_real_data_init: bool = True,
                       save_intermediate: bool = True,
                       save_every: int = 100,
                       save_dir: Optional[str] = None,
                       init_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Comprehensive activation maximization with detailed monitoring and plotting.
        
        Args:
            layer_name: Name of target layer
            filter_idx: Index of target filter
            iterations: Number of optimization steps
            learning_rate: Optimization learning rate
            image_size: Size of input image
            use_real_data_init: Initialize with real wave data instead of random
            skip_normalization: Skip normalization step during optimization
            save_intermediate: Save intermediate patterns during optimization
            save_every: Save intermediate patterns every N iterations
            save_dir: Directory to save plots (if None, uses default)
            init_tensor: Specific tensor to use for initialization (overrides use_real_data_init)
            
        Returns:
            Dictionary with comprehensive results including patterns, monitoring data, plots
        """
        print(f"\n🎯 COMPREHENSIVE ACTIVATION MAXIMIZATION")
        print("=" * 70)
        print(f"🔍 Target: {layer_name} filter {filter_idx}")
        print(f"📊 Config: {iterations} iterations, LR={learning_rate}")
        print(f"🌊 Real data init: {'✅' if use_real_data_init else '❌'}")
        print(f"📈 Normalization: ALWAYS ON ({self.dataset_name})")
        
        # Load real data if requested
        wave_samples = None
        if use_real_data_init and init_tensor is None:
            wave_samples, _ = self.load_real_wave_samples()
            if wave_samples is not None:
                print(f"✅ Loaded {len(wave_samples)} real wave samples")
            else:
                print("❌ Failed to load real data, using random initialization")
        
        # Initialize monitoring
        monitoring_data = {
            'iteration': [],
            'loss': [],
            'activation': [],
            'input_mean': [],
            'input_std': [],
            'input_min': [],
            'input_max': [],
            'grad_magnitude': [],
            'grad_mean': [],
            'grad_std': [],
            'intermediate_patterns': []
        }
        
        # Initialize input tensor
        if init_tensor is not None:
            # Use provided initialization tensor
            print(f"🎯 Using provided initialization tensor")
            input_tensor = init_tensor.clone().detach().to(self.device)
            input_tensor.requires_grad_(True)
            initial_pattern = input_tensor.clone().detach()
            print(f"📊 Init stats: mean={input_tensor.mean():.6f}, std={input_tensor.std():.6f}")
        elif use_real_data_init and wave_samples is not None:
            # Use random real sample
            sample_idx = random.randint(0, len(wave_samples) - 1)
            initial_sample = wave_samples[sample_idx]
            print(f"🌊 Using real sample {sample_idx} as initialization")
            print(f"📊 Sample stats: mean={initial_sample.mean():.6f}, std={initial_sample.std():.6f}")
            
            input_tensor = torch.from_numpy(initial_sample).float()
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            input_tensor.requires_grad_(True)
            initial_pattern = input_tensor.clone().detach()
        else:
            # Random initialization
            print("🎲 Using random initialization")
            input_tensor = torch.randn(1, 1, image_size, image_size, 
                                     requires_grad=True, device=self.device)
            initial_pattern = input_tensor.clone().detach()
        
        # Use training normalization constants resolved at init
        wave_mean = self.wave_mean
        wave_std = self.wave_std
        
        # Optimizer
        optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)
        
        # Tracking variables
        loss_history = []
        best_loss = float('inf')
        best_input = None
        
        print(f"\n🚀 Starting optimization...")
        
        # Main optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass with training normalization (always applied)
            model_input = (input_tensor - wave_mean) / wave_std
            print_norm_status = "NORMALIZED" if i == 0 else ""
                
            if print_norm_status:
                print(f"📊 Input type: {print_norm_status}")
            
            # Forward pass
            _ = self.model(model_input)
            
            # Get target activation
            target_activation = self.activations[layer_name][:, filter_idx]
            
            # Loss: maximize filter activation
            activation_loss = -target_activation.mean()
            total_loss = activation_loss
            
            # Backward pass
            total_loss.backward()
            
            # Collect comprehensive monitoring data
            with torch.no_grad():
                grad_mag = input_tensor.grad.norm().item() if input_tensor.grad is not None else 0.0
                grad_mean = input_tensor.grad.mean().item() if input_tensor.grad is not None else 0.0
                grad_std = input_tensor.grad.std().item() if input_tensor.grad is not None else 0.0
                
                monitoring_data['iteration'].append(i)
                monitoring_data['loss'].append(total_loss.item())
                monitoring_data['activation'].append(-activation_loss.item())
                monitoring_data['input_mean'].append(input_tensor.mean().item())
                monitoring_data['input_std'].append(input_tensor.std().item())
                monitoring_data['input_min'].append(input_tensor.min().item())
                monitoring_data['input_max'].append(input_tensor.max().item())
                monitoring_data['grad_magnitude'].append(grad_mag)
                monitoring_data['grad_mean'].append(grad_mean)
                monitoring_data['grad_std'].append(grad_std)
                
                # Save intermediate patterns
                if save_intermediate and i % save_every == 0:
                    pattern_copy = input_tensor.clone().detach()
                    monitoring_data['intermediate_patterns'].append({
                        'iteration': i,
                        'pattern': pattern_copy,
                        'activation': -activation_loss.item()
                    })
            
            # Update best result
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_input = input_tensor.clone().detach()
            
            # Optimize
            optimizer.step()
            
            # Progress reporting
            if i % 100 == 0 or i == iterations - 1:
                print(f"   Step {i:4d}: Loss={total_loss.item():.4f}, "
                      f"Activation={-activation_loss.item():.2f}, "
                      f"GradMag={grad_mag:.6f}, "
                      f"InputStd={input_tensor.std().item():.4f}")
        
        print(f"\n✅ Optimization complete!")
        final_activation = monitoring_data['activation'][-1]
        loss_reduction = monitoring_data['loss'][0] - monitoring_data['loss'][-1]
        avg_grad_mag = np.mean(monitoring_data['grad_magnitude'])
        grad_variation = np.std(monitoring_data['grad_magnitude'])
        
        print(f"📊 Final Results:")
        print(f"   Final Activation: {final_activation:.2f}")
        print(f"   Loss Reduction: {loss_reduction:.4f}")
        print(f"   Average Gradient Magnitude: {avg_grad_mag:.6f}")
        print(f"   Gradient Variation (std): {grad_variation:.6f}")
        if grad_variation > 1e-6:
            print(f"   ✅ GRADIENTS ARE VARYING! (std > 1e-6)")
        else:
            print(f"   ❌ Gradients are constant (std ≤ 1e-6)")
        
        # Create comprehensive results dictionary
        results = {
            'best_pattern': best_input,
            'initial_pattern': initial_pattern,
            'monitoring_data': monitoring_data,
            'config': {
                'layer_name': layer_name,
                'filter_idx': filter_idx,
                'iterations': iterations,
                'learning_rate': learning_rate,
                'use_real_data_init': use_real_data_init,
                'dataset_name': self.dataset_name,
                'final_activation': final_activation,
                'loss_reduction': loss_reduction,
                'grad_variation': grad_variation
            }
        }
        
        # ALWAYS create comprehensive plots and analysis
        plot_path = self.create_comprehensive_plots(results, save_dir)
        results['plot_path'] = plot_path
        
        return results

    def create_comprehensive_plots(self, results: Dict[str, Any], save_dir: Optional[str] = None):
        """Create comprehensive visualization plots"""
        if save_dir is None:
            save_dir = Path("experiments/activation_maximization/comprehensive")
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        monitoring_data = results['monitoring_data']
        config = results['config']
        
        # Create main figure with multiple panels - 3 rows for 500 iterations
        fig = plt.figure(figsize=(20, 15))
        
        # Use the same training normalization used during optimization
        wave_mean = self.wave_mean
        wave_std = self.wave_std

        # Panel 1: Initial vs Final patterns (in model input space)
        ax1 = plt.subplot(3, 4, 1)
        initial_raw = results['initial_pattern'][0, 0].cpu().numpy()
        # Show normalized version that model actually sees
        initial_normalized = (initial_raw - wave_mean) / wave_std
        im1 = ax1.imshow(initial_normalized, cmap='RdBu_r', interpolation='nearest')
        ax1.set_title('Initial Pattern\n(Model Input Space)', fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = plt.subplot(3, 4, 2)
        final_raw = results['best_pattern'][0, 0].cpu().numpy()
        # Show normalized version that model actually sees
        final_normalized = (final_raw - wave_mean) / wave_std
        im2 = ax2.imshow(final_normalized, cmap='RdBu_r', interpolation='nearest')
        ax2.set_title('Final Optimized Pattern\n(Model Input Space)', fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Panel 2: Evolution timeline - 6 steps for 500 iterations (every 100)
        evolution_data = monitoring_data['intermediate_patterns']
        num_steps = min(6, len(evolution_data))
        
        for i in range(num_steps):
            ax = plt.subplot(3, 6, 7 + i)
            if i < len(evolution_data):
                step_data = evolution_data[i]
                pattern_raw = step_data['pattern'][0, 0].cpu().numpy()
                # Show normalized version that model actually sees
                pattern_normalized = (pattern_raw - wave_mean) / wave_std
                
                im = ax.imshow(pattern_normalized, cmap='RdBu_r', interpolation='nearest')
                ax.set_title(f"Step {step_data['iteration']}\nAct: {step_data['activation']:.1f}", 
                           fontsize=9, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Panel 3: Loss and activation curves
        ax3 = plt.subplot(3, 4, 9)
        iterations = monitoring_data['iteration']
        losses = monitoring_data['loss']
        activations = monitoring_data['activation']
        
        ax3.plot(iterations, losses, 'b-', label='Loss', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.grid(True, alpha=0.3)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(iterations, activations, 'r-', label='Activation', linewidth=2)
        ax3_twin.set_ylabel('Activation', color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        ax3.set_title('Optimization Progress', fontweight='bold')
        
        # Panel 4: Gradient analysis
        ax4 = plt.subplot(3, 4, 10)
        grad_mags = monitoring_data['grad_magnitude']
        ax4.plot(iterations, grad_mags, 'purple', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Gradient Magnitude')
        ax4.set_title('Gradient Evolution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Input statistics
        ax5 = plt.subplot(3, 4, 11)
        input_means = monitoring_data['input_mean']
        input_stds = monitoring_data['input_std']
        ax5.plot(iterations, input_means, 'g-', label='Mean', linewidth=2)
        ax5.plot(iterations, input_stds, 'm-', label='Std', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Input Statistics')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Input Evolution', fontweight='bold')
        
        # Panel 6: Summary statistics
        ax6 = plt.subplot(3, 4, 12)
        ax6.axis('off')
        
        summary_stats = {
            'Final Activation': f"{config['final_activation']:.2f}",
            'Loss Reduction': f"{config['loss_reduction']:.4f}",
            'Gradient Variation': f"{config['grad_variation']:.6f}",
            'Real Data Init': '✅' if config['use_real_data_init'] else '❌',
            'Dataset': self.dataset_name,
            'Gradients Varying': '✅' if config['grad_variation'] > 1e-6 else '❌'
        }
        
        stats_text = '\n'.join([f"{k}: {v}" for k, v in summary_stats.items()])
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.set_title('Summary', fontweight='bold')
        
        # Main title
        norm_status = "WITH NORMALIZATION"
        init_status = "REAL DATA" if config['use_real_data_init'] else "RANDOM"
        
        fig.suptitle(f'Comprehensive Activation Maximization\n'
                    f'{config["layer_name"]} Filter {config["filter_idx"]} | '
                    f'{norm_status} | {init_status} INIT', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"comprehensive_layer_{config['layer_name']}_filter_{config['filter_idx']:02d}"
        if config['use_real_data_init']:
            filename += "_REAL_INIT"
        filename += ".png"
        
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Comprehensive plot saved: {save_path}")
        return save_path
    
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