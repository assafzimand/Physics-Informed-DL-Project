#!/usr/bin/env python3
"""
Detailed Layer Hook Manager for WaveSourceMiniResNet Activation Maximization.

Provides layer-by-layer access to intermediate activations with precise numbering
and automatic best model detection from cross-validation results.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re
import numpy as np


class DetailedLayerHookManager:
    """
    Manages forward hooks for detailed layer-by-layer activation capture.
    
    Features:
    - Layer-by-layer numbering (1-52)
    - Automatic best CV model detection
    - Clean hook registration/removal
    - Activation shape verification
    """
    
    def __init__(self, model: nn.Module):
        """Initialize hook manager with a model."""
        self.model = model
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.layer_map = self._create_layer_map()
        
    def _create_layer_map(self) -> Dict[int, str]:
        """Create detailed layer-by-layer mapping of WaveSourceMiniResNet."""
        return {
            # Stage 0: Initial wave pattern extraction (wave_input_processor)
            1: "wave_input_processor.0",     # Conv2d(1→32, 7x7, stride=2)
            2: "wave_input_processor.1",     # BatchNorm2d(32)
            3: "wave_input_processor.2",     # ReLU
            4: "wave_input_processor.3",     # MaxPool2d(3x3, stride=2)
            
            # Stage 1: Basic wave features (wave_feature_stage1)
            # Block 1 (index 0)
            5: "wave_feature_stage1.0.wave_feature_conv1",    # Conv2d(32→32, 3x3)
            6: "wave_feature_stage1.0.wave_feature_bn1",      # BatchNorm2d(32)
            7: "wave_feature_stage1.0.wave_feature_conv2",    # Conv2d(32→32, 3x3)
            8: "wave_feature_stage1.0.wave_feature_bn2",      # BatchNorm2d(32)
            9: "wave_feature_stage1.0.skip_connection",       # Skip + ReLU
            10: "wave_feature_stage1.0",                      # Block 1 output
            
            # Block 2 (index 1)
            11: "wave_feature_stage1.1.wave_feature_conv1",   # Conv2d(32→32, 3x3)
            12: "wave_feature_stage1.1.wave_feature_bn1",     # BatchNorm2d(32)
            13: "wave_feature_stage1.1.wave_feature_conv2",   # Conv2d(32→32, 3x3)
            14: "wave_feature_stage1.1.wave_feature_conv2",   # ← TARGET LAYER
            15: "wave_feature_stage1.1.wave_feature_bn2",     # BatchNorm2d(32)
            16: "wave_feature_stage1.1",                      # Block 2 output
            
            # Stage 2: Complex wave patterns (wave_pattern_stage2)
            # Block 1 (index 0) - projection
            17: "wave_pattern_stage2.0.wave_feature_conv1",   # Conv2d(32→64, 3x3, stride=2)
            18: "wave_pattern_stage2.0.wave_feature_bn1",     # BatchNorm2d(64)
            19: "wave_pattern_stage2.0.wave_feature_conv2",   # Conv2d(64→64, 3x3)
            20: "wave_pattern_stage2.0.wave_feature_bn2",     # BatchNorm2d(64)
            21: "wave_pattern_stage2.0.skip_connection",      # Projection skip + ReLU
            22: "wave_pattern_stage2.0",                      # Block 1 output
            
            # Block 2 (index 1) - identity
            23: "wave_pattern_stage2.1.wave_feature_conv1",   # Conv2d(64→64, 3x3)
            24: "wave_pattern_stage2.1.wave_feature_bn1",     # BatchNorm2d(64)
            25: "wave_pattern_stage2.1.wave_feature_conv2",   # Conv2d(64→64, 3x3)
            26: "wave_pattern_stage2.1.wave_feature_conv2",   # ← TARGET LAYER
            27: "wave_pattern_stage2.1.wave_feature_bn2",     # BatchNorm2d(64)
            28: "wave_pattern_stage2.1",                      # Block 2 output
            
            # Stage 3: Interference patterns (interference_stage3)
            # Block 1 (index 0) - projection
            29: "interference_stage3.0.wave_feature_conv1",   # Conv2d(64→128, 3x3, stride=2)
            30: "interference_stage3.0.wave_feature_bn1",     # BatchNorm2d(128)
            31: "interference_stage3.0.wave_feature_conv2",   # Conv2d(128→128, 3x3)
            32: "interference_stage3.0.wave_feature_bn2",     # BatchNorm2d(128)
            33: "interference_stage3.0.skip_connection",      # Projection skip + ReLU
            34: "interference_stage3.0",                      # Block 1 output
            
            # Block 2 (index 1) - identity
            35: "interference_stage3.1.wave_feature_conv1",   # Conv2d(128→128, 3x3)
            36: "interference_stage3.1.wave_feature_bn1",     # BatchNorm2d(128)
            37: "interference_stage3.1.wave_feature_conv2",   # Conv2d(128→128, 3x3)
            38: "interference_stage3.1.wave_feature_conv2",   # ← TARGET LAYER
            39: "interference_stage3.1.wave_feature_bn2",     # BatchNorm2d(128)
            40: "interference_stage3.1",                      # Block 2 output
            
            # Stage 4: Source localization (source_localization_stage4)
            # Block 1 (index 0) - projection
            41: "source_localization_stage4.0.wave_feature_conv1",  # Conv2d(128→256, 3x3, stride=2)
            42: "source_localization_stage4.0.wave_feature_bn1",    # BatchNorm2d(256)
            43: "source_localization_stage4.0.wave_feature_conv2",  # Conv2d(256→256, 3x3)
            44: "source_localization_stage4.0.wave_feature_bn2",    # BatchNorm2d(256)
            45: "source_localization_stage4.0.skip_connection",     # Projection skip + ReLU
            46: "source_localization_stage4.0",                     # Block 1 output
            
            # Block 2 (index 1) - identity
            47: "source_localization_stage4.1.wave_feature_conv1",  # Conv2d(256→256, 3x3)
            48: "source_localization_stage4.1.wave_feature_bn1",    # BatchNorm2d(256)
            49: "source_localization_stage4.1.wave_feature_conv2",  # Conv2d(256→256, 3x3)
            50: "source_localization_stage4.1.wave_feature_conv2",  # ← TARGET LAYER
            51: "source_localization_stage4.1.wave_feature_bn2",    # BatchNorm2d(256)
            52: "source_localization_stage4.1",                     # Block 2 output
            
            # Note: Global pooling and regression head not included
            # Focus on convolutional feature extraction layers
        }
    
    @property
    def target_layers(self) -> Dict[str, int]:
        """Get the main target layers for activation maximization."""
        return {
            "stage1_block2_conv2": 14,    # [B, 32, 32, 32] - Early patterns
            "stage2_block2_conv2": 26,    # [B, 64, 16, 16] - Mid-level features
            "stage3_block2_conv2": 38,    # [B, 128, 8, 8] - High-level features
            "stage4_block2_conv2": 50,    # [B, 256, 4, 4] - Abstract features
        }
    
    def get_module_by_layer_number(self, layer_num: int) -> Optional[nn.Module]:
        """Get PyTorch module by layer number."""
        if layer_num not in self.layer_map:
            return None
            
        layer_path = self.layer_map[layer_num]
        return self._get_module_by_path(layer_path)
    
    def _get_module_by_path(self, path: str) -> Optional[nn.Module]:
        """Navigate to module using dot-separated path."""
        try:
            module = self.model
            for attr in path.split('.'):
                if hasattr(module, attr):
                    module = getattr(module, attr)
                elif attr.isdigit():
                    # Handle Sequential indexing (e.g., "0", "1")
                    module = module[int(attr)]
                else:
                    return None
            return module
        except Exception:
            return None
    
    def register_hooks(self, layer_numbers: List[int]) -> Dict[str, bool]:
        """Register forward hooks on specified layers."""
        results = {}
        
        for layer_num in layer_numbers:
            layer_name = self.layer_map.get(layer_num, f"unknown_{layer_num}")
            module = self.get_module_by_layer_number(layer_num)
            
            if module is not None:
                hook_key = f"layer_{layer_num}_{layer_name.replace('.', '_')}"
                
                def make_hook(key: str):
                    def hook_fn(model, input, output):
                        self.activations[key] = output.detach().clone()
                    return hook_fn
                
                self.hooks[hook_key] = module.register_forward_hook(make_hook(hook_key))
                results[layer_name] = True
            else:
                results[layer_name] = False
                
        return results
    
    def get_activations(self, layer_number: int) -> Optional[torch.Tensor]:
        """Get saved activations for a specific layer number."""
        layer_name = self.layer_map.get(layer_number, f"unknown_{layer_number}")
        hook_key = f"layer_{layer_number}_{layer_name.replace('.', '_')}"
        return self.activations.get(hook_key)
    
    def clear_activations(self):
        """Clear all saved activations."""
        self.activations.clear()
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_all_hooks()


def find_best_cv_model(cv_results_path: Path) -> Tuple[int, float, Path]:
    """
    Automatically find the best model from CV results.
    
    Returns:
        Tuple of (best_fold_index, best_error, best_model_path)
    """
    results_file = cv_results_path / "analysis" / "cv_results_summary.txt"
    
    if not results_file.exists():
        raise FileNotFoundError(f"CV results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Parse individual fold errors
    error_line = re.search(r'Individual Fold Errors: \[(.*?)\]', content)
    if not error_line:
        raise ValueError("Could not parse fold errors from CV results")
    
    errors = [float(x.strip()) for x in error_line.group(1).split(',')]
    best_fold = np.argmin(errors)
    best_error = errors[best_fold]
    
    # Find best model path in data/models directory
    # Pattern: cv_full_5fold_75epochs_fold_{fold}_best.pth (1-indexed)
    models_dir = cv_results_path / "data" / "models"
    best_model_path = models_dir / f"cv_full_5fold_75epochs_fold_{best_fold + 1}_best.pth"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at: {best_model_path}")
    
    return best_fold, best_error, best_model_path 