"""
Wave Deconvolutional Network Analyzer

Implementation based on Dosovitskiy & Brox approach for inverting CNN features.
Adapted for single-channel wave field data and regression models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from models.wave_source_resnet import WaveSourceMiniResNet


class WaveFeatureInverter(nn.Module):
    """
    Up-convolutional network for inverting wave features back to input space.
    Based on Dosovitskiy & Brox: learns to predict expected pre-image.
    """
    
    def __init__(self, feature_dim: int, feature_spatial_size: int):
        super(WaveFeatureInverter, self).__init__()
        
        self.feature_dim = feature_dim
        self.feature_spatial_size = feature_spatial_size
        
        # Calculate required upsampling to reach 128x128
        self.target_size = 128
        upsample_factor = self.target_size // feature_spatial_size
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Up-convolutional layers for reconstruction
        layers = []
        in_channels = 256
        current_size = feature_spatial_size
        
        while current_size < self.target_size:
            out_channels = max(in_channels // 2, 32)
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 
                                 kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
            current_size *= 2
        
        # Final layer to single channel
        layers.append(
            nn.ConvTranspose2d(in_channels, 1, kernel_size=3, padding=1)
        )
        
        self.upconv_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass: features -> reconstructed wave field"""
        x = self.feature_processor(x)
        x = self.upconv_layers(x)
        return x


class WaveDeconventAnalyzer:
    """
    Main analyzer for visualizing what wave patterns activate specific filters.
    
    Uses the Dosovitskiy & Brox approach: trains up-convolutional networks
    to reconstruct input wave fields from intermediate feature representations.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load trained WaveSourceMiniResNet
        self.wave_model = WaveSourceMiniResNet(grid_size=128)
        
        # Load checkpoint and extract model state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.wave_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.wave_model.load_state_dict(checkpoint)
            
        self.wave_model.to(self.device)
        self.wave_model.eval()
        
        # Dictionary to store feature inverters for different layers
        self.feature_inverters = {}
        
        # Hook storage for intermediate activations
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks on the stages we want to analyze
        self.wave_model.wave_feature_stage1.register_forward_hook(
            save_activation('stage_1'))
        self.wave_model.wave_pattern_stage2.register_forward_hook(
            save_activation('stage_2'))
        self.wave_model.interference_stage3.register_forward_hook(
            save_activation('stage_3'))
        self.wave_model.source_localization_stage4.register_forward_hook(
            save_activation('stage_4'))
    
    def extract_filter_features(self, wave_sample: torch.Tensor, 
                               stage: str, filter_indices: List[int]) -> Dict:
        """
        Extract features for specific filters from a wave sample.
        
        Args:
            wave_sample: Input wave field [1, 1, 128, 128]
            stage: Stage name (e.g., 'stage_2')
            filter_indices: List of filter indices to isolate
            
        Returns:
            Dictionary with isolated filter activations
        """
        with torch.no_grad():
            # Forward pass to get all activations
            _ = self.wave_model(wave_sample)
            
            # Get the full activation for this stage
            full_activation = self.activations[stage]  # [1, channels, H, W]
            
            filter_features = {}
            for filter_idx in filter_indices:
                # Create activation with only this filter active
                isolated_activation = torch.zeros_like(full_activation)
                isolated_activation[0, filter_idx] = full_activation[0, filter_idx]
                filter_features[filter_idx] = isolated_activation
            
            return filter_features
    
    def get_stage_info(self, stage: str) -> Tuple[int, int]:
        """Get feature dimensions for a given stage."""
        stage_dims = {
            'stage_1': (32, 32),   # 32 channels, 32x32 spatial
            'stage_2': (64, 16),   # 64 channels, 16x16 spatial  
            'stage_3': (128, 8),   # 128 channels, 8x8 spatial
            'stage_4': (256, 4)    # 256 channels, 4x4 spatial
        }
        return stage_dims[stage]
    
    def create_filter_visualization(self, sample_id: str, stage: str, 
                                  filter_indices: List[int],
                                  save_dir: str = "experiments/deconvent/plots"):
        """
        Create visualization showing what wave patterns activate specific filters.
        
        This is our Phase 2 implementation - start with a basic approach.
        For now, we'll show the isolated filter activations directly.
        Later we'll implement the full up-convolutional inversion.
        """
        print(f"🎨 Creating filter visualization for {sample_id}, {stage}")
        print(f"📊 Analyzing filters: {filter_indices}")
        
        # Load the sample (this is placeholder - we'll integrate with existing data)
        # For now, create a dummy wave field
        wave_sample = torch.randn(1, 1, 128, 128).to(self.device)
        
        # Extract filter features
        filter_features = self.extract_filter_features(wave_sample, stage, filter_indices)
        
        # Create visualization plot
        n_filters = len(filter_indices)
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, filter_idx in enumerate(filter_indices[:9]):  # Show up to 9 filters
            activation = filter_features[filter_idx][0, filter_idx].cpu().numpy()
            
            im = axes[i].imshow(activation, cmap='viridis', aspect='equal')
            axes[i].set_title(f'Filter #{filter_idx}', fontweight='bold')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(len(filter_indices), 9):
            axes[i].axis('off')
        
        plt.suptitle(f'Filter Analysis: {sample_id} - {stage}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        plot_file = save_path / f"{sample_id}_{stage}_filter_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Filter visualization saved: {plot_file}")
        return str(plot_file)


def main():
    """Demo function to test the basic framework."""
    print("🚀 Wave Deconvent Analyzer - Phase 1 Demo")
    
    # This is a placeholder - we'll integrate with actual model and data
    model_path = "experiments/cv_full/data/models/cv_full_5fold_75epochs_fold_2_best.pth"
    
    try:
        analyzer = WaveDeconventAnalyzer(model_path)
        
        # Demo: analyze 3 filters from stage 2
        sample_id = "demo_sample"
        stage = "stage_2" 
        filter_indices = [0, 15, 31]  # First, middle, last filter
        
        plot_path = analyzer.create_filter_visualization(
            sample_id, stage, filter_indices)
        
        print(f"🎉 Demo complete! Check: {plot_path}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 This is expected - we need to integrate with existing data pipeline")


if __name__ == "__main__":
    main() 