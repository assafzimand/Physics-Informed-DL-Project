"""
Wave Source Localization Mini-ResNet

A custom ResNet-based CNN for predicting wave source coordinates from wave interference patterns.
Designed with interpretability in mind using meaningful layer names.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveResidualBlock(nn.Module):
    """
    Residual block optimized for wave pattern analysis.
    Uses batch normalization and skip connections for stable gradient flow.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(WaveResidualBlock, self).__init__()
        
        # Main convolution path
        self.wave_feature_conv1 = nn.Conv2d(in_channels, out_channels, 
                                           kernel_size=3, stride=stride, 
                                           padding=1, bias=False)
        self.wave_feature_bn1 = nn.BatchNorm2d(out_channels)
        
        self.wave_feature_conv2 = nn.Conv2d(out_channels, out_channels, 
                                           kernel_size=3, stride=1, 
                                           padding=1, bias=False)
        self.wave_feature_bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (projection if needed)
        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Store input for skip connection
        residual = x
        
        # Main path: extract wave features
        out = F.relu(self.wave_feature_bn1(self.wave_feature_conv1(x)))
        out = self.wave_feature_bn2(self.wave_feature_conv2(out))
        
        # Add skip connection
        out += self.skip_connection(residual)
        out = F.relu(out)
        
        return out


class WaveSourceMiniResNet(nn.Module):
    """
    Mini-ResNet for wave source localization.
    
    Architecture:
    - Initial feature extraction from wave patterns
    - 4 stages of residual blocks with increasing complexity
    - Global pooling and coordinate regression head
    - Outputs (x, y) coordinates in range [0, 127]
    """
    
    def __init__(self, input_channels=1, grid_size=128):
        super(WaveSourceMiniResNet, self).__init__()
        
        self.grid_size = grid_size
        
        # Stage 0: Initial wave pattern extraction
        self.wave_input_processor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Output: 32 x 32 x 32
        
        # Stage 1: Basic wave feature detection (32 channels)
        self.wave_feature_stage1 = nn.Sequential(
            WaveResidualBlock(32, 32, stride=1),
            WaveResidualBlock(32, 32, stride=1)
        )
        # Output: 32 x 32 x 32
        
        # Stage 2: Complex wave pattern analysis (64 channels)
        self.wave_pattern_stage2 = nn.Sequential(
            WaveResidualBlock(32, 64, stride=2),
            WaveResidualBlock(64, 64, stride=1)
        )
        # Output: 64 x 16 x 16
        
        # Stage 3: Interference pattern understanding (128 channels)
        self.interference_stage3 = nn.Sequential(
            WaveResidualBlock(64, 128, stride=2),
            WaveResidualBlock(128, 128, stride=1)
        )
        # Output: 128 x 8 x 8
        
        # Stage 4: Source localization features (256 channels)
        self.source_localization_stage4 = nn.Sequential(
            WaveResidualBlock(128, 256, stride=2),
            WaveResidualBlock(256, 256, stride=1)
        )
        # Output: 256 x 4 x 4
        
        # Global feature aggregation
        self.global_wave_pool = nn.AdaptiveAvgPool2d(1)
        
        # Coordinate regression head
        self.coordinate_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # (x, y) coordinates
        )
        
        # Store intermediate activations for interpretability
        self.activations = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks for interpretability analysis."""
        
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks on key layers for analysis
        self.wave_feature_stage1.register_forward_hook(save_activation('basic_wave_features'))
        self.wave_pattern_stage2.register_forward_hook(save_activation('complex_wave_patterns'))
        self.interference_stage3.register_forward_hook(save_activation('interference_patterns'))
        self.source_localization_stage4.register_forward_hook(save_activation('source_localization'))
        
    def forward(self, x):
        """
        Forward pass for wave source localization.
        
        Args:
            x: Input wave field tensor [batch_size, 1, 128, 128]
            
        Returns:
            coordinates: Predicted (x, y) coordinates [batch_size, 2]
        """
        # Clear previous activations
        self.activations.clear()
        
        # Stage 0: Process input wave field
        x = self.wave_input_processor(x)
        
        # Stage 1: Extract basic wave features
        x = self.wave_feature_stage1(x)
        
        # Stage 2: Analyze complex wave patterns
        x = self.wave_pattern_stage2(x)
        
        # Stage 3: Understand interference patterns
        x = self.interference_stage3(x)
        
        # Stage 4: Localize wave source
        x = self.source_localization_stage4(x)
        
        # Global pooling to aggregate spatial information
        x = self.global_wave_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Predict coordinates
        coordinates = self.coordinate_predictor(x)
        
        # Ensure coordinates are in valid range [0, grid_size-1]
        coordinates = torch.sigmoid(coordinates) * (self.grid_size - 1)
        
        return coordinates
    
    def get_activation(self, layer_name):
        """Get stored activation for interpretability analysis."""
        return self.activations.get(layer_name, None)
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_wave_source_model(grid_size=128):
    """
    Factory function to create a WaveSourceMiniResNet model.
    
    Args:
        grid_size: Size of input grid (default 128 for 128x128 images)
        
    Returns:
        model: Initialized WaveSourceMiniResNet
    """
    model = WaveSourceMiniResNet(input_channels=1, grid_size=grid_size)
    
    # Initialize weights using modern best practices
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model