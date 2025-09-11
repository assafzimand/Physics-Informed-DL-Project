"""
Wave Source Localization Mini-ResNet

A custom ResNet-based CNN for predicting wave source coordinates from wave
interference patterns. The model expects inputs to be PRE-NORMALIZED using the
training dataset's mean/std (handled externally by callers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WaveResidualBlock(nn.Module):
    """
    Residual block optimized for wave pattern analysis.

    Structure:
    - Conv(3x3) + BN + ReLU
    - Conv(3x3) + BN
    - Optional projection on the skip path when spatial/channel dims change
    - Residual add + ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(WaveResidualBlock, self).__init__()

        # Main convolution path
        self.wave_feature_conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.wave_feature_bn1 = nn.BatchNorm2d(out_channels)

        self.wave_feature_conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.wave_feature_bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (projection if needed)
        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.wave_feature_bn1(self.wave_feature_conv1(x)))
        out = self.wave_feature_bn2(self.wave_feature_conv2(out))

        out += self.skip_connection(residual)
        out = F.relu(out)
        return out


class WaveSourceMiniResNet(nn.Module):
    """
    Mini-ResNet for wave source localization.

    - Initial feature extraction from wave patterns
    - 4 stages of residual blocks with increasing channel depth
    - Global pooling and coordinate regression head
    - Outputs (x, y) coordinates in range [0, grid_size-1]

    IMPORTANT: Input must be normalized by training mean/std before calling forward.
    """

    def __init__(self, input_channels: int = 1, grid_size: int = 128) -> None:
        super(WaveSourceMiniResNet, self).__init__()

        self.grid_size = grid_size

        # Stage 0: Initial wave pattern extraction
        self.wave_input_processor = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Output: 32 x 32 x 32

        # Stage 1
        self.wave_feature_stage1 = nn.Sequential(
            WaveResidualBlock(32, 32, stride=1), WaveResidualBlock(32, 32, stride=1)
        )
        # Output: 32 x 32 x 32

        # Stage 2
        self.wave_pattern_stage2 = nn.Sequential(
            WaveResidualBlock(32, 64, stride=2), WaveResidualBlock(64, 64, stride=1)
        )
        # Output: 64 x 16 x 16

        # Stage 3
        self.interference_stage3 = nn.Sequential(
            WaveResidualBlock(64, 128, stride=2), WaveResidualBlock(128, 128, stride=1)
        )
        # Output: 128 x 8 x 8

        # Stage 4
        self.source_localization_stage4 = nn.Sequential(
            WaveResidualBlock(128, 256, stride=2), WaveResidualBlock(256, 256, stride=1)
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
            nn.Linear(64, 2),  # (x, y) coordinates
        )

        # Store intermediate activations for interpretability
        self.activations: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers forward hooks on key stages for interpretability analysis."""

        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()

            return hook

        self.wave_feature_stage1.register_forward_hook(
            save_activation("basic_wave_features")
        )
        self.wave_pattern_stage2.register_forward_hook(
            save_activation("complex_wave_patterns")
        )
        self.interference_stage3.register_forward_hook(
            save_activation("interference_patterns")
        )
        self.source_localization_stage4.register_forward_hook(
            save_activation("source_localization")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input wave field tensor of shape [batch_size, 1, 128, 128], already normalized.

        Returns:
            Predicted (x, y) coordinates in [0, grid_size-1] with shape [batch_size, 2].
        """
        # Clear previous activations
        self.activations.clear()

        x = self.wave_input_processor(x)
        x = self.wave_feature_stage1(x)
        x = self.wave_pattern_stage2(x)
        x = self.interference_stage3(x)
        x = self.source_localization_stage4(x)

        x = self.global_wave_pool(x)
        x = x.view(x.size(0), -1)

        coordinates = self.coordinate_predictor(x)
        coordinates = torch.sigmoid(coordinates) * (self.grid_size - 1)
        return coordinates

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Returns stored activation for a named layer, if available."""
        return self.activations.get(layer_name, None)

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_wave_source_model(grid_size: int = 128) -> WaveSourceMiniResNet:
    """
    Factory function to create and initialize a WaveSourceMiniResNet.

    Args:
        grid_size: Size of input grid (default 128 for 128x128 images).

    Returns:
        Initialized WaveSourceMiniResNet model.
    """
    model = WaveSourceMiniResNet(input_channels=1, grid_size=grid_size)

    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    return model
