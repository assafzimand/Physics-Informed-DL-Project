"""
Deconvolutional blocks for WaveSourceMiniResNet.
Implements reverse versions of our wave-specific ResNet blocks.
"""

import torch
import torch.nn as nn

from .reverse_operations import (
    ReverseReLU, ReverseBatchNorm2d, ReverseConv2d, 
    ReverseMaxPool2d, ReverseSkipConnection
)


class ReverseWaveResidualBlock(nn.Module):
    """
    Reverse version of WaveResidualBlock.
    Reconstructs input from the output of a WaveResidualBlock.
    """
    
    def __init__(self, forward_block, layer_name: str):
        super(ReverseWaveResidualBlock, self).__init__()
        
        self.layer_name = layer_name
        
        # Extract components from forward block
        self.in_channels = forward_block.wave_feature_conv2.out_channels
        self.out_channels = forward_block.wave_feature_conv1.in_channels
        
        # Create reverse operations in reverse order
        # Forward path: conv1 -> bn1 -> relu -> conv2 -> bn2 -> skip -> relu
        # Reverse path: reverse_relu -> reverse_skip -> reverse_bn2 -> 
        #               reverse_conv2 -> reverse_relu -> reverse_bn1 -> 
        #               reverse_conv1
        
        # Final ReLU (reverse first)
        self.reverse_final_relu = ReverseReLU()
        
        # Skip connection reversal
        skip_conn = forward_block.skip_connection
        self.reverse_skip = ReverseSkipConnection(skip_conn)
        
        # Second conv block (bn2 -> conv2)
        self.reverse_bn2 = ReverseBatchNorm2d(forward_block.wave_feature_bn2)
        self.reverse_conv2 = ReverseConv2d(forward_block.wave_feature_conv2)
        
        # Intermediate ReLU
        self.reverse_intermediate_relu = ReverseReLU()
        
        # First conv block (bn1 -> conv1)
        self.reverse_bn1 = ReverseBatchNorm2d(forward_block.wave_feature_bn1)
        self.reverse_conv1 = ReverseConv2d(forward_block.wave_feature_conv1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the WaveResidualBlock operations.
        
        Forward block does: 
        1. residual = x
        2. out = relu(bn1(conv1(x)))
        3. out = bn2(conv2(out))
        4. out = out + skip_connection(residual)
        5. out = relu(out)
        
        Reverse does the opposite in reverse order.
        """
        
        # Step 1: Reverse final ReLU
        out = self.reverse_final_relu(x)
        
        # Step 2: Handle skip connection properly
        # In forward ResNet: out = main_path + skip_connection(residual)
        # For reverse: we need to isolate the main_path contribution
        
        # In deconvent, we typically focus on the main path
        # The skip connection just helps with gradient flow, not feature visualization
        # So we use the output directly as our main path approximation
        main_path = out
        
        # Alternative: If we had the original residual input, we could:
        # main_path = out - self.reverse_skip_connection(residual)
        # But in deconvent visualization, main path focus is standard
        
        # Step 3: Reverse second conv block
        out = self.reverse_bn2(main_path)
        out = self.reverse_conv2(out)
        
        # Step 4: Reverse intermediate ReLU
        out = self.reverse_intermediate_relu(out)
        
        # Step 5: Reverse first conv block  
        out = self.reverse_bn1(out)
        out = self.reverse_conv1(out)
        
        return out


class ReverseWaveInputProcessor(nn.Module):
    """
    Reverse the initial wave input processing stage.
    Reverses: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    """
    
    def __init__(self, forward_processor):
        super(ReverseWaveInputProcessor, self).__init__()
        
        # Extract components from forward processor (nn.Sequential)
        components = list(forward_processor.children())
        
        # Create reverse components in reverse order
        self.reverse_maxpool = ReverseMaxPool2d(components[3])  # MaxPool2d
        self.reverse_relu = ReverseReLU()                       # ReLU
        # BatchNorm2d
        self.reverse_bn = ReverseBatchNorm2d(components[1])     
        self.reverse_conv = ReverseConv2d(components[0])        # Conv2d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the input processing operations."""
        out = self.reverse_maxpool(x)
        out = self.reverse_relu(out)
        out = self.reverse_bn(out)
        out = self.reverse_conv(out)
        return out


class WaveDeconvolutionalNetwork(nn.Module):
    """
    Full deconvolutional network for WaveSourceMiniResNet.
    Implements the Zeiler & Fergus approach for visualizing learned features.
    """
    
    def __init__(self, forward_model):
        super(WaveDeconvolutionalNetwork, self).__init__()
        
        self.forward_model = forward_model
        
        # Create reverse stages in reverse order
        # Forward: input -> stage0 -> stage1 -> stage2 -> stage3 -> stage4 -> 
        #          global_pool -> fc
        # Reverse: fc -> global_pool -> stage4 -> stage3 -> stage2 -> stage1 -> 
        #          stage0 -> output
        
        # Reverse global pooling (approximate with upsampling)
        self.reverse_global_pool = nn.Upsample(size=(4, 4), mode='nearest')
        
        # Reverse Stage 4: Source localization features
        stage4 = forward_model.source_localization_stage4
        self.reverse_stage4_block1 = ReverseWaveResidualBlock(
            stage4[1], "stage4_block1"
        )
        self.reverse_stage4_block0 = ReverseWaveResidualBlock(
            stage4[0], "stage4_block0"
        )
        
        # Reverse Stage 3: Interference pattern understanding  
        stage3 = forward_model.interference_stage3
        self.reverse_stage3_block1 = ReverseWaveResidualBlock(
            stage3[1], "stage3_block1"
        )
        self.reverse_stage3_block0 = ReverseWaveResidualBlock(
            stage3[0], "stage3_block0"
        )
        
        # Reverse Stage 2: Complex wave pattern analysis
        stage2 = forward_model.wave_pattern_stage2
        self.reverse_stage2_block1 = ReverseWaveResidualBlock(
            stage2[1], "stage2_block1"
        )
        self.reverse_stage2_block0 = ReverseWaveResidualBlock(
            stage2[0], "stage2_block0"
        )
        
        # Reverse Stage 1: Basic wave feature detection
        stage1 = forward_model.wave_feature_stage1
        self.reverse_stage1_block1 = ReverseWaveResidualBlock(
            stage1[1], "stage1_block1"
        )
        self.reverse_stage1_block0 = ReverseWaveResidualBlock(
            stage1[0], "stage1_block0"
        )
        
        # Reverse Stage 0: Initial wave pattern extraction
        self.reverse_stage0 = ReverseWaveInputProcessor(
            forward_model.wave_input_processor
        )
        
    def forward_to_stage(self, x: torch.Tensor, 
                        target_stage: int) -> torch.Tensor:
        """
        Run forward pass up to a specific stage.
        
        Args:
            x: Input wave field
            target_stage: Stage to stop at (0-4)
            
        Returns:
            Activations at the target stage
        """
        with torch.no_grad():
            # Stage 0: Initial processing
            out = self.forward_model.wave_input_processor(x)
            if target_stage == 0:
                return out
                
            # Stage 1: Basic features
            out = self.forward_model.wave_feature_stage1(out)
            if target_stage == 1:
                return out
                
            # Stage 2: Complex patterns
            out = self.forward_model.wave_pattern_stage2(out)
            if target_stage == 2:
                return out
                
            # Stage 3: Interference patterns
            out = self.forward_model.interference_stage3(out)
            if target_stage == 3:
                return out
                
            # Stage 4: Source localization
            out = self.forward_model.source_localization_stage4(out)
            if target_stage == 4:
                return out
                
        raise ValueError(f"Invalid target_stage: {target_stage}")
    
    def reverse_from_stage(self, activations: torch.Tensor, 
                          source_stage: int) -> torch.Tensor:
        """
        Run reverse pass from a specific stage back to input.
        
        Args:
            activations: Activations from the source stage
            source_stage: Stage to start reverse from (0-4)
            
        Returns:
            Reconstructed input
        """
        out = activations
        
        # Reverse from the specified stage back to input
        if source_stage >= 4:
            out = self.reverse_stage4_block1(out)
            out = self.reverse_stage4_block0(out)
            
        if source_stage >= 3:
            out = self.reverse_stage3_block1(out)
            out = self.reverse_stage3_block0(out)
            
        if source_stage >= 2:
            out = self.reverse_stage2_block1(out)
            out = self.reverse_stage2_block0(out)
            
        if source_stage >= 1:
            out = self.reverse_stage1_block1(out)
            out = self.reverse_stage1_block0(out)
            
        if source_stage >= 0:
            out = self.reverse_stage0(out)
            
        return out
    
    def visualize_filter(self, input_sample: torch.Tensor, target_stage: int, 
                        filter_idx: int) -> torch.Tensor:
        """
        Visualize what a specific filter detects using deconvolutional network.
        
        Args:
            input_sample: Input wave field [1, 1, 128, 128]
            target_stage: Stage containing the filter of interest (1-4)
            filter_idx: Index of the filter to visualize
            
        Returns:
            Reconstructed input showing what the filter detects [1, 1, 128, 128]
        """
        # Get activations at target stage
        stage_activations = self.forward_to_stage(input_sample, target_stage)
        
        # Isolate the specific filter (zero out all others)
        isolated_activations = torch.zeros_like(stage_activations)
        isolated_activations[0, filter_idx, :, :] = (
            stage_activations[0, filter_idx, :, :]
        )
        
        # Reverse back to input
        reconstructed_input = self.reverse_from_stage(
            isolated_activations, target_stage
        )
        
        return reconstructed_input 