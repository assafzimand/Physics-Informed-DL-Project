#!/usr/bin/env python3
"""
Comprehensive Feature Visualization for WaveSourceMiniResNet
Shows original wave + all 9 strongest features from all 5 stages in one plot.
WITH ORIGINAL COORDINATE MAPPING - shows where each feature maps back to in the input image.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')


def map_to_original_coordinates(feature_x, feature_y, stage):
    """
    Map feature map coordinates back to original 128x128 image coordinates.
    
    Args:
        feature_x, feature_y: Coordinates in the feature map
        stage: Stage number (0-4)
    
    Returns:
        (orig_x, orig_y): Corresponding coordinates in original image
    """
    # Downsampling factors for each stage
    downsampling_factors = {
        0: 4,   # 128→32
        1: 4,   # 32→32 (same as stage 0)  
        2: 8,   # 128→16
        3: 16,  # 128→8
        4: 32   # 128→4
    }
    
    factor = downsampling_factors[stage]
    
    # Map to center of receptive field in original image
    orig_x = int(feature_x * factor + factor // 2)
    orig_y = int(feature_y * factor + factor // 2)
    
    # Ensure coordinates are within bounds
    orig_x = max(0, min(127, orig_x))
    orig_y = max(0, min(127, orig_y))
    
    return orig_x, orig_y


def load_sample_features(sample_id, activations_dir):
    """Load features for a specific sample across all stages."""
    activations_dir = Path(activations_dir)
    
    sample_features = {}
    sample_info = None
    
    # Load features from each stage
    for stage in range(5):  # Stages 0-4
        stage_dir = activations_dir / f"stage_{stage}"
        feature_file = stage_dir / f"{sample_id}_features.npz"
        
        if feature_file.exists():
            data = np.load(feature_file, allow_pickle=True)
            sample_features[f'stage_{stage}'] = {
                'features': data['features'],  # Shape: [9, height, width]
                'indices': data['indices'].tolist(),
                'activation_shape': data['activation_shape'].tolist()
            }
            
            # Get sample info from first stage
            if sample_info is None:
                sample_info = data['sample_info'].item()
    
    return sample_features, sample_info


def load_original_wave_data(sample_info):
    """Load the original wave data for a sample."""
    samples_dir = Path("experiments/feature_analysis/samples/raw_data")
    original_idx = sample_info['original_index']
    
    # Find the sample file
    for sample_file in samples_dir.glob(f"*_idx_{original_idx}.npz"):
        data = np.load(sample_file)
        return data['wave_field'], data['coordinates']
    
    return None, None


def create_comprehensive_visualization(sample_id, save_plot=True):
    """Create comprehensive visualization with original coordinate mapping."""
    
    # Load features and sample info
    features, sample_info = load_sample_features(sample_id, "experiments/feature_analysis/activations")
    
    if sample_info is None:
        print(f"❌ Could not load features for sample {sample_id}")
        return
    
    # Load original wave data
    wave_field, coordinates = load_original_wave_data(sample_info)
    if wave_field is None:
        print(f"❌ Could not load original wave data for sample {sample_id}")
        return
    
    # Get sample metadata
    category = sample_info['category']
    true_coords = sample_info['true_coordinates']
    pred_coords = sample_info['predicted_coordinates']
    
    # Create the comprehensive visualization
    fig = plt.figure(figsize=(30, 15))
    
    # Title with sample info
    error = np.linalg.norm([pred_coords[0] - true_coords[0], 
                           pred_coords[1] - true_coords[1]])
    fig.suptitle(f'Comprehensive Feature Analysis: {sample_id}\n'
                f'{category.capitalize()} Source | True: ({true_coords[0]:.1f}, {true_coords[1]:.1f}) | '
                f'Pred: ({pred_coords[0]:.1f}, {pred_coords[1]:.1f}) | Error: {error:.2f}px', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Stage information for titles
    stage_info = {
        0: "Stage 0 (32×32)",
        1: "Stage 1 (32×32)", 
        2: "Stage 2 (16×16)",
        3: "Stage 3 (8×8)",
        4: "Stage 4 (4×4)"
    }
    
    # Row 1: Original wave + Stage 0 features
    # Column 1: Original wave
    ax = plt.subplot(5, 10, 1)
    final_wave = wave_field[-1]  # Final timestep
    im = ax.imshow(final_wave, cmap='RdBu_r', aspect='equal')
    ax.set_title('Original Wave\n(Final timestep)', fontweight='bold', fontsize=12)
    
    # Add source location markers
    ax.plot(true_coords[0], true_coords[1], 'go', markersize=10, 
           markeredgecolor='black', markeredgewidth=2, label='True')
    ax.plot(pred_coords[0], pred_coords[1], 'r^', markersize=10,
           markeredgecolor='black', markeredgewidth=2, label='Pred')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Columns 2-10: Stage 0 features
    if 'stage_0' in features:
        stage_features = features['stage_0']
        feature_maps = stage_features['features']
        feature_indices = stage_features['indices']
        
        for feat_idx in range(9):
            col = feat_idx + 2  # Columns 2-10
            ax = plt.subplot(5, 10, col)
            
            if feat_idx < len(feature_maps):
                feature_map = feature_maps[feat_idx]
                filter_idx = feature_indices[feat_idx]
                
                # Find peak activation coordinates
                peak_y, peak_x = np.unravel_index(np.argmax(feature_map), feature_map.shape)
                
                # Map to original image coordinates
                orig_x, orig_y = map_to_original_coordinates(peak_x, peak_y, stage=0)
                
                im = ax.imshow(feature_map, cmap='viridis', aspect='equal')
                if feat_idx == 4:  # Middle feature - add stage title
                    ax.set_title(f'{stage_info[0]}\nF#{filter_idx} @({peak_x},{peak_y})→({orig_x},{orig_y})', 
                                fontweight='bold', fontsize=9)
                else:
                    ax.set_title(f'F#{filter_idx} @({peak_x},{peak_y})→({orig_x},{orig_y})', 
                                fontweight='bold', fontsize=9)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
            if feat_idx < len(feature_maps):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Rows 2-5: Stages 1-4 features
    for stage in range(1, 5):
        row = stage  # Rows 1-4 (0-indexed)
        stage_key = f'stage_{stage}'
        
        if stage_key in features:
            stage_features = features[stage_key]
            feature_maps = stage_features['features']
            feature_indices = stage_features['indices']
            
            # All 10 columns for this stage's features
            for feat_idx in range(9):
                col = feat_idx + 1  # Columns 1-9
                panel_idx = row * 10 + col + 1  # Calculate subplot index
                
                ax = plt.subplot(5, 10, panel_idx)
                
                if feat_idx < len(feature_maps):
                    feature_map = feature_maps[feat_idx]
                    filter_idx = feature_indices[feat_idx]
                    
                    # Find peak activation coordinates
                    peak_y, peak_x = np.unravel_index(np.argmax(feature_map), feature_map.shape)
                    
                    # Map to original image coordinates
                    orig_x, orig_y = map_to_original_coordinates(peak_x, peak_y, stage=stage)
                    
                    im = ax.imshow(feature_map, cmap='viridis', aspect='equal')
                    
                    if feat_idx == 4:  # Middle feature - add stage title
                        ax.set_title(f'{stage_info[stage]}\nF#{filter_idx} @({peak_x},{peak_y})→({orig_x},{orig_y})', 
                                    fontweight='bold', fontsize=9)
                    else:
                        ax.set_title(f'F#{filter_idx} @({peak_x},{peak_y})→({orig_x},{orig_y})', 
                                    fontweight='bold', fontsize=9)
                    
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Hide the first column for stages 1-4 (we only use columns 2-10)
            ax_first = plt.subplot(5, 10, row * 10 + 1)
            ax_first.axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        # Save plot
        output_dir = Path("experiments/feature_analysis/plots/comprehensive_with_origins")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_dir / f"{sample_id}_comprehensive_features.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive visualization saved: {plot_file}")
    
    plt.close()
    return fig


def main():
    """Main function to create comprehensive feature visualization with origin mapping."""
    print("🎨 Creating comprehensive feature visualization with original coordinate mapping...")
    
    # Get all sample IDs
    sample_ids = []
    activations_dir = Path("experiments/feature_analysis/activations/stage_0")
    for feature_file in activations_dir.glob("*_features.npz"):
        sample_id = feature_file.name.replace('_features.npz', '')
        sample_ids.append(sample_id)
    
    sample_ids.sort()
    print(f"📊 Found {len(sample_ids)} samples to process")
    
    # Process all samples
    for i, sample_id in enumerate(sample_ids):
        print(f"🎯 Processing sample {i+1}/{len(sample_ids)}: {sample_id}")
        create_comprehensive_visualization(sample_id, save_plot=True)
    
    print(f"\n🎉 All comprehensive visualizations complete!")
    print(f"📁 Check plots in: experiments/feature_analysis/plots/comprehensive_with_origins/")


if __name__ == "__main__":
    main() 