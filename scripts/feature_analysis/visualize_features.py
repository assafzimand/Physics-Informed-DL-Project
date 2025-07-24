#!/usr/bin/env python3
"""
Feature Visualization for WaveSourceMiniResNet
Visualizes what wave patterns activate specific filters most strongly.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('src')


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


def create_feature_visualization(sample_id, stage, n_features=9, save_plot=True):
    """Create visualization for one sample's features at one stage."""
    
    # Load features and sample info
    features, sample_info = load_sample_features(sample_id, "experiments/feature_analysis/activations")
    
    if sample_info is None:
        print(f"❌ Could not load features for sample {sample_id}")
        return
    
    if f'stage_{stage}' not in features:
        print(f"❌ Stage {stage} not found for sample {sample_id}")
        return
    
    # Load original wave data
    wave_field, coordinates = load_original_wave_data(sample_info)
    if wave_field is None:
        print(f"❌ Could not load original wave data for sample {sample_id}")
        return
    
    # Get stage features
    stage_features = features[f'stage_{stage}']
    feature_maps = stage_features['features']  # Shape: [9, height, width]
    feature_indices = stage_features['indices']
    
    # Get sample metadata
    category = sample_info['category']
    true_coords = sample_info['true_coordinates']
    pred_coords = sample_info['predicted_coordinates']
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Title with sample info
    fig.suptitle(f'Feature Analysis: {sample_id} (Stage {stage})\n'
                f'{category.capitalize()} Source | True: ({true_coords[0]:.1f}, {true_coords[1]:.1f}) | '
                f'Pred: ({pred_coords[0]:.1f}, {pred_coords[1]:.1f})', 
                fontsize=16, fontweight='bold')
    
    # Plot layout: 2 rows
    # Row 1: Original wave field (final timestep) + first 4 features  
    # Row 2: Remaining 5 features
    
    # Row 1: Original wave + 4 features
    for i in range(5):
        ax = plt.subplot(2, 5, i + 1)
        
        if i == 0:
            # Show original wave field (final timestep)
            final_wave = wave_field[-1]  # Last timestep
            im = ax.imshow(final_wave, cmap='RdBu_r', aspect='equal')
            ax.set_title(f'Original Wave\n(Final timestep)', fontweight='bold')
            
            # Add source location marker
            ax.plot(true_coords[0], true_coords[1], 'go', markersize=8, 
                   markeredgecolor='black', markeredgewidth=2, label='True Source')
            ax.plot(pred_coords[0], pred_coords[1], 'r^', markersize=8,
                   markeredgecolor='black', markeredgewidth=2, label='Predicted')
            ax.legend(loc='upper right', fontsize=8)
            
        else:
            # Show feature map
            feature_idx = i - 1
            if feature_idx < len(feature_maps):
                feature_map = feature_maps[feature_idx]
                filter_idx = feature_indices[feature_idx]
                
                im = ax.imshow(feature_map, cmap='viridis', aspect='equal')
                ax.set_title(f'Feature {feature_idx + 1}\n(Filter #{filter_idx})', fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: Remaining 5 features
    for i in range(5):
        ax = plt.subplot(2, 5, i + 6)
        feature_idx = i + 4
        
        if feature_idx < len(feature_maps):
            feature_map = feature_maps[feature_idx]
            filter_idx = feature_indices[feature_idx]
            
            im = ax.imshow(feature_map, cmap='viridis', aspect='equal')
            ax.set_title(f'Feature {feature_idx + 1}\n(Filter #{filter_idx})', fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        # Save plot
        output_dir = Path("experiments/feature_analysis/plots/individual")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_dir / f"{sample_id}_stage_{stage}_features.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Feature visualization saved: {plot_file}")
    
    plt.show()
    
    return fig


def create_stage_comparison(sample_id, save_plot=True):
    """Create a comparison showing how features evolve across stages."""
    
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
    
    # Create the visualization - 6 columns (original + 5 stages), 3 rows (3 top features per stage)
    fig = plt.figure(figsize=(24, 12))
    
    # Title
    fig.suptitle(f'Stage Evolution: {sample_id}\n'
                f'{category.capitalize()} Source | True: ({true_coords[0]:.1f}, {true_coords[1]:.1f}) | '
                f'Pred: ({pred_coords[0]:.1f}, {pred_coords[1]:.1f})', 
                fontsize=16, fontweight='bold')
    
    # Column 1: Original wave
    for row in range(3):
        ax = plt.subplot(3, 6, row * 6 + 1)
        
        if row == 1:  # Middle row - show the original wave
            final_wave = wave_field[-1]
            im = ax.imshow(final_wave, cmap='RdBu_r', aspect='equal')
            ax.set_title('Original\nWave', fontweight='bold')
            
            # Add source markers
            ax.plot(true_coords[0], true_coords[1], 'go', markersize=6, 
                   markeredgecolor='black', markeredgewidth=1.5)
            ax.plot(pred_coords[0], pred_coords[1], 'r^', markersize=6,
                   markeredgecolor='black', markeredgewidth=1.5)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Columns 2-6: Stages 0-4 (top 3 features each)
    for stage in range(5):
        stage_key = f'stage_{stage}'
        
        if stage_key in features:
            stage_features = features[stage_key]
            feature_maps = stage_features['features']  # Shape: [9, height, width]
            feature_indices = stage_features['indices']
            
            # Show top 3 features for this stage
            for row in range(3):
                ax = plt.subplot(3, 6, row * 6 + stage + 2)
                
                if row < len(feature_maps):
                    feature_map = feature_maps[row]
                    filter_idx = feature_indices[row]
                    
                    im = ax.imshow(feature_map, cmap='viridis', aspect='equal')
                    
                    if row == 0:  # Top row - add stage title
                        ax.set_title(f'Stage {stage}\nF#{filter_idx}', fontweight='bold')
                    else:
                        ax.set_title(f'F#{filter_idx}', fontweight='bold')
                    
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_plot:
        # Save plot
        output_dir = Path("experiments/feature_analysis/plots/summary")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_dir / f"{sample_id}_stage_evolution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Stage evolution plot saved: {plot_file}")
    
    plt.show()
    
    return fig


def main():
    """Main function to create feature visualizations."""
    print("🎨 Starting feature visualization...")
    
    # Load extraction summary to see what samples we have
    summary_file = "experiments/feature_analysis/activations/extraction_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"📊 Found {summary['total_samples']} samples with {summary['n_features_per_stage']} features per stage")
    
    # Get list of sample IDs
    sample_ids = []
    activations_dir = Path("experiments/feature_analysis/activations/stage_0")
    for feature_file in activations_dir.glob("*_features.npz"):
        sample_id = feature_file.name.replace('_features.npz', '')
        sample_ids.append(sample_id)
    
    sample_ids.sort()
    
    # For demo, visualize first few samples
    print(f"🎯 Creating visualizations for first 3 samples...")
    
    for i, sample_id in enumerate(sample_ids[:3]):
        print(f"\n📸 Sample {i+1}: {sample_id}")
        
        # Create stage evolution plot
        create_stage_comparison(sample_id, save_plot=True)
        
        # Create detailed view for stage 2 (complex wave patterns)
        create_feature_visualization(sample_id, stage=2, save_plot=True)
    
    print(f"\n🎉 Feature visualization complete!")
    print(f"📁 Check plots in: experiments/feature_analysis/plots/")


if __name__ == "__main__":
    main() 