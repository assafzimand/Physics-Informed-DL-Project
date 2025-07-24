#!/usr/bin/env python3
"""
Fix and Continue T=250 Feature Analysis
Continues from where the previous script left off, with correct model path.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from models.wave_source_resnet import WaveSourceMiniResNet


def extract_t250_features():
    """Extract features using correct T=250 best model path."""
    print("🧠 Extracting T=250 features with correct model path...")
    
    # Correct T=250 model path
    model_path = "experiments/t250_cv_full/data/models/best_model.pt"
    samples_dir = "experiments/feature_analysis_t250/samples"
    output_dir = "experiments/feature_analysis_t250/activations"
    
    if not Path(model_path).exists():
        print(f"❌ T=250 model not found: {model_path}")
        return False
    
    # Load T=250 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        model_state_dict = checkpoint
        config = {}
    
    grid_size = config.get('grid_size', 128)
    model = WaveSourceMiniResNet(grid_size=grid_size)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ T=250 model loaded (grid_size: {grid_size})")
    
    # Get sample files
    raw_data_dir = Path(samples_dir) / "raw_data"
    sample_files = sorted(raw_data_dir.glob("*.npz"))
    
    print(f"📊 Processing {len(sample_files)} T=250 samples...")
    
    # Process each sample
    for sample_file in tqdm(sample_files, desc="Extracting features"):
        sample_data = np.load(sample_file)
        wave_field = torch.from_numpy(sample_data['wave_field']).float()
        coordinates = sample_data['coordinates']
        category = str(sample_data['category'])
        original_index = int(sample_data['original_index'])
        
        # Extract activations
        model.activations.clear()
        
        with torch.no_grad():
            wave_input = wave_field.unsqueeze(0).to(device)
            prediction = model(wave_input)
        
        # Get activations from all stages
        stage_names = ['basic_wave_features', 'complex_wave_patterns',
                      'interference_patterns', 'source_localization']
        
        activations = {}
        for i, stage_name in enumerate(stage_names):
            if stage_name in model.activations:
                activation = model.activations[stage_name].squeeze(0).cpu()
                activations[f'stage_{i+1}'] = activation
        
        # Add stage 0
        with torch.no_grad():
            stage_0 = model.wave_input_processor(wave_input)
            activations['stage_0'] = stage_0.squeeze(0).cpu()
        
        # Get top 9 features for each stage
        sample_id = sample_file.stem
        sample_info = {
            'path': str(sample_file),
            'original_index': original_index,
            'category': category,
            'true_coordinates': coordinates.tolist(),
            'predicted_coordinates': prediction.squeeze().cpu().tolist()
        }
        
        for stage_name, activation in activations.items():
            # Get top 9 features
            channels, height, width = activation.shape
            channel_strengths = activation.view(channels, -1).mean(dim=1)
            top_indices = torch.topk(channel_strengths, min(9, channels)).indices
            top_features = activation[top_indices]
            
            # Save features
            stage_dir = Path(output_dir) / stage_name
            feature_file = stage_dir / f"{sample_id}_features.npz"
            
            np.savez(feature_file,
                    features=top_features.numpy(),
                    indices=top_indices.tolist(),
                    activation_shape=list(activation.shape),
                    sample_info=sample_info)
    
    print(f"✅ T=250 feature extraction complete!")
    return True


def create_t250_comprehensive_plots():
    """Create comprehensive plots for all T=250 samples."""
    print("🎨 Creating T=250 comprehensive plots...")
    
    # Get sample IDs
    activations_dir = Path("experiments/feature_analysis_t250/activations/stage_0")
    sample_ids = []
    for feature_file in activations_dir.glob("*_features.npz"):
        sample_id = feature_file.name.replace('_features.npz', '')
        sample_ids.append(sample_id)
    
    sample_ids.sort()
    print(f"📊 Creating plots for {len(sample_ids)} T=250 samples...")
    
    # Process all samples
    for sample_id in tqdm(sample_ids, desc="Creating plots"):
        # Load features
        features, sample_info = load_t250_sample_features(sample_id)
        if sample_info is None:
            continue
            
        # Load original wave data
        wave_field, coordinates = load_t250_original_wave_data(sample_info)
        if wave_field is None:
            continue
        
        # Create comprehensive plot
        create_t250_comprehensive_plot(sample_id, features, sample_info, wave_field)
    
    print(f"✅ All T=250 comprehensive plots created!")


def load_t250_sample_features(sample_id):
    """Load T=250 sample features."""
    activations_dir = Path("experiments/feature_analysis_t250/activations")
    
    sample_features = {}
    sample_info = None
    
    for stage in range(5):
        stage_dir = activations_dir / f"stage_{stage}"
        feature_file = stage_dir / f"{sample_id}_features.npz"
        
        if feature_file.exists():
            data = np.load(feature_file, allow_pickle=True)
            sample_features[f'stage_{stage}'] = {
                'features': data['features'],
                'indices': data['indices'].tolist(),
                'activation_shape': data['activation_shape'].tolist()
            }
            
            if sample_info is None:
                sample_info = data['sample_info'].item()
    
    return sample_features, sample_info


def load_t250_original_wave_data(sample_info):
    """Load T=250 original wave data."""
    samples_dir = Path("experiments/feature_analysis_t250/samples/raw_data")
    original_idx = sample_info['original_index']
    
    for sample_file in samples_dir.glob(f"*_idx_{original_idx}.npz"):
        data = np.load(sample_file)
        return data['wave_field'], data['coordinates']
    
    return None, None


def create_t250_comprehensive_plot(sample_id, features, sample_info, wave_field):
    """Create comprehensive plot for T=250 sample."""
    category = sample_info['category']
    true_coords = sample_info['true_coordinates']
    pred_coords = sample_info['predicted_coordinates']
    
    # Create plot
    fig = plt.figure(figsize=(30, 15))
    
    # Title
    error = np.linalg.norm([pred_coords[0] - true_coords[0],
                           pred_coords[1] - true_coords[1]])
    fig.suptitle(f'T=250 Feature Analysis: {sample_id}\n'
                f'{category.capitalize()} Source | True: ({true_coords[0]:.1f}, {true_coords[1]:.1f}) | '
                f'Pred: ({pred_coords[0]:.1f}, {pred_coords[1]:.1f}) | Error: {error:.2f}px',
                fontsize=20, fontweight='bold', y=0.98)
    
    # Stage info
    stage_info = {
        0: "Stage 0 (32×32)",
        1: "Stage 1 (32×32)",
        2: "Stage 2 (16×16)",
        3: "Stage 3 (8×8)",
        4: "Stage 4 (4×4)"
    }
    
    # Plot original wave + all features
    # Row 1: Original + Stage 0
    ax = plt.subplot(5, 10, 1)
    final_wave = wave_field[-1]
    im = ax.imshow(final_wave, cmap='RdBu_r', aspect='equal')
    ax.set_title('Original Wave\n(Final timestep)', fontweight='bold', fontsize=12)
    
    ax.plot(true_coords[0], true_coords[1], 'go', markersize=10,
           markeredgecolor='black', markeredgewidth=2, label='True')
    ax.plot(pred_coords[0], pred_coords[1], 'r^', markersize=10,
           markeredgecolor='black', markeredgewidth=2, label='Pred')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Stage 0 features (columns 2-10)
    if 'stage_0' in features:
        stage_features = features['stage_0']
        feature_maps = stage_features['features']
        feature_indices = stage_features['indices']
        
        for feat_idx in range(9):
            col = feat_idx + 2
            ax = plt.subplot(5, 10, col)
            
            if feat_idx < len(feature_maps):
                feature_map = feature_maps[feat_idx]
                filter_idx = feature_indices[feat_idx]
                
                im = ax.imshow(feature_map, cmap='viridis', aspect='equal')
                if feat_idx == 4:
                    ax.set_title(f'{stage_info[0]}\nFilter #{filter_idx}', fontweight='bold', fontsize=10)
                else:
                    ax.set_title(f'F#{filter_idx}', fontweight='bold', fontsize=10)
                
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Stages 1-4 (rows 2-5)
    for stage in range(1, 5):
        row = stage
        stage_key = f'stage_{stage}'
        
        if stage_key in features:
            stage_features = features[stage_key]
            feature_maps = stage_features['features']
            feature_indices = stage_features['indices']
            
            for feat_idx in range(9):
                col = feat_idx + 1
                panel_idx = row * 10 + col + 1
                
                ax = plt.subplot(5, 10, panel_idx)
                
                if feat_idx < len(feature_maps):
                    feature_map = feature_maps[feat_idx]
                    filter_idx = feature_indices[feat_idx]
                    
                    im = ax.imshow(feature_map, cmap='viridis', aspect='equal')
                    
                    if feat_idx == 4:
                        ax.set_title(f'{stage_info[stage]}\nFilter #{filter_idx}', fontweight='bold', fontsize=10)
                    else:
                        ax.set_title(f'F#{filter_idx}', fontweight='bold', fontsize=10)
                    
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Hide first column for stages 1-4
            ax_first = plt.subplot(5, 10, row * 10 + 1)
            ax_first.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("experiments/feature_analysis_t250/plots/comprehensive")
    plot_file = output_dir / f"{sample_id}_comprehensive_features.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved: {plot_file}")


def main():
    """Continue T=250 analysis from where it left off."""
    print("🔧 Fixing and continuing T=250 analysis...")
    
    # Step 4: Extract features (with correct model path)
    extraction_success = extract_t250_features()
    if not extraction_success:
        return
    
    # Step 5: Create comprehensive plots
    create_t250_comprehensive_plots()
    
    print(f"\n🎉 T=250 feature analysis completed!")
    print(f"📁 Check results in: experiments/feature_analysis_t250/")
    print(f"📊 Check plots in: experiments/feature_analysis_t250/plots/comprehensive/")


if __name__ == "__main__":
    main() 