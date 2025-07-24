#!/usr/bin/env python3
"""
Complete T=250 Feature Analysis Workflow
Does everything: sample selection, dataset creation, feature extraction, and visualization.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append('.')
sys.path.append('src')

from data.wave_dataset import WaveDataset
from models.wave_source_resnet import WaveSourceMiniResNet


def create_t250_directories():
    """Create directory structure for T=250 analysis."""
    print("📁 Creating T=250 directory structure...")
    
    base_dir = Path("experiments/feature_analysis_t250")
    dirs_to_create = [
        base_dir,
        base_dir / "samples",
        base_dir / "samples" / "raw_data", 
        base_dir / "activations",
        base_dir / "plots",
        base_dir / "plots" / "comprehensive"
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create activation stage directories
    for stage in range(5):
        (base_dir / "activations" / f"stage_{stage}").mkdir(exist_ok=True)
    
    print(f"✅ Directory structure created: {base_dir}")
    return base_dir


def select_t250_samples(n_samples=20):
    """Select 20 diverse samples from T=250 validation dataset."""
    print("🔍 Selecting T=250 samples...")
    
    dataset_path = "data/wave_dataset_T250_validation.h5"
    output_dir = "experiments/feature_analysis_t250/samples"
    
    if not Path(dataset_path).exists():
        print(f"❌ T=250 validation dataset not found: {dataset_path}")
        return None
    
    # Load dataset
    dataset = WaveDataset(dataset_path)
    total_samples = len(dataset)
    print(f"📊 Loaded {total_samples} T=250 validation samples")
    
    # Categorize samples by source location
    def categorize_source_location(coords, grid_size=128):
        x, y = coords
        margin = 20
        
        # Corner check
        if (x < margin and y < margin) or \
           (x < margin and y > grid_size - margin) or \
           (x > grid_size - margin and y < margin) or \
           (x > grid_size - margin and y > grid_size - margin):
            return "corner"
        # Edge check
        elif x < margin or x > grid_size - margin or \
             y < margin or y > grid_size - margin:
            return "edge"
        # Center
        else:
            return "center"
    
    # Categorize all samples
    samples_by_category = {"corner": [], "edge": [], "center": []}
    
    for i in range(total_samples):
        _, coords = dataset[i]
        coords_np = coords.numpy()
        category = categorize_source_location(coords_np)
        samples_by_category[category].append({
            'index': i,
            'coords': coords_np.tolist(),
            'category': category
        })
    
    # Print distribution
    for cat, samples in samples_by_category.items():
        print(f"   {cat.capitalize()}: {len(samples)} samples")
    
    # Select diverse samples (same logic as T=500)
    np.random.seed(43)  # Different seed than T=500
    selected_samples = []
    
    # Select samples from each category
    corner_samples = samples_by_category["corner"]
    if len(corner_samples) >= 4:
        selected_corners = np.random.choice(len(corner_samples), 4, replace=False)
        for idx in selected_corners:
            selected_samples.append(corner_samples[idx])
    
    edge_samples = samples_by_category["edge"] 
    if len(edge_samples) >= 8:
        selected_edges = np.random.choice(len(edge_samples), 8, replace=False)
        for idx in selected_edges:
            selected_samples.append(edge_samples[idx])
    
    center_samples = samples_by_category["center"]
    if len(center_samples) >= 4:
        selected_centers = np.random.choice(len(center_samples), 4, replace=False)
        for idx in selected_centers:
            selected_samples.append(center_samples[idx])
    
    # Fill remaining with random samples
    used_indices = set(s['index'] for s in selected_samples)
    remaining_samples = []
    for category_samples in samples_by_category.values():
        for sample in category_samples:
            if sample['index'] not in used_indices:
                remaining_samples.append(sample)
    
    needed_random = max(0, n_samples - len(selected_samples))
    if len(remaining_samples) >= needed_random:
        random_indices = np.random.choice(len(remaining_samples), needed_random, replace=False)
        for idx in random_indices:
            selected_samples.append(remaining_samples[idx])
    
    # Sort by index
    selected_samples.sort(key=lambda x: x['index'])
    
    print(f"✅ Selected {len(selected_samples)} T=250 samples")
    
    # Save metadata and raw data
    output_dir = Path(output_dir)
    
    # Save metadata
    metadata = {
        'total_samples': len(selected_samples),
        'source_dataset': str(dataset_path),
        'samples': selected_samples,
        'dataset_type': 'T=250'
    }
    
    metadata_file = output_dir / "sample_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save raw data
    raw_data_dir = output_dir / "raw_data"
    for i, sample_info in enumerate(selected_samples):
        sample_idx = sample_info['index']
        wave_data, coords = dataset[sample_idx]
        
        sample_file = raw_data_dir / f"sample_{i:02d}_idx_{sample_idx}.npz"
        np.savez(sample_file,
                wave_field=wave_data.numpy(),
                coordinates=coords.numpy(),
                original_index=sample_idx,
                category=sample_info['category'])
    
    print(f"✅ T=250 samples saved to: {output_dir}")
    return selected_samples


def create_t250_dataset():
    """Create consolidated T=250 analysis dataset."""
    print("📦 Creating T=250 consolidated dataset...")
    
    samples_dir = Path("experiments/feature_analysis_t250/samples")
    output_file = "data/wave_dataset_t250_analysis_20samples.h5"
    raw_data_dir = samples_dir / "raw_data"
    
    # Get sample files
    sample_files = sorted(raw_data_dir.glob("*.npz"))
    n_samples = len(sample_files)
    
    if n_samples == 0:
        print("❌ No T=250 sample files found!")
        return None
    
    # Load first sample to get dimensions
    first_sample = np.load(sample_files[0])
    wave_shape = first_sample['wave_field'].shape
    
    print(f"📊 Creating T=250 dataset: {wave_shape} for {n_samples} samples")
    
    # Create HDF5 dataset
    with h5py.File(output_file, 'w') as f:
        wave_data = f.create_dataset('wave_fields',
                                   shape=(n_samples, *wave_shape),
                                   dtype=np.float32,
                                   compression='gzip')
        
        coordinates = f.create_dataset('coordinates',
                                     shape=(n_samples, 2),
                                     dtype=np.float32)
        
        original_indices = f.create_dataset('original_indices',
                                          shape=(n_samples,),
                                          dtype=np.int32)
        
        categories = f.create_dataset('categories',
                                    shape=(n_samples,),
                                    dtype=h5py.string_dtype())
        
        # Fill datasets
        for i, sample_file in enumerate(sample_files):
            sample_data = np.load(sample_file)
            wave_data[i] = sample_data['wave_field']
            coordinates[i] = sample_data['coordinates']
            original_indices[i] = sample_data['original_index']
            categories[i] = str(sample_data['category'])
        
        # Add metadata
        f.attrs['total_samples'] = n_samples
        f.attrs['source_dataset'] = 'data/wave_dataset_T250_validation.h5'
        f.attrs['timesteps'] = wave_shape[0]
        f.attrs['grid_size'] = wave_shape[1]
        f.attrs['wave_speed'] = 16.7  # T=250 wave speed
        f.attrs['description'] = 'T=250 Analysis dataset: 20 diverse samples'
    
    print(f"✅ T=250 consolidated dataset created: {output_file}")
    return output_file


def extract_t250_features():
    """Extract features using T=250 best model."""
    print("🧠 Extracting T=250 features...")
    
    # NOTE: Model path is hardcoded - should be updated if different best fold
    model_path = "experiments/t250_cv_full/data/models/t250_cv_full_5fold_75epochs_fold_2_best.pth"
    samples_dir = "experiments/feature_analysis_t250/samples"
    output_dir = "experiments/feature_analysis_t250/activations"
    
    if not Path(model_path).exists():
        print(f"❌ T=250 model not found: {model_path}")
        return None
    
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
    
    # Save summary
    summary_file = Path(output_dir) / "extraction_summary.json"
    summary_data = {
        'model_path': model_path,
        'n_features_per_stage': 9,
        'total_samples': len(sample_files),
        'dataset_type': 'T=250'
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
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
    
    # Create comprehensive plots (similar to T=500 but with T=250 paths)
    for i, sample_id in enumerate(tqdm(sample_ids, desc="Creating plots")):
        # Load features
        features, sample_info = load_t250_sample_features(sample_id)
        if sample_info is None:
            continue
            
        # Load original wave data
        wave_field, coordinates = load_t250_original_wave_data(sample_info)
        if wave_field is None:
            continue
        
        # Create plot (same logic as T=500 comprehensive plot)
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
    
    # Plot original wave + all features (same layout as T=500)
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


def main():
    """Run complete T=250 feature analysis."""
    print("🚀 Starting complete T=250 feature analysis workflow...\n")
    
    # Step 1: Create directories
    base_dir = create_t250_directories()
    
    # Step 2: Select samples
    selected_samples = select_t250_samples(n_samples=20)
    if selected_samples is None:
        return
    
    # Step 3: Create consolidated dataset
    dataset_file = create_t250_dataset()
    if dataset_file is None:
        return
    
    # Step 4: Extract features
    extraction_success = extract_t250_features()
    if not extraction_success:
        return
    
    # Step 5: Create comprehensive plots
    create_t250_comprehensive_plots()
    
    print(f"\n🎉 Complete T=250 feature analysis finished!")
    print(f"📁 All results saved in: {base_dir}")
    print(f"📊 Check comprehensive plots in: {base_dir}/plots/comprehensive/")


if __name__ == "__main__":
    main() 